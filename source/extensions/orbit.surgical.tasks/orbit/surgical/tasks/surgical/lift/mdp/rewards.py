# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PhasedPickupRewardWrapper:
    def __init__(self, ee_frame_cfg, object_cfg, robot_cfg):
        self.ee_frame_cfg = ee_frame_cfg
        self.object_cfg = object_cfg
        self.robot_cfg = robot_cfg
        self.phase_flags = None
        self.__name__ = "PhasedPickupRewardWrapper"

    def __init__flags(self, env):
        num_envs = env.num_envs
        device = env.device
        return {
            "gripped": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "z_hold_counter": torch.zeros(num_envs, dtype=torch.float32, device=device)
        }

    def __call__(self, env):
        if self.phase_flags is None:
            self.phase_flags = self.__init__flags(env)

        reward, self.phase_flags = shaped_pickup_reward(
            env,
            self.phase_flags,
            self.ee_frame_cfg,
            self.object_cfg,
            self.robot_cfg,
        )
        return reward

    def reset(self, env_ids: torch.Tensor | None = None):
        # """Reset phase flags either for all environments or for the specified env_ids."""
        if self.phase_flags is None:
            return
        if env_ids is None:
            # Reset all flags for all environments
            for key in self.phase_flags:
                self.phase_flags[key].zero_()
        else:
            # Reset only the environments in env_ids
            for key in self.phase_flags:
                self.phase_flags[key][env_ids] = False


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def shaped_pickup_reward(
    env: ManagerBasedRLEnv,
    phase_flags: dict[str, torch.Tensor],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # === Phase 1 & 2 Combined: 3D Distance to Object ===
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = object.data.root_pos_w
    dist = torch.norm(ee_pos - obj_pos, dim=1)  # full 3D distance
    shaping = 1.0 - torch.tanh(dist / 0.05)  # std = 5cm
    reward += shaping * 1.0  # weight = 1.0

    # === Phase 2.5: Maintain Z alignment before gripping ===
    z_close = torch.abs(ee_pos[:, 2] - obj_pos[:, 2]) < 0.005  # Z-axis threshold
    phase_flags.setdefault("z_hold_counter", torch.zeros(env.num_envs, device=env.device))

    # Increase counter if within Z range; reset if out of range
    phase_flags["z_hold_counter"] += z_close.float()
    phase_flags["z_hold_counter"] *= z_close.float()  # reset to 0 when misaligned

    grip_ready = phase_flags["z_hold_counter"] > 20  # wait ~0.1s (20 steps @ 200Hz)

    # === Phase 3: Gripper close ===
    gripper_joint_1 = robot.data.joint_pos[:, 6]
    gripper_joint_2 = robot.data.joint_pos[:, 7]
    gripper_open = 0.5 * (torch.abs(gripper_joint_1) + torch.abs(gripper_joint_2))
    gripping = gripper_open < 0.3
    newly_gripped = gripping & grip_ready & ~phase_flags["gripped"]
    reward += newly_gripped.float() * 2.0
    phase_flags["gripped"] |= gripping

    # === Phase 4: Lift ===
    lifted = object.data.root_pos_w[:, 2] > 0.02
    lifted_success = lifted & phase_flags["gripped"]
    reward += lifted_success.float() * 3.0
    print(phase_flags["gripped"])

    return reward, phase_flags
