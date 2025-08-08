# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of a CMO-PPO agent from RSL-RL."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a CMO-PPO agent with RSL-RL.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
   RslRlOnPolicyRunnerCfg,
   RslRlVecEnvWrapper,
   export_policy_as_jit,
   export_policy_as_onnx,
)
import orbit.surgical.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg


def main():
    """Play with a CMO-PPO agent."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    obs, _ = env.get_observations()

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    warmup_episodes = 3  # skip first 3 for each env

    reach_success  = torch.zeros(num_envs, dtype=torch.bool, device=device)
    orient_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    running_mask   = torch.ones(num_envs, dtype=torch.bool, device=device)  # still running?
    episode_counter_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)

    while simulation_app.is_running():
        with torch.inference_mode():
            # --- save "pre-step" metrics so you can evaluate terminal state even if wrapper auto-resets ---
            raw_env = env.unwrapped
            prev_phase = raw_env.mode_flags.clone()  # shape (num_envs,)
            prev_reached = raw_env.check_reached.clone()  # shape (num_envs,)
            prev_orient = raw_env.check_orient.clone()  # shape (
            # recompute needle + goal positions before stepping
            needle = raw_env.scene["object"]
            needle_center_prev = needle.data.root_pos_w
            needle_quat_prev = needle.data.root_quat_w
            ee_frame = raw_env.scene["ee_1_frame"]
            N = raw_env.num_envs

            from orbit.surgical.tasks.surgical.correct_orientation.mdp.visualization import quat_to_rot_matrix, _contact_point_world

            contact_center_prev = _contact_point_world(needle_center_prev, needle_quat_prev, needle_center_prev.device)
            needle_rot_prev = quat_to_rot_matrix(needle_quat_prev)
            offset_local = torch.tensor([0.0005, 0.001, 0.0], device=needle_center_prev.device).expand(N, 3)
            offset_world = torch.bmm(needle_rot_prev, offset_local.unsqueeze(-1)).squeeze(-1)
            needle_center_prev = contact_center_prev + offset_world

            # per-env goal: make it relative by using env index * spacing as base (assumes envs laid out along X)
            spacing = getattr(raw_env.scene, "env_spacing", 1.0)  # fallback if not set
            # base origin for each env in world coords (adjust axis if your layout differs)
            env_indices = torch.arange(N, device=device).unsqueeze(1)  # (N,1)
            env_bases = torch.zeros((N, 3), device=device)
            env_bases[:, 0] = env_indices.squeeze(1) * spacing  # shift along X

            # relative offset from that base
            relative_offset = torch.tensor([-0.1863, 0.1419, 0.1296], device=device)  # per-env offset
            goal_point = env_bases + relative_offset.unsqueeze(0)  # (N,3)

            dist_to_goal_prev = torch.norm(needle_center_prev - goal_point, dim=1)
            # print(prev_phase)
            # print(dist_to_goal_prev)

            # --- step ---
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Use the saved pre-step values for evaluation of those that just finished
            done_idx = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                # increment episode counters
                episode_counter_per_env[done_idx] += 1

                for env_id in done_idx:
                    if episode_counter_per_env[env_id] <= warmup_episodes:
                        continue

                    # evaluate using pre-step (terminal) values
                    reach_success[env_id] = (prev_reached[env_id] == 1)
                    orient_success[env_id] = (prev_orient[env_id] == 1)
                    running_mask[env_id] = False

                # ✅ stop when all envs have been evaluated once (beyond warmup)
                if not running_mask.any():
                    break

    reach_count = reach_success.sum().item()
    orient_count = orient_success.sum().item()

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Reach Phase Success: {reach_count}/{num_envs} ({100*reach_count/num_envs:.1f}%)")
    print(f"Orient Success (<2mm): {orient_count}/{num_envs} ({100*orient_count/num_envs:.1f}%)")
    print("==========================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
