from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from .visualization import *
from .visualization import _contact_point_world
from .shared_phase_flags import get_mode_flags
from .events import reset_goal_about_pivot_xz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Returns the object's position in the robot's root frame (body frame)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    obj_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)
    return obj_pos_b


def gripper_tip_positions(env) -> torch.Tensor:
    """
    Returns a (N,6) tensor of absolute positions of the blue and yellow gripper tips:
    [blue_x, blue_y, blue_z, yellow_x, yellow_y, yellow_z].
    """
    device = env.device
    N = env.num_envs
    ee_frame = env.scene["ee_1_frame"]
    # EE‐center in world
    ee_center = ee_contact_point_world(ee_frame, device)       # (N,3)
    quat = ee_frame.data.target_quat_w[..., 0, :]              # (N,4)
    R = quat_to_rot_matrix(quat)                               # (N,3,3)
    # tip offsets in EE‐frame
    blue_offset = torch.tensor([0.0045, 0.0, 0.0], device=device).unsqueeze(0).expand(N,3).unsqueeze(-1)
    yellow_offset = torch.tensor([-0.0045, 0.0, 0.0], device=device).unsqueeze(0).expand(N,3).unsqueeze(-1)
    # world positions
    blue_tip   = ee_center + torch.bmm(R, blue_offset).squeeze(-1)
    yellow_tip = ee_center + torch.bmm(R, yellow_offset).squeeze(-1)
    return torch.cat([blue_tip, yellow_tip], dim=1)  # (N,6)


def gripper_tip_contact_distances(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns distances from the gripper's left (blue), right (yellow), and EE center to the corresponding needle contact points."""
    device = env.device
    N = env.num_envs

    obj = env.scene["object"]
    ee_frame = env.scene["ee_1_frame"]

    # Needle pose and rotation
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    needle_rot = quat_to_rot_matrix(obj_quat)
    contact_center = _contact_point_world(obj_pos, obj_quat, device)

    # Needle-side contact targets
    contact_blue = contact_center + torch.bmm(needle_rot, torch.tensor([0.0, 0.0, 0.0045], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    contact_yellow = contact_center + torch.bmm(needle_rot, torch.tensor([0.0, 0.0, -0.0045], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    contact_center_ee = contact_center + torch.bmm(needle_rot, torch.tensor([0.0005, 0.001, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)

    # Gripper-side tip positions
    base_grip_pos = ee_contact_point_world(ee_frame, device)
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    grip_rot = quat_to_rot_matrix(grip_quat)

    blue_tip = base_grip_pos + torch.bmm(grip_rot, torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    yellow_tip = base_grip_pos + torch.bmm(grip_rot, torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)

    # Euclidean distances
    blue_dist = torch.norm(blue_tip - contact_blue, dim=1, keepdim=True)
    yellow_dist = torch.norm(yellow_tip - contact_yellow, dim=1, keepdim=True)
    center_dist = torch.norm(base_grip_pos - contact_center_ee, dim=1, keepdim=True)

    if not hasattr(env, "reference_path") or not hasattr(env, "current_path_index"):
        return torch.zeros((N, 3), device=device)

    return torch.cat([blue_dist, yellow_dist, center_dist], dim=1)


def center_to_path(env) -> torch.Tensor:
    """
    Returns a (N, 3) tensor of vectors from the gripper's current position to the current path target.
    Should be registered as an observation term.
    """
    device = env.device
    N = env.num_envs

    # Get current gripper position
    ee_frame = env.scene["ee_1_frame"]
    ee_center = ee_contact_point_world(ee_frame, device)  # (N, 3)

    # Get current target from path
    if not hasattr(env, "reference_path") or not hasattr(env, "current_path_index"):
        return torch.zeros((N, 3), device=device)

    ref_path = env.reference_path  # (N, num_steps, 3)
    path_idx = env.current_path_index  # (N,)
    max_step = ref_path.shape[1] - 1
    safe_idx = torch.clamp(path_idx, max=max_step)

    ref_center = ref_path[torch.arange(N, device=device), safe_idx]  # (N, 3)

    # Direction vector to current target
    vec_to_path = ref_center - ee_center  # (N,3)

    return vec_to_path


def blue_to_path(env) -> torch.Tensor:
    device = env.device
    N = env.num_envs

    ee_frame = env.scene["ee_1_frame"]
    ee_center = ee_contact_point_world(ee_frame, device)
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    grip_rot = quat_to_rot_matrix(grip_quat)
    blue_tip = ee_center + torch.bmm(
        grip_rot,
        torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
    ).squeeze(-1)

    if not hasattr(env, "reference_path_blue") or not hasattr(env, "current_path_index"):
        return torch.zeros((N, 3), device=device)

    ref_path = env.reference_path_blue
    path_idx = env.current_path_index
    max_step = ref_path.shape[1] - 1
    safe_idx = torch.clamp(path_idx, max=ref_path.shape[1] - 1)

    ref_blue = ref_path[torch.arange(N, device=device), safe_idx]

    # Direction vector to current target
    vec_to_path = ref_blue - blue_tip  # (N,3)

    return vec_to_path

def yellow_to_path(env) -> torch.Tensor:
    """
    Returns a (N,3) direction vector from the yellow gripper tip to the current path target.
    Masked to zero before step index >=5.
    """
    device = env.device
    N = env.num_envs

    # Get current gripper position
    ee_frame = env.scene["ee_1_frame"]
    ee_center = ee_contact_point_world(ee_frame, device)
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    grip_rot = quat_to_rot_matrix(grip_quat)
    yellow_tip = ee_center + torch.bmm(
        grip_rot,
        torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
    ).squeeze(-1)

    # Get current target from path
    if not hasattr(env, "reference_path_yellow") or not hasattr(env, "current_path_index"):
        return torch.zeros((N, 3), device=device)

    ref_path = env.reference_path_yellow
    path_idx = env.current_path_index  # (N,)
    max_step = ref_path.shape[1] - 1
    safe_idx = torch.clamp(path_idx, max=max_step)

    ref_yellow = ref_path[torch.arange(N, device=device), safe_idx]  # (N,3)

    # Direction vector from tip → target
    vec_to_path = ref_yellow - yellow_tip  # (N,3)

    return vec_to_path


def center_to_arc_path(env) -> torch.Tensor:
    device, N = env.device, env.num_envs
    if (not hasattr(env, "reference_path_2_center")) or (not hasattr(env, "current_path_index_2")):
        return torch.zeros((N, 3), device=device)
    ref = env.reference_path_2_center
    if ref.numel() == 0:
        return torch.zeros((N, 3), device=device)
    idx = torch.clamp(env.current_path_index_2, 0, ref.shape[1] - 1)

    ee = ee_contact_point_world(env.scene["ee_1_frame"], device)  # (N,3)
    tgt = ref[torch.arange(N, device=device), idx]                # (N,3)
    if not hasattr(env, "mode_flags"): get_mode_flags(env)
    mask1 = (env.mode_flags == 1).unsqueeze(-1)
    vec = tgt - ee
    return torch.where(mask1, vec, torch.zeros_like(vec))

def blue_to_arc_path(env) -> torch.Tensor:
    device, N = env.device, env.num_envs
    if (not hasattr(env, "reference_path_2_blue")) or (not hasattr(env, "current_path_index_2")):
        return torch.zeros((N, 3), device=device)
    ref = env.reference_path_2_blue
    if ref.numel() == 0:
        return torch.zeros((N, 3), device=device)
    idx = torch.clamp(env.current_path_index_2, 0, ref.shape[1] - 1)

    ee_f = env.scene["ee_1_frame"]
    ee = ee_contact_point_world(ee_f, device)
    R = quat_to_rot_matrix(ee_f.data.target_quat_w[..., 0, :])
    blue_tip = ee + torch.bmm(R, torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N,3).unsqueeze(-1)).squeeze(-1)

    tgt = ref[torch.arange(N, device=device), idx]
    if not hasattr(env, "mode_flags"): get_mode_flags(env)
    mask1 = (env.mode_flags == 1).unsqueeze(-1)
    vec = tgt - blue_tip
    return torch.where(mask1, vec, torch.zeros_like(vec))

def yellow_to_arc_path(env) -> torch.Tensor:
    device, N = env.device, env.num_envs
    if (not hasattr(env, "reference_path_2_yellow")) or (not hasattr(env, "current_path_index_2")):
        return torch.zeros((N, 3), device=device)
    ref = env.reference_path_2_yellow
    if ref.numel() == 0:
        return torch.zeros((N, 3), device=device)
    idx = torch.clamp(env.current_path_index_2, 0, ref.shape[1] - 1)

    ee_f = env.scene["ee_1_frame"]
    ee = ee_contact_point_world(ee_f, device)
    R = quat_to_rot_matrix(ee_f.data.target_quat_w[..., 0, :])
    yellow_tip = ee + torch.bmm(R, torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N,3).unsqueeze(-1)).squeeze(-1)

    tgt = ref[torch.arange(N, device=device), idx]
    if not hasattr(env, "mode_flags"): get_mode_flags(env)
    mask1 = (env.mode_flags == 1).unsqueeze(-1)
    vec = tgt - yellow_tip
    return torch.where(mask1, vec, torch.zeros_like(vec))

# def blue_to_path_2(env) -> torch.Tensor:
#     device = env.device
#     N = env.num_envs

#     ee_frame = env.scene["ee_1_frame"]
#     ee_center = ee_contact_point_world(ee_frame, device)
#     grip_quat = ee_frame.data.target_quat_w[..., 0, :]
#     grip_rot = quat_to_rot_matrix(grip_quat)
#     blue_tip = ee_center + torch.bmm(
#         grip_rot,
#         torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
#     ).squeeze(-1)

#     if not hasattr(env, "reference_path_2") or not hasattr(env, "current_path_index_2"):
#         return torch.zeros((N, 3), device=device)

#     ref_path = env.reference_path_2
#     path_idx = env.current_path_index_2
#     if ref_path.numel() == 0 or torch.all(ref_path == 0):
#         return torch.zeros(N, 3, device=device)
#     max_step = max(ref_path.shape[1] - 1, 0)
#     safe_idx = torch.clamp(path_idx, min=0, max=max_step)

#     ref_blue = ref_path[torch.arange(N, device=device), safe_idx]

#     # Direction vector to current target
#     vec_to_path = ref_blue - blue_tip  # (N,3)

#     return vec_to_path


# def yellow_to_path_2(env) -> torch.Tensor:
#     device = env.device
#     N = env.num_envs

#     ee_frame = env.scene["ee_1_frame"]
#     ee_center = ee_contact_point_world(ee_frame, device)
#     grip_quat = ee_frame.data.target_quat_w[..., 0, :]
#     grip_rot = quat_to_rot_matrix(grip_quat)
#     yellow_tip = ee_center + torch.bmm(
#         grip_rot,
#         torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
#     ).squeeze(-1)

#     if not hasattr(env, "reference_path_2") or not hasattr(env, "current_path_index_2"):
#         return torch.zeros((N, 3), device=device)

#     ref_path = env.reference_path_2
#     path_idx = env.current_path_index_2
#     if ref_path.numel() == 0 or torch.all(ref_path == 0):
#         return torch.zeros(N, 3, device=device)
#     max_step = max(ref_path.shape[1] - 1, 0)
#     safe_idx = torch.clamp(path_idx, min=0, max=max_step)

#     ref_yellow = ref_path[torch.arange(N, device=device), safe_idx]

#     # Direction vector to current target
#     vec_to_path = ref_yellow - yellow_tip  # (N,3)

#     return vec_to_path


# def center_to_path_2(env) -> torch.Tensor:
#     device = env.device
#     N = env.num_envs

#     # EE center in world
#     ee_frame = env.scene["ee_1_frame"]
#     ee_center = ee_contact_point_world(ee_frame, device)  # (N,3)

#     # Current arc-path target
#     if not hasattr(env, "reference_path_2") or not hasattr(env, "current_path_index_2"):
#         return torch.zeros((N, 3), device=device)
#     ref_path = env.reference_path_2
#     if ref_path.numel() == 0:
#         return torch.zeros((N, 3), device=device)

#     idx = torch.clamp(env.current_path_index_2, min=0, max=ref_path.shape[1] - 1)
#     ref_center = ref_path[torch.arange(N, device=device), idx]  # (N,3)

#     return ref_center - ee_center


# def tip_to_path_obs(env) -> torch.Tensor:
#     device = env.device
#     N = env.num_envs

#     # compute blue and yellow vectors
#     blue_vec = blue_to_path_2(env)    # (N,3)
#     yellow_vec = yellow_to_path_2(env)  # (N,3)

#     # ensure side flag exists: +1 means use blue, -1 means use yellow
#     if not hasattr(env, "needle_side_flag"):
#         env.needle_side_flag = torch.zeros(N, dtype=torch.long, device=device)
#     side_flag = env.needle_side_flag  # (N,)

#     use_blue = (side_flag == 1).unsqueeze(-1)   # (N,1)
#     use_yellow = (side_flag == -1).unsqueeze(-1)  # (N,1)

#     # masked selection: only the relevant tip vector passes through
#     active_vec = blue_vec * use_blue + yellow_vec * use_yellow  # (N,3)

#     return active_vec


def phase_flags_observation(env) -> torch.BoolTensor:
    """
    Returns a one‐hot boolean flag per environment indicating its current phase.
    Phase 0: mode_flags == 0
    Phase 1: mode_flags == 1
    Output shape: (N, 2), where N = env.num_envs
    """
    device = env.device
    N = env.num_envs

    # Ensure mode_flags exists
    if not hasattr(env, "mode_flags"):
        env.mode_flags = torch.zeros(N, dtype=torch.long, device=device)

    flags = env.mode_flags  # shape: (N,)

    # Build boolean masks for each phase
    phase0 = flags == 0     # shape: (N,)
    phase1 = flags == 1     # shape: (N,)

    # Ensure needle_side_flag exists (fallback to zero = neither)
    if not hasattr(env, "needle_side_flag"):
        env.needle_side_flag = torch.zeros(N, dtype=torch.long, device=device)
    side_flag = env.needle_side_flag  # shape: (N,)

    side_left = side_flag == 1
    side_right = side_flag == -1

    return torch.stack([phase0, phase1, side_left, side_right], dim=1)

# def needle_to_path(env) -> torch.Tensor:
#     """
#     Returns a (N,3) direction vector from the needle center to the current path target.
#     Masked to zero before step index >=5.
#     """
#     device = env.device
#     N = env.num_envs

#     # Get current needle position
#     needle = env.scene["object"]
#     needle_pos = needle.data.root_pos_w  # (N, 3)
#     needle_quat = needle.data.root_quat_w
#     needle_rot = quat_to_rot_matrix(needle_quat)
#     contact_center = _contact_point_world(needle_pos, needle_quat, device)
#     offset_local = torch.tensor([0.0005, 0.001, 0.0], device=device).expand(N, 3)
#     offset_world = torch.bmm(needle_rot, offset_local.unsqueeze(-1)).squeeze(-1)
#     needle_center = contact_center + offset_world

#     # Get current target from path
#     if not hasattr(env, "reference_path_2") or not hasattr(env, "current_path_index_2"):
#         return torch.zeros((N, 3), device=device)

#     ref_path = env.reference_path_2  # (N, num_steps, 3)
#     path_idx = env.current_path_index_2  # (N,)
#     max_step = ref_path.shape[1] - 1
#     safe_idx = torch.clamp(path_idx, max=max_step)

#     ref_center = ref_path[torch.arange(N, device=device), safe_idx]  # (N, 3)

#     # Direction vector to current target
#     vec_to_path = ref_center - needle_center  # (N,3)

#     return vec_to_path


def goal_observation(env) -> torch.Tensor:
    """
    Returns goal-related observations for each environment:
    - Relative vector (goal - needle_center)  (N, 3)
    - Distance to goal                        (N, 1)
    
    Uses env.goal_point (randomized per reset).
    """
    device = env.device
    N = env.num_envs

    # --- Needle center ---
    needle = env.scene["object"]
    needle_pos = needle.data.root_pos_w
    needle_quat = needle.data.root_quat_w
    needle_rot = quat_to_rot_matrix(needle_quat)
    contact_center = _contact_point_world(needle_pos, needle_quat, device)

    offset_local = torch.tensor([0.0005, 0.001, 0.0], device=device).expand(N, 3)
    offset_world = torch.bmm(needle_rot, offset_local.unsqueeze(-1)).squeeze(-1)
    needle_center = contact_center + offset_world

    # --- Randomized goal (set by reset_goal_point_about_pivot_xz) ---
    if not hasattr(env, "goal_point"):
        reset_goal_about_pivot_xz(env)     # creates env.goal_point on first call
    goal_point = env.goal_point  # (N, 3)

    # --- Phase gating (only active in phase 1) ---
    if not hasattr(env, "mode_flags"):
        get_mode_flags(env)  # ensure exists
    phase_mask = (env.mode_flags == 1)

    # --- Compute relative vector and distance ---
    goal_vec = goal_point - needle_center                   # (N, 3)
    goal_dist = torch.norm(goal_vec, dim=1, keepdim=True)   # (N, 1)
    goal_obs = torch.cat([goal_vec, goal_dist], dim=1)      # (N, 4)

    # Zero out when not in phase 1
    goal_obs = torch.where(
        phase_mask.unsqueeze(-1),
        goal_obs,
        torch.zeros_like(goal_obs)
    )

    return goal_obs


def current_mode_onehot(env) -> torch.Tensor:
    """
    Returns a one-hot encoded tensor of current phase:
    [reach, orient] -> shape: (N, num_phases)
    """
    N = env.num_envs
    device = env.device

    # Ensure mode_flags exists
    if not hasattr(env, "mode_flags"):
        env.mode_flags = torch.zeros(N, dtype=torch.long, device=device)

    # Auto-detect max phase count (at least 2 for now)
    num_phases = 2

    return torch.nn.functional.one_hot(
        env.mode_flags.clamp(max=num_phases - 1),
        num_classes=num_phases
    ).float()


def needle_side_onehot(env) -> torch.Tensor:
    """
    Returns a one-hot encoded tensor indicating which side the GOAL is on 
    relative to the needle:
    [goal_left_of_needle, goal_right_of_needle] -> shape: (N, 2)
    """
    N = env.num_envs
    device = env.device

    # Needle center (world position)
    needle = env.scene["object"]
    needle_pos = needle.data.root_pos_w  # (N, 3)

    # Randomized goal point
    if not hasattr(env, "goal_point"):
        reset_goal_about_pivot_xz(env)     # creates env.goal_point on first call
    goal_point = env.goal_point  # (N, 3)

    # Compute Y difference (goal - needle)
    y_diff = goal_point[:, 1] - needle_pos[:, 1]  # (N,)

    # side_flag: 0 if goal is LEFT (+Y), 1 if goal is RIGHT (-Y)
    side_flag = (y_diff < 0.0).long()

    # One-hot encode: [left, right]
    return torch.nn.functional.one_hot(side_flag, num_classes=2).float()


def tip_to_path_distances(env) -> torch.Tensor:
    """
    Returns [dist_left_tip_to_path, dist_right_tip_to_path] for each env.
    Shape: (N, 2)
    """
    N = env.num_envs
    device = env.device
    
    if (not hasattr(env, "current_path_index_2")) or (not hasattr(env, "reference_path_2")):
        return torch.zeros((N, 2), device=device)
    ref_path = env.reference_path_2
    if ref_path.numel() == 0:
        return torch.zeros((N, 2), device=device)
    idx = torch.clamp(env.current_path_index_2, max=ref_path.shape[1] - 1)
    ref_center = ref_path[torch.arange(N), idx]

    # EE center & tip positions
    ee_frame = env.scene["ee_1_frame"]
    ee_center = ee_frame.data.target_pos_w[..., 0, :]
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    from orbit.surgical.tasks.surgical.correct_orientation.mdp.visualization import quat_to_rot_matrix
    grip_rot = quat_to_rot_matrix(grip_quat)

    blue_tip = ee_center + torch.bmm(
        grip_rot, torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
    ).squeeze(-1)
    yellow_tip = ee_center + torch.bmm(
        grip_rot, torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)
    ).squeeze(-1)

    # Distances
    left_tip_dist = torch.norm(blue_tip - ref_center, dim=-1, keepdim=True)
    right_tip_dist = torch.norm(yellow_tip - ref_center, dim=-1, keepdim=True)

    return torch.cat([left_tip_dist, right_tip_dist], dim=-1)


def needle_linear_speed(env) -> torch.Tensor:
    """
    Returns the scalar linear speed of the needle’s root in world‐space:
     - shape: (N, 1)
    """
    device = env.device
    N = env.num_envs

    obj = env.scene["object"]
    # world‐space linear velocity, shape (N, 3)
    lin_vel = obj.data.root_lin_vel_w
    # speed scalar, keepdim→(N,1)
    speed = torch.norm(lin_vel, dim=1, keepdim=True)

    return speed
