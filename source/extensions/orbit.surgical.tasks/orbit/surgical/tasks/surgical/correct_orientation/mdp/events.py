import torch
from isaaclab.envs import ManagerBasedEnv
from typing import Optional
from .shared_phase_flags import get_mode_flags
from .visualization import _contact_point_world, ee_contact_point_world, quat_to_rot_matrix, visualize_reference_path
from .path_generator import LinearPathGenerator


def _active_ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids


def reset_only_robot1(env: ManagerBasedEnv,
                      env_ids: Optional[torch.Tensor] = None) -> None:
    """
    Reset *only* robot_1 to its init-state, keeping robot_2 and the needle intact.
    Works for batched environments.
    """
    robot = env.scene["robot_1"]
    device = env.device
    ids = _active_ids(env, env_ids)

    # ── joint state tensors ──────────────────────────────────────────────────
    joint_pos = torch.zeros((ids.numel(), robot.num_joints), device=device)
    for j_idx, name in enumerate(robot.joint_names):
        if name in robot.cfg.init_state.joint_pos:
            joint_pos[:, j_idx] = robot.cfg.init_state.joint_pos[name]
    joint_vel = torch.zeros_like(joint_pos)

    # ── root pose (add per-env origin) ───────────────────────────────────────
    origins = env.scene.env_origins[ids]                      # (M,3)
    local_pos = torch.tensor(robot.cfg.init_state.pos, device=device)
    local_quat = torch.tensor(robot.cfg.init_state.rot, device=device)
    root_pose = torch.zeros((ids.numel(), 7), device=device)
    root_pose[:, :3] = origins + local_pos                         # world xyz
    root_pose[:, 3:] = local_quat

    # ── write to sim ────────────────────────────────────────────────────────
    robot.write_root_pose_to_sim(root_pose, env_ids=ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)


def reset_needle_about_pivot(env, env_ids: Optional[torch.Tensor] = None) -> None:
    """
    Re-randomise the needle around its pivot without disturbing the robots.
    Works for batched environments.
    """
    needle = env.scene["object"]
    device = env.device
    ids = _active_ids(env, env_ids)
    N = ids.numel()

    # Initialize flag if it doesn't exist
    if not hasattr(env, "check_reached"):
        env.check_reached = torch.zeros(N, dtype=torch.long, device=device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.check_reached[ids] = 0
    if not hasattr(env, "check_orient"):
        env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.check_orient[ids] = 0
    if not hasattr(env, "path_initialized"):
        env.path_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.path_initialized[ids] = False

    if not hasattr(env, "path_initialized_2"):
        env.path_initialized_2 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.path_initialized_2[ids] = False
    if not hasattr(env, "current_path_index"):
        env.current_path_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.current_path_index[ids] = 0
    if not hasattr(env, "current_path_index_2"):
        env.current_path_index_2 = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.current_path_index_2[ids] = 0
    if not hasattr(env, "align_counter"):
        env.align_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        # Reset only the relevant envs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        env.align_counter[ids] = 0

    root_state = needle.data.root_state_w.clone()      # (num_envs,13)

    # ── CONSTANTS  (LOCAL  ↦  per-env world) ────────────────────────────────
    pivot_local = torch.tensor([-0.200, 0.1435, 0.1505], device=device)   # in env frame
    offset_local = torch.tensor([0.005, 0.000 , -0.010], device=device)
    q_init = torch.tensor([0.66446, 0.66446, -0.24184, 0.24184], device=device)  # (w,x,y,z)

    origins = env.scene.env_origins[ids]                             # (N,3)
    pivot_world = origins + pivot_local                                  # (N,3)

    # ── random roll about needle axis ───────────────────────────────────────
    roll_deg = torch.empty(0, device=device)
    while roll_deg.numel() < N:
        roll_deg = torch.empty(N, device=device).uniform_(-50.0, 50.0)
        # roll_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)

    roll_deg = roll_deg[:N]  # trim to exactly N
    roll_rad = torch.deg2rad(roll_deg)

    half = roll_rad * 0.5
    q_roll = torch.stack([torch.cos(half), torch.sin(half),                # (N,4)
                          torch.zeros_like(half), torch.zeros_like(half)],
                         dim=1)

    # quaternion multiplication  q_final = q_roll ∘ q_init
    w1, x1, y1, z1 = q_roll.T
    w2, x2, y2, z2 = q_init
    q_final = torch.stack([ w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2 ], dim=1)

    # ── rotate offset vector by q_roll (fast formula) ───────────────────────
    u = q_roll[:, 1:]                       # (N,3)
    s = q_roll[:, :1]                       # (N,1)
    offset = offset_local.expand(N, 3)           # (N,3)

    dot = (u * offset).sum(1, keepdim=True)
    rotated_offset = (2 * dot) * u \
        + (s * s - (u * u).sum(1, keepdim=True)) * offset \
        + 2 * s * torch.cross(u, offset, dim=1)                      # (N,3)

    # ── write new pose ──────────────────────────────────────────────────────
    root_state[ids, :3] = pivot_world + rotated_offset
    root_state[ids, 3:7] = q_final
    root_state[ids, 7:13].zero_()                  # zero the velocities
    env.initial_needle_pos = needle.data.root_pos_w.clone()

    needle.write_root_state_to_sim(root_state[ids], env_ids=ids)


def reset_mode_flags(env, env_ids: torch.Tensor | None = None):
    """
    Reset mode_flags for all or selected environments to stage 0.
    This keeps the ID consistent across reward + reset scopes.
    """
    if not hasattr(env, "mode_flags"):
        get_mode_flags(env)  # create if not exists

    if env_ids is None:
        env.mode_flags[:] = 0  # reset all to stage 0
    else:
        env.mode_flags[env_ids] = 0  # reset only specific environments
