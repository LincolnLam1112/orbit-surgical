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
    Reset *only* robot_1 to its init-state, but with a randomized root position
    around the configured init_state.pos. Joint positions stay as in init_state.
    """
    robot = env.scene["robot_1"]
    device = env.device
    ids = _active_ids(env, env_ids)

    # ── joint state tensors (same as before) ────────────────────────────────
    joint_pos = torch.zeros((ids.numel(), robot.num_joints), device=device)
    for j_idx, name in enumerate(robot.joint_names):
        if name in robot.cfg.init_state.joint_pos:
            joint_pos[:, j_idx] = robot.cfg.init_state.joint_pos[name]
    joint_vel = torch.zeros_like(joint_pos)

    # ── root pose with RANDOMIZED local position ───────────────────────────
    origins = env.scene.env_origins[ids]  # (M, 3)

    base_pos = torch.tensor(robot.cfg.init_state.pos, device=device)  # (3,)
    base_quat = torch.tensor(robot.cfg.init_state.rot, device=device) # (4,)

    pos_noise = torch.empty_like(base_pos).uniform_(
        -0.05, 0.05
    )

    # pos_noise = torch.empty_like(base_pos).uniform_(
    #     0.0, 0.0
    # )


    local_pos = base_pos + pos_noise              # randomized local pos
    local_quat = base_quat                        # keep same orientation

    root_pose = torch.zeros((ids.numel(), 7), device=device)
    root_pose[:, :3] = origins + local_pos        # world xyz
    root_pose[:, 3:] = local_quat                 # world quat

    # ── write to sim ────────────────────────────────────────────────────────
    robot.write_root_pose_to_sim(root_pose, env_ids=ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)


def reset_needle_about_pivot_xz(env, env_ids: Optional[torch.Tensor] = None) -> None:
    """
    Randomize needle about its pivot with TWO rotations:
    1) around Z by a random angle, then
    2) around X by a random angle,
    and then apply on top of q_init. Works batched.
    """
    needle = env.scene["object"]
    device = env.device
    ids = _active_ids(env, env_ids)
    N = ids.numel()

    root_state = needle.data.root_state_w.clone()  # (num_envs, 13)

    # # ── CONSTANTS (LOCAL → per-env world) ───────────────────────────────────
    # # pivot_local  = torch.tensor([-0.200, 0.1435, 0.1505], device=device)   # env frame
    # # pivot_local  = torch.tensor([-0.200, 0.1435, 0.1], device=device)   # env frame
    # pivot_local  = torch.tensor([0.04, 0.043, 0.054], device=device)   # env frame
    # offset_local = torch.tensor([ 0.005, 0.000 , -0.010], device=device)
    # # offset_local = torch.tensor([0.0, 0.0 , 0.0], device=device)
    # # q_init       = torch.tensor([0.66446, 0.66446, -0.24184, 0.24184], device=device)  # (w,x,y,z)
    # q_init = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)

    # ── CONSTANTS (LOCAL → per-env world) ───────────────────────────────────
    cfg = getattr(env, "cfg", None)

    # Pivot position (env-frame): read from the pivot object init_state or fall back
    pivot_local = torch.tensor(
        getattr(getattr(getattr(cfg, "scene", None), "needle_pivot_xform", None), "init_state", None).pos
        if (cfg and hasattr(cfg.scene.needle_pivot_xform, "init_state"))
        else (-0.200, 0.1435, 0.1),
        device=device
    )

    # Anchor offset (env-frame): use the single source of truth from cfg
    anchor_off = getattr(getattr(cfg, "pivot_joint", None), "anchor_offset_local", None)
    if anchor_off is None:
        anchor_off = (0.005, 0.0, -0.010)
    offset_local = torch.tensor(anchor_off, device=device)

    # print(anchor_off)

    # Base orientation
    q_init = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)
    # q_init = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    origins     = env.scene.env_origins[ids]     # (N,3)
    pivot_world = origins + pivot_local          # (N,3)

    # ── random angles (deg) for Z and X ─────────────────────────────────────
    z_deg = torch.empty(0, device=device)
    x_deg = torch.empty(0, device=device)
    while z_deg.numel() < N or x_deg.numel() < N:
        z_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)  # Z first
        x_deg = torch.empty(N, device=device).uniform_(-20.0, 20.0)  # X second
        # x_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)  # X second
    z_deg = z_deg[:N]
    x_deg = x_deg[:N]
    z_half = torch.deg2rad(z_deg) * 0.5
    x_half = torch.deg2rad(x_deg) * 0.5

    # ── unit quats for Z and X ──────────────────────────────────────────────
    # Z-axis: (w, x, y, z) = (cos, 0, 0, sin)
    q_z = torch.stack([torch.cos(z_half),
                       torch.zeros_like(z_half),
                       torch.zeros_like(z_half),
                       torch.sin(z_half)], dim=1)
    # X-axis: (w, x, y, z) = (cos, sin, 0, 0)
    q_x = torch.stack([torch.cos(x_half),
                       torch.sin(x_half),
                       torch.zeros_like(x_half),
                       torch.zeros_like(x_half)], dim=1)

    # ── compose delta: apply Z then X  → q_delta = q_x ∘ q_z ────────────────
    # (rightmost is applied first; mirrors your q_final = q_roll ∘ q_init style)
    w1, x1, y1, z1 = q_x.T
    w2, x2, y2, z2 = q_z.T
    q_delta = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=1)

    # ── total orientation: q_final = q_delta ∘ q_init ───────────────────────
    W1, X1, Y1, Z1 = q_delta.T
    W2, X2, Y2, Z2 = q_init
    q_final = torch.stack([
        W1*W2 - X1*X2 - Y1*Y2 - Z1*Z2,
        W1*X2 + X1*W2 + Y1*Z2 - Z1*Y2,
        W1*Y2 - X1*Z2 + Y1*W2 + Z1*X2,
        W1*Z2 + X1*Y2 - Y1*X2 + Z1*W2
    ], dim=1)

    anchor_local = torch.tensor(
        env.cfg.pivot_joint.anchor_offset_local, device=device
    ).expand_as(pivot_world)  # (N, 3)

    # rotate anchor offset by needle orientation (same as _quat_apply)
    u = q_final[:, 1:]
    s = q_final[:, :1]
    dot = (u * anchor_local).sum(1, keepdim=True)
    rotated_off = (2 * dot) * u \
        + (s * s - (u * u).sum(1, keepdim=True)) * anchor_local \
        + 2 * s * torch.cross(u, anchor_local, dim=1)

    # make the anchor land exactly at the pivot
    needle_world = pivot_world - rotated_off

    root_state[ids, :3]  = needle_world
    root_state[ids, 3:7] = q_final
    root_state[ids, 7:13].zero_()   # zero velocities
    env.initial_needle_pos = needle.data.root_pos_w.clone()
    needle.write_root_state_to_sim(root_state[ids], env_ids=ids)

    from .joint_utils import update_hinge_targets_after_reset
    update_hinge_targets_after_reset(env, env_ids, axis="X")


def reset_mode_flags(env, env_ids: torch.Tensor | None = None):
    """
    Reset mode_flags for all or selected environments to stage 0.
    This keeps the ID consistent across reward + reset scopes.
    """
    device = env.device
    ids = _active_ids(env, env_ids)
    N = ids.numel()

    # Ensure temporary clamps are removed on reset
    # if hasattr(env, "_clamp_joint_active"):
    #     # destroy for all requested env_ids
    #     destroy_clamp_joint(env, env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids))

    if not hasattr(env, "mode_flags"):
        get_mode_flags(env)  # create if not exists
    # Initialize flag if it doesn't exist
    if not hasattr(env, "check_reached"):
        env.check_reached = torch.zeros(N, dtype=torch.long, device=device)
    else:
        # Reset only the relevant envs
        env.check_reached[ids] = 0
    if not hasattr(env, "check_orient"):
        env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
    else:
        # Reset only the relevant envs
        env.check_orient[ids] = 0
    if not hasattr(env, "path_initialized"):
        env.path_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        # Reset only the relevant envs
        env.path_initialized[ids] = False

    if not hasattr(env, "path_initialized_2"):
        env.path_initialized_2 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        # Reset only the relevant envs
        env.path_initialized_2[ids] = False
    if not hasattr(env, "current_path_index"):
        env.current_path_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        # Reset only the relevant envs
        env.current_path_index[ids] = 0
    if not hasattr(env, "current_path_index_2"):
        env.current_path_index_2 = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        # Reset only the relevant envs
        env.current_path_index_2[ids] = 0
    reset_goal_about_pivot_xz(env)


    # if not hasattr(env, "reference_path"):
    #     env.reference_path = torch.zeros((env.num_envs, 10, 3), device=env.device)
    #     env.reference_path_blue = torch.zeros((env.num_envs, 10, 3), device=env.device)
    #     env.reference_path_yellow = torch.zeros((env.num_envs, 10, 3), device=env.device)
    # else:
    #     env.reference_path[ids] = 0
    #     if hasattr(env, "reference_path_blue"):
    #         env.reference_path_blue[ids] = 0
    #     if hasattr(env, "reference_path_yellow"):
    #         env.reference_path_yellow[ids] = 0


    if not hasattr(env, "reference_path_2"):
        env.reference_path_2 = torch.zeros((env.num_envs, 3, 3), device=env.device)
    else:
        env.reference_path_2[ids] = 0

    if env_ids is None:
        env.mode_flags[:] = 0  # reset all to stage 0
    else:
        env.mode_flags[env_ids] = 0  # reset only specific environments
    q = env.scene["robot_1"].data.joint_pos

    if not hasattr(env, "_prev_q"):
        env._prev_q = q.clone()        # # Contouring-style nudging: reward forward tangential motion; penalize lateral slip
        # gate_close = (needle_dist_to_path < 0.010).float()
        # # positive tangential progress
        # reward[active_ids_1] += 1.2 * torch.clamp(progress_along_arc[active_ids_1], min=0.0) * gate_close
        # # lateral (off-tangent) velocity penalty
        # proj = (torch.sum(v_center * tangent_2, dim=1, keepdim=True) * tangent_2)  # (N,3)
        # v_lat = v_center - proj
        # reward[active_ids_1] -= 0.15 * torch.norm(v_lat[active_ids_1], dim=1)
    else:
        env._prev_q[ids] = q[ids].clone()

    if not hasattr(env, "_prev_push_tip"):
        env._prev_push_tip = torch.zeros((N, 3), device=device)
    else:
        env._prev_push_tip[ids] = torch.zeros((N, 3), device=device)


def reset_goal_about_pivot_xz(env, env_ids: Optional[torch.Tensor] = None) -> None:
    """
    Set goal to track the needle spawn pose: use the SAME pivot, anchor offset, and X/Z
    rotations as the needle reset, then place the goal a small distance along the
    needle's local +X (tip-forward).
    """
    device = env.device
    ids = _active_ids(env, env_ids)
    N = ids.numel()

    # --- read single source of truth from cfg (matches needle reset) ---
    cfg = getattr(env, "cfg", None)
    pivot_local = torch.tensor(
        getattr(getattr(getattr(cfg, "scene", None), "needle_pivot_xform", None), "init_state", None).pos,
        device=device,
    )

    offset_local = torch.tensor((0.005, 0.0, -0.02231), device=device)

    # Base orientation (same as needle)
    q_init = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)

    # Per-env world pivot
    origins     = env.scene.env_origins[ids]
    pivot_world = origins + pivot_local

    # --- SAME randomization as needle (Z then X) ---
    z_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)
    x_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)
    # x_deg = torch.empty(N, device=device).uniform_(0.0, 0.0)
    z_half = torch.deg2rad(z_deg) * 0.5
    x_half = torch.deg2rad(x_deg) * 0.5

    # unit quats
    q_z = torch.stack([torch.cos(z_half), torch.zeros_like(z_half), torch.zeros_like(z_half), torch.sin(z_half)], dim=1)
    q_x = torch.stack([torch.cos(x_half), torch.sin(x_half), torch.zeros_like(x_half), torch.zeros_like(x_half)], dim=1)

    # compose q_delta = q_x ∘ q_z
    w1, x1, y1, z1 = q_x.T
    w2, x2, y2, z2 = q_z.T
    q_delta = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=1)

    # needle final orientation: q_final = q_delta ∘ q_init
    W1, X1, Y1, Z1 = q_delta.T
    W2, X2, Y2, Z2 = q_init
    q_final = torch.stack([
        W1*W2 - X1*X2 - Y1*Y2 - Z1*Z2,
        W1*X2 + X1*W2 + Y1*Z2 - Z1*Y2,
        W1*Y2 - X1*Z2 + Y1*W2 + Z1*X2,
        W1*Z2 + X1*Y2 - Y1*X2 + Z1*W2
    ], dim=1)

    # rotate anchor offset by q_delta (same fast formula)
    u = q_delta[:, 1:]
    s = q_delta[:, :1]
    off = offset_local.expand(N, 3)
    rotated_off = (2*(u*off).sum(1, keepdim=True))*u \
        + (s*s - (u*u).sum(1, keepdim=True))*off \
        + 2*s*torch.cross(u, off, dim=1)

    # needle world position at spawn (matches needle reset)
    needle_world = pivot_world + rotated_off

    # goal offset along needle local +X (tip-forward); tweak distance if desired
    goal_fwd_local = torch.tensor([-0.02, -0.015, 0.0], device=device)  # 1.5 cm ahead
    # rotate by q_final
    u2 = q_final[:, 1:]; s2 = q_final[:, :1]
    dot2 = (u2 * goal_fwd_local.expand(N, 3)).sum(1, keepdim=True)
    goal_fwd_world = (2*dot2)*u2 \
        + (s2*s2 - (u2*u2).sum(1, keepdim=True))*goal_fwd_local.expand(N, 3) \
        + 2*s2*torch.cross(u2, goal_fwd_local.expand(N, 3), dim=1)

    goal_world = needle_world + goal_fwd_world

    # write
    if not hasattr(env, "goal_point"):
        env.goal_point = torch.zeros((env.num_envs, 3), device=device)
    env.goal_point[ids] = goal_world

    # keep angles for debugging if helpful
    env._goal_last_z_deg = z_deg.clone()
    env._goal_last_x_deg = x_deg.clone()