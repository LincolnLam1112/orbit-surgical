# phased_orientation_reward.py

import torch
from .visualization import visualize_fixed_point, _contact_point_world, ee_contact_point_world, quat_to_rot_matrix, visualize_contact_point, visualize_ee_contact_point, visualize_goal_point, visualize_gripper_tips, visualize_needle_center_contact, visualize_needle_contact_left_right, visualize_needle_fixed_contact_relative, visualize_reference_path, visualize_gripper_links, visualize_reference_path_2
from .shared_phase_flags import get_mode_flags
from .path_generator import LinearPathGenerator, ArcPathGenerator
from isaacsim.core.utils.prims import delete_prim
from .events import reset_goal_about_pivot_xz
from .joint_utils import _expand_env_path


def path_following_reward(env):
    # --- Generate path on first use per env ---
    needs_path = ~env.path_initialized
    N = env.num_envs
    device = env.device
    _ = get_mode_flags(env)
    needs_path_2 = torch.zeros(N, dtype=torch.bool, device=device)
    needs_path_2 = ~env.path_initialized_2

    # Get fresh positions
    ee_frame = env.scene["ee_1_frame"]
    robot = env.scene["robot_1"]
    ee_center = ee_contact_point_world(ee_frame, device)
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    grip_rot = quat_to_rot_matrix(grip_quat)
    blue_tip = ee_center + torch.bmm(grip_rot, torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    yellow_tip = ee_center + torch.bmm(grip_rot, torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    needle = env.scene["object"]
    needle_pos = needle.data.root_pos_w
    needle_quat = needle.data.root_quat_w
    needle_rot = quat_to_rot_matrix(needle_quat)
    contact_center = _contact_point_world(needle_pos, needle_quat, device)

    offset_local = torch.tensor([-0.025, 0.0, 0.0], device=device).expand(N, 3)
    offset_world = torch.bmm(needle_rot, offset_local.unsqueeze(-1)).squeeze(-1)
    needle_center = contact_center + offset_world
    offset_blue_local = torch.tensor([-0.025, 0.0, 0.0045], device=device).expand(N, 3)
    offset_yellow_local = torch.tensor([-0.025, 0.0, -0.0045], device=device).expand(N, 3)
    contact_blue = contact_center + torch.bmm(needle_rot, offset_blue_local.unsqueeze(-1)).squeeze(-1)
    contact_yellow = contact_center + torch.bmm(needle_rot, offset_yellow_local.unsqueeze(-1)).squeeze(-1)

    if needs_path.any():
        # Generate path only for environments that need it
        env.path_generator = getattr(env, "path_generator", LinearPathGenerator(num_steps=10))
        env.path_generator_blue = getattr(env, "path_generator_blue", LinearPathGenerator(num_steps=10))
        env.path_generator_yellow = getattr(env, "path_generator_yellow", LinearPathGenerator(num_steps=10))

        new_paths = env.path_generator.generate(ee_center[needs_path], needle_center[needs_path])
        new_paths_blue = env.path_generator_blue.generate(blue_tip[needs_path], contact_blue[needs_path])
        new_paths_yellow = env.path_generator_yellow.generate(yellow_tip[needs_path], contact_yellow[needs_path])

        if not hasattr(env, "reference_path"):
            env.reference_path = torch.zeros((N, 10, 3), device=device)
            env.reference_path_blue = torch.zeros((N, 10, 3), device=device)
            env.reference_path_yellow = torch.zeros((N, 10, 3), device=device)
        if not hasattr(env, "current_path_index"):
            env.current_path_index = torch.zeros(N, dtype=torch.long, device=device)
        if not hasattr(env, "path_initialized"):
            env.path_initialized = torch.zeros(N, dtype=torch.bool, device=device)

        env.reference_path[needs_path] = new_paths
        env.reference_path_blue[needs_path] = new_paths_blue
        env.reference_path_yellow[needs_path] = new_paths_yellow
        env.current_path_index[needs_path] = 0
        env.path_initialized[needs_path] = True

        # # Allocate once for ALL envs (not the masked count N)
        # if not hasattr(env, "reference_path"):
        #     M = env.num_envs
        #     env.reference_path        = torch.zeros((M, 10, 3), device=device)
        #     env.reference_path_blue   = torch.zeros((M, 10, 3), device=device)
        #     env.reference_path_yellow = torch.zeros((M, 10, 3), device=device)
        # if not hasattr(env, "current_path_index"):
        #     env.current_path_index = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        # if not hasattr(env, "path_initialized"):
        #     env.path_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        # # ... after assigning env.reference_path[...] = new_paths, etc.

        # # Delete old markers per-env that just got (re)generated
        # base_ns = getattr(env.scene, "env_ns", "/World/envs/env")
        # eids = torch.nonzero(needs_path, as_tuple=False).squeeze(-1).tolist()
        # for eid in eids:
        #     for i in range(10):
        #         delete_prim(f"{base_ns}_{eid}/Visuals/PathDotStep_center_{i}")
        #         delete_prim(f"{base_ns}_{eid}/Visuals/PathDotStep_blue_{i}")
        #         delete_prim(f"{base_ns}_{eid}/Visuals/PathDotStep_yellow_{i}")

        # # Draw for all envs (or just eids) after updates:
        # visualize_reference_path(env, env_ids=eids if len(eids) > 0 else None)

    # --- Get current targets ---
    idx = torch.clamp(env.current_path_index, max=9)
    ref_center = env.reference_path[torch.arange(N), idx]
    ref_blue = env.reference_path_blue[torch.arange(N), idx]
    ref_yellow = env.reference_path_yellow[torch.arange(N), idx]

    # --- Current gripper points ---
    obj = env.scene["object"]
    ee_center = ee_contact_point_world(ee_frame, env.device)
    grip_quat = ee_frame.data.target_quat_w[..., 0, :]
    grip_rot = quat_to_rot_matrix(grip_quat)
    blue_tip = ee_center + torch.bmm(grip_rot, torch.tensor([0.0045, 0.0, 0.0], device=env.device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
    yellow_tip = ee_center + torch.bmm(grip_rot, torch.tensor([-0.0045, 0.0, 0.0], device=env.device).expand(N, 3).unsqueeze(-1)).squeeze(-1)

    # --- Distances to targets ---
    center_dist = torch.norm(ee_center - ref_center, dim=1)
    blue_dist = torch.norm(blue_tip - ref_blue, dim=1)
    yellow_dist = torch.norm(yellow_tip - ref_yellow, dim=1)

    # --- Distance thresholds for alignment ---
    # center_good = center_dist < 0.0095
    # blue_good = blue_dist < 0.011
    # yellow_good = yellow_dist < 0.011

    center_good = center_dist < 0.05
    blue_good = blue_dist < 0.05
    yellow_good = yellow_dist < 0.05

    # --- Step-wise phase gating ---
    step = env.current_path_index
    pre_phase = step < 5
    mid_phase = (step >= 5) & (step <= 8)
    final_phase = step == 9
    post_phase = step > 9  # safety only

    robot_data = robot.data
    joint_names = robot_data.joint_names
    blue_idx = joint_names.index("psm_tool_gripper2_joint")
    yellow_idx = joint_names.index("psm_tool_gripper1_joint")
    blue_grip = robot_data.joint_pos[:, blue_idx]
    yellow_grip = robot_data.joint_pos[:, yellow_idx]
    gripper_open = (blue_grip > 0.45) & (yellow_grip < -0.45)

    # per-env relative goal: base is each env's origin, plus fixed offset

    if not hasattr(env, "goal_point"):
        reset_goal_about_pivot_xz(env)     # creates env.goal_point on first call
    goal_point = env.goal_point 

    # relative_offset = torch.tensor([-0.21, 0.1419, 0.065], device=device)  # in world units
    # goal_point = env.unwrapped.scene.env_origins + relative_offset.unsqueeze(0)  # (N,3)
    # visualize_fixed_point(env)
    # delete_prim(f"/Visuals/GoalPoint")
    # visualize_goal_point(env)

    needle_z = obj.data.root_pos_w[:, 2]

    # --- Advancement logic based on phase ---
    advance = torch.where(pre_phase, center_good, torch.where(mid_phase, center_good & blue_good & yellow_good, final_phase))

    # --- Path index increment ---
    max_step = env.reference_path.shape[1] - 1
    env.current_path_index = torch.minimum(
        env.current_path_index + advance.long(),
        torch.full_like(env.current_path_index, max_step)
    )

    # --- Compute live distances
    live_center_dist = torch.norm(ee_center - needle_center, dim=1)
    live_blue_dist = torch.norm(blue_tip - contact_blue, dim=1)
    live_yellow_dist = torch.norm(yellow_tip - contact_yellow, dim=1)
    wrong_dist_blue = torch.norm(blue_tip - contact_yellow, dim=1)
    wrong_dist_yellow = torch.norm(yellow_tip - contact_blue, dim=1)

    live_align = (live_center_dist < 0.01) & (live_blue_dist < 0.005) & (live_yellow_dist < 0.005)
    live_align_2 = (live_center_dist < 0.02) & (live_blue_dist < 0.02) & (live_yellow_dist < 0.02)

    if not hasattr(env, "mode_flags"):
        env.mode_flags = torch.zeros(N, dtype=torch.long, device=device)

    # Compute Y-axis offset relative to goal
    needle_offset_y = needle_center[:, 1] - goal_point[:, 1]

    # +1 if needle is LEFT of goal (positive y), -1 if RIGHT (negative y)
    if not hasattr(env, "needle_side_flag"):
        env.needle_side_flag = torch.zeros(N, dtype=torch.long, device=device)

    env.needle_side_flag = torch.where(
        (goal_point[:, 1] - needle_center[:, 1]) > 0.0,
        torch.ones_like(env.needle_side_flag),      # goal is LEFT (+Y)
        -torch.ones_like(env.needle_side_flag)      # goal is RIGHT (-Y)
    )

    # Pivot world (center of rotation) for arc paths
    _cfg = getattr(env, "cfg", None)
    _pivot_local = torch.tensor(
        getattr(getattr(getattr(_cfg, "scene", None), "needle_pivot_xform", None), "init_state", None).pos
        if (_cfg and hasattr(_cfg.scene.needle_pivot_xform, "init_state"))
        else (-0.200, 0.1435, 0.1),
        device=device,
    )
    pivot_world = env.scene.env_origins + _pivot_local  # (N,3)

    # Handle env_ids safely
    env_ids = torch.arange(N, device=device)  # use all envs

    # --- Reward shaping ---
    reward = torch.zeros(env.num_envs, device=env.device)

    mode_0 = (env.mode_flags[env_ids] == 0)
    mode_1 = (env.mode_flags[env_ids] == 1)

    # --- Phase 0 logic ---
    if mode_0.any():
        active_ids_0 = env_ids[mode_0]
        final_reached = (env.current_path_index[active_ids_0] == max_step)

        reward[active_ids_0] += 0.15 * torch.exp(-center_dist[active_ids_0] / 0.1).squeeze(-1)
        reward[active_ids_0] += 0.15 * torch.exp(-blue_dist[active_ids_0] / 0.1).squeeze(-1)
        reward[active_ids_0] += 0.15 * torch.exp(-yellow_dist[active_ids_0] / 0.1).squeeze(-1)

        reward[active_ids_0] += 1.5 * torch.exp(-center_dist[active_ids_0] / 0.008).squeeze(-1)
        reward[active_ids_0] += 1.5 * torch.exp(-blue_dist[active_ids_0] / 0.008).squeeze(-1)
        reward[active_ids_0] += 1.5 * torch.exp(-yellow_dist[active_ids_0] / 0.008).squeeze(-1)

        # reward[active_ids_0] += 1.0 * advance[active_ids_0].float()
        # reward[active_ids_0] += 3.0 * final_reached.float()

        reward[active_ids_0] += 0.05 * torch.exp(-live_center_dist[active_ids_0] / 0.02).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 0.05 * torch.exp(-live_blue_dist[active_ids_0] / 0.02).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 0.05 * torch.exp(-live_yellow_dist[active_ids_0] / 0.02).squeeze(-1) * final_reached.float()

        reward[active_ids_0] += 1.35 * torch.exp(-live_center_dist[active_ids_0] / 0.0085).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 1.35 * torch.exp(-live_blue_dist[active_ids_0] / 0.0085).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 1.35 * torch.exp(-live_yellow_dist[active_ids_0] / 0.0085).squeeze(-1) * final_reached.float()

        reward[active_ids_0] += 7.5 * torch.exp(-live_center_dist[active_ids_0] / 0.003).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 7.5 * torch.exp(-live_blue_dist[active_ids_0] / 0.003).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 7.5 * torch.exp(-live_yellow_dist[active_ids_0] / 0.003).squeeze(-1) * final_reached.float()

        final_step = live_align[active_ids_0] & final_reached
        # reward[active_ids_0] += 5.0 * final_step.float()
        # print(env.mode_flags, env.current_path_index)

        needle_drift = torch.norm(obj.data.root_lin_vel_w[active_ids_0], dim=1)
        drift_penalty_mask = (needle_drift > 0.05) & live_align_2[active_ids_0]
        move_needle = active_ids_0[drift_penalty_mask]
        reward[move_needle] -= 0.5

        # — one‑time “huge” bonus for exactly those envs that just unlocked Phase 1 —
        if final_step.any():
            to1 = active_ids_0[final_step]
            env.mode_flags[to1] = 1      # then move them into Phase 1
            if not hasattr(env, "check_reached"):
                env.check_reached = torch.zeros(N, dtype=torch.long, device=device)
            env.check_reached[to1] = 1

        # print(env.current_path_index, live_align, live_center_dist, live_blue_dist, live_yellow_dist, reward)
        # print(final_reached, live_align, live_center_dist, live_blue_dist, live_yellow_dist)

    # --- Phase 1 logic ---
    if mode_1.any():
        active_ids_1 = env_ids[mode_1]

        reward[active_ids_1] += 1.0

        side_flag = env.needle_side_flag
        env.path_generator_2 = getattr(env, "path_generator_2", LinearPathGenerator(num_steps=3))
        steps = getattr(env.path_generator_2, "num_steps", 3)

        if needs_path_2.any():
            # Arc path around pivot

            # masks
            left_need  = needs_path_2 & (side_flag == 1)
            right_need = needs_path_2 & (side_flag == -1)
            need_idx   = torch.nonzero(needs_path_2, as_tuple=False).squeeze(-1)

            # --- Ensure ALL three Phase-1 buffers exist (center / blue / yellow) ---
            if (not hasattr(env, "reference_path_2_center")) or (env.reference_path_2_center.shape[1] != steps):
                env.reference_path_2_center = torch.zeros((N, steps, 3), device=device)
            if (not hasattr(env, "reference_path_2_blue")) or (env.reference_path_2_blue.shape[1] != steps):
                env.reference_path_2_blue = torch.zeros((N, steps, 3), device=device)
            if (not hasattr(env, "reference_path_2_yellow")) or (env.reference_path_2_yellow.shape[1] != steps):
                env.reference_path_2_yellow = torch.zeros((N, steps, 3), device=device)

            if not hasattr(env, "current_path_index_2"):
                env.current_path_index_2 = torch.zeros(N, dtype=torch.long, device=device)
            if not hasattr(env, "path_initialized_2"):
                env.path_initialized_2 = torch.zeros(N, dtype=torch.bool, device=device)

            # --- Goal offsets: left(+Y), right(-Y); center uses goal as-is ---
            goal_off_y = 0.0045
            goal_left  = goal_point.clone();  goal_left[:, 1]  -= goal_off_y
            goal_right = goal_point.clone();  goal_right[:, 1] += goal_off_y
            goal_center = goal_point

            # Generate CENTER path for all envs that need Phase-1 paths
            if need_idx.numel() > 0:
                env.reference_path_2_center[need_idx, :steps, :] = env.path_generator_2.generate(
                    ee_center[need_idx], goal_center[need_idx]
                )

                # blue path (left contact → left goal)
                env.reference_path_2_blue[need_idx, :steps, :] = env.path_generator_2.generate(
                    contact_blue[need_idx], goal_left[need_idx]
                )
                # yellow path (right contact → right goal)
                env.reference_path_2_yellow[need_idx, :steps, :] = env.path_generator_2.generate(
                    contact_yellow[need_idx], goal_right[need_idx]
                )

            # init indices once paths are ready
            env.current_path_index_2[need_idx] = 0
            env.path_initialized_2[need_idx] = True

        #     visualize_reference_path_2(env, which=("arc_center","arc_blue", "arc_yellow"))
        # print(env.check_reached, env.check_orient)

        # then use active_ids normally
        idx_2 = torch.clamp(env.current_path_index_2, max=steps - 1)
        ref_center_2_c = env.reference_path_2_center[torch.arange(N), idx_2]   # center→goal
        ref_center_2_b = env.reference_path_2_blue[torch.arange(N), idx_2]     # blue→goal+Y
        ref_center_2_y = env.reference_path_2_yellow[torch.arange(N), idx_2]   # yellow→goal−Y

# # === New ===

        # # Tangent of the arc at current index (next waypoint - current)
        # max_step_2 = env.reference_path_2.shape[1] - 1
        # next_idx_2 = torch.clamp(idx_2 + 1, max=max_step_2)
        # tangent_2 = env.reference_path_2[torch.arange(N), next_idx_2] - ref_center_2
        # tangent_2 = torch.nn.functional.normalize(tangent_2, dim=1)

        # # Needle-center velocity
        # obj = env.scene["object"]
        # v_center = obj.data.root_lin_vel_w  # (N,3)

        # # Signed tangential progress (only positive progress counts)
        # progress_along_arc = torch.sum(v_center * tangent_2, dim=1)  # (N,)
        
        # --- Correct pushing tip (decided by needle_side_flag) ---
        left_tip_dist_to_path_b = torch.norm(blue_tip - ref_center_2_b, dim=1)[active_ids_1]
        right_tip_dist_to_path_y = torch.norm(yellow_tip - ref_center_2_y, dim=1)[active_ids_1]
        needle_dist_to_path_c = torch.norm(needle_center - ref_center_2_c, dim=1)[active_ids_1]
        goal_dist = torch.norm(needle_center - goal_point, dim=1)[active_ids_1]
        pushing_left_mask = (env.needle_side_flag[active_ids_1] == 1)

        # tip_dist_to_path = torch.where(
        #     pushing_left_mask,
        #     left_tip_dist_to_path,
        #     right_tip_dist_to_path
        # )

        # center_close = goal_dist_center < 0.003
        # tip_close_b = (left_tip_dist_to_path_b < 0.0075)
        # tip_close_y = (right_tip_dist_to_path_y < 0.0075)
        tip_close_c = (needle_dist_to_path_c < 0.005)
        # advance_2 = tip_close_b & tip_close_y & tip_close_c
        advance_2 = tip_close_c
        # & (progress_along_arc[active_ids_1] > 0.0) # === New ===

        # --- Path index increment ---
        max_step_2 = env.reference_path_2.shape[1] - 1
        env.current_path_index_2[active_ids_1] = torch.minimum(
            env.current_path_index_2[active_ids_1] + advance_2.long(),
            torch.full_like(env.current_path_index_2[active_ids_1], max_step_2)
        )

        blue_align = torch.exp(-live_blue_dist[active_ids_1] / 0.005)
        yellow_align = torch.exp(-live_yellow_dist[active_ids_1] / 0.005)

        alignment_reward = 1.5 * blue_align + 1.5 * yellow_align

        center_align = torch.exp(-live_center_dist[active_ids_1] / 0.01)

        # blue_to_center = torch.norm(blue_tip - needle_center, dim=1)
        # yellow_to_center = torch.norm(yellow_tip - needle_center, dim=1)

        # blue_center_align = torch.exp(-blue_to_center[active_ids_1] / 0.01)
        # yellow_center_align = torch.exp(-yellow_to_center[active_ids_1] / 0.01)

        # reward[active_ids_1] += 1.5 * center_align
        # reward[active_ids_1] += alignment_reward

        # --- Direction-reversal penalty (stop left↔right sweeping) ---
        # Select the currently "pushing" tip position for active envs
        # push_tip_active = torch.where(
        #     pushing_left_mask.unsqueeze(-1),
        #     blue_tip[active_ids_1],          # (M,3)
        #     yellow_tip[active_ids_1]         # (M,3)
        # )

        # Persistent buffer to track previous pushing-tip position
        if not hasattr(env, "_prev_push_tip"):
            env._prev_push_tip = torch.zeros((N, 3), device=device)

        scale_table = torch.tensor([1.0, 1.75, 4.5], device=env.device)

        idx = env.current_path_index_2[active_ids_1].clamp(max=2)  # ensure 0–2

        coeff_scale = scale_table[idx]

        # --- Progressive path-following rewards ---
        # Base coefficients (used at path_index=0)
        base_tip_coeff_wide = 3.0
        base_tip_coeff_tight = 4.0
        base_needle_coeff_wide = 3.0
        base_needle_coeff_tight = 4.0

        # # Apply progressive scaling
        # reward[active_ids_1] += (base_goal_coeff * coeff_scale) * torch.exp(-goal_dist / 0.01)
        
        # Left tip (blue) rewards with progressive scaling
        reward[active_ids_1] += (base_tip_coeff_wide * coeff_scale) * torch.exp(-left_tip_dist_to_path_b / 0.012)
        reward[active_ids_1] += (base_tip_coeff_tight * coeff_scale) * torch.exp(-left_tip_dist_to_path_b / 0.005)
        
        # # Right tip (yellow) rewards with progressive scaling
        reward[active_ids_1] += (base_tip_coeff_wide * coeff_scale) * torch.exp(-right_tip_dist_to_path_y / 0.012)
        reward[active_ids_1] += (base_tip_coeff_tight * coeff_scale) * torch.exp(-right_tip_dist_to_path_y / 0.005)
        
        # Needle center rewards with progressive scaling
        reward[active_ids_1] += (base_needle_coeff_wide * coeff_scale) * torch.exp(-needle_dist_to_path_c / 0.012)
        reward[active_ids_1] += (base_needle_coeff_tight * coeff_scale) * torch.exp(-needle_dist_to_path_c / 0.005)

        # steady_progress_bonus = 10.0 * coeff_scale * advance_2.float() * (
        #     torch.exp(-left_tip_dist_to_path_b / 0.01) +
        #     torch.exp(-right_tip_dist_to_path_y / 0.01) +
        #     torch.exp(-needle_dist_to_path_c / 0.01)
        # ) / 3.0  # average of three alignment terms
        # reward[active_ids_1] += steady_progress_bonus

        # print(reward)

        # reward[active_ids_1] += 2.0 * torch.exp(-goal_dist / 0.01)

        # reward[active_ids_1] += 20.0 * torch.exp(-left_tip_dist_to_path_b / 0.02)
        # reward[active_ids_1] += 45.0 * torch.exp(-left_tip_dist_to_path_b / 0.008)

        # reward[active_ids_1] += 20.0 * torch.exp(-right_tip_dist_to_path_y / 0.02)
        # reward[active_ids_1] += 45.0 * torch.exp(-right_tip_dist_to_path_y / 0.008)

        # reward[active_ids_1] += 20.0 * torch.exp(-needle_dist_to_path_c / 0.02)
        # reward[active_ids_1] += 45.0 * torch.exp(-needle_dist_to_path_c / 0.008)

# === New ===
        # 4) Terminal bonus when final step reached
        final_reached_2 = (env.current_path_index_2[active_ids_1] == max_step_2) & (needle_dist_to_path_c < 0.0055)
        # reward[active_ids_1] += 50.0 * final_reached_2.float()

        if final_reached_2.any():
            to2 = active_ids_1[final_reached_2]
            if not hasattr(env, "check_orient"):
                env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
            env.check_orient[to2] = 1

        reward[active_ids_1] += 100.0 * final_reached_2.float()

        # print(live_align_2, live_center_dist, live_blue_dist, live_yellow_dist)
        # print(env.current_path_index_2, final_reached_2, live_center_dist, reward)

    # if not hasattr(env, "check_reached"):
    #     env.check_reached = torch.zeros(N, dtype=torch.long, device=device)
    # if not hasattr(env, "check_orient"):
    #     env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
    # print(env.check_reached, env.check_orient)

    # near_gate = (live_center_dist < 0.02).float()
    reward += 0.075 * gripper_open.float()

    # # wrong_side_penalty = live_align_2 & ((wrong_dist_blue > live_blue_dist) | (wrong_dist_yellow > live_yellow_dist))
    # # # reward[wrong_side_penalty] += 5.0

    # --- Minimal joint-movement penalty ---
    q = env.scene["robot_1"].data.joint_pos                # (N, DoF)
    if not hasattr(env, "_prev_q"):
        env._prev_q = q.clone()

    # NEW (scale by gate and normalize by DoF)
    dq = q - env._prev_q
    reward -= 0.0075 * torch.sum(dq * dq, dim=1)
    # reward -= 0.035 * torch.sum(dq * dq, dim=1)              # L2^2 penalty (tune 0.1)

    env._prev_q = q.clone()

    # --- Action smoothness penalties (Δu and jerk) ------------------------------
    # Hard-coded knobs (no launcher/config needed)
    LAM_DIFF = 0.1   # weight for first difference penalty  ||a_t - a_{t-1}||^2
    LAM_JERK = 0.075  # weight for second difference penalty ||a_t - 2a_{t-1} + a_{t-2}||^2

    # Get the concatenated action tensor from IsaacLab's ActionManager
    actions = getattr(getattr(env, "action_manager", None), "action", None)  # shape: [num_envs, act_dim]
    if actions is not None:
        # Initialize buffers on first call
        if not hasattr(env, "_sm_prev_action"):
            env._sm_prev_action = actions.clone()          # a_{t-1}
            env._sm_prev_prev_action = actions.clone()     # a_{t-2}

        # First difference (velocity of actions)
        diff = actions - env._sm_prev_action                     # a_t - a_{t-1}
        diff_sq = torch.sum(diff * diff, dim=1)                  # ||·||^2 per env

        # Second difference (discrete jerk of actions)
        jerk = actions - 2.0 * env._sm_prev_action + env._sm_prev_prev_action
        jerk_sq = torch.sum(jerk * jerk, dim=1)

        # NEW: much lighter + gated (and normalized by act_dim)
        act_dim = actions.shape[1]
        reward += (-LAM_DIFF) * (diff_sq / act_dim)
        reward += (-LAM_JERK) * (jerk_sq / act_dim)

        # Update history
        env._sm_prev_prev_action = env._sm_prev_action.clone()
        env._sm_prev_action = actions.clone()

        # Optional: lightweight telemetry for logging
        if not hasattr(env, "extras"):
            env.extras = {}
        env.extras["a_diff_l2_mean"] = diff_sq.mean().item()
        env.extras["a_jerk_l2_mean"] = jerk_sq.mean().item()

    reward += -0.1   # per step; small but breaks “do nothing”

    # --- Encourage nudging the needle toward the goal ----------------------------
    needle_vel = obj.data.root_lin_vel_w  # (N,3)
    to_goal = goal_point - needle_center
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=1)
    speed_toward_goal = torch.sum(needle_vel * to_goal_norm, dim=1)  # projection
    reward += 0.8 * torch.clamp(speed_toward_goal, min=0.0)
    
    # print(env.mode_flags)
    # print(reward)

    # # Optional debug
    # # print(f"Step {step[0].item()}, dist_c: {center_dist[0].item():.4f}, dist_b: {blue_dist[0].item():.4f}, dist_y: {yellow_dist[0].item():.4f}")
    # # visualize_ee_contact_point(env)
    # # visualize_gripper_tips(env)
    # visualize_needle_center_contact(env)
    # visualize_needle_contact_left_right(env)
    # visualize_goal_point(env)
    # print(env.needle_side_flag)

    return reward


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular difference (in radians) between two quaternions.
    """
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(torch.abs(dot))