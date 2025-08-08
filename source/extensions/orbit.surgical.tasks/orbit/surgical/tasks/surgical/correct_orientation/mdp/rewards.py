# phased_orientation_reward.py

import torch
from .visualization import _contact_point_world, ee_contact_point_world, quat_to_rot_matrix, visualize_contact_point, visualize_ee_contact_point, visualize_goal_point, visualize_gripper_tips, visualize_needle_center_contact, visualize_needle_contact_left_right, visualize_needle_fixed_contact_relative, visualize_reference_path, visualize_gripper_links, visualize_reference_path_2
from .shared_phase_flags import get_mode_flags
from .path_generator import LinearPathGenerator
from isaacsim.core.utils.prims import delete_prim


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

    offset_local = torch.tensor([0.0005, 0.001, 0.0], device=device).expand(N, 3)
    offset_world = torch.bmm(needle_rot, offset_local.unsqueeze(-1)).squeeze(-1)
    needle_center = contact_center + offset_world
    offset_blue_local = torch.tensor([0.0, 0.0, 0.0045], device=device).expand(N, 3)
    offset_yellow_local = torch.tensor([0.0, 0.0, -0.0045], device=device).expand(N, 3)
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

        # for i in range(10):
        #     delete_prim(f"/Visuals/PathDotStep_center_{i}")
        #     delete_prim(f"/Visuals/PathDotStep_blue_{i}")
        #     delete_prim(f"/Visuals/PathDotStep_yellow_{i}")

        # visualize_reference_path(env)

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
    center_good = center_dist < 0.0095
    blue_good = blue_dist < 0.011
    yellow_good = yellow_dist < 0.011

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
    yaw_idx   = robot_data.joint_names.index("psm_tool_yaw_joint")
    pitch_idx = robot_data.joint_names.index("psm_tool_pitch_joint")
    roll_idx  = robot_data.joint_names.index("psm_tool_roll_joint")
    yaw_vals  = robot_data.joint_pos[:, yaw_idx]           # (N,)
    pitch_vals = robot_data.joint_pos[:, pitch_idx]         # (N,)
    roll_vals = robot_data.joint_pos[:, roll_idx]           # (N,)
    blue_grip = robot_data.joint_pos[:, blue_idx]
    yellow_grip = robot_data.joint_pos[:, yellow_idx]
    gripper_open = (blue_grip > 0.45) & (yellow_grip < -0.45)

    # per-env relative goal: base is each env's origin, plus fixed offset
    relative_offset = torch.tensor([-0.1863, 0.1419, 0.1296], device=device)  # in world units
    goal_point = env.unwrapped.scene.env_origins + relative_offset.unsqueeze(0)  # (N,3)

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

    live_align = (live_center_dist < 0.008) & (live_blue_dist < 0.005) & (live_yellow_dist < 0.005)
    live_align_2 = (live_center_dist < 0.02) & (live_blue_dist < 0.02) & (live_yellow_dist < 0.02)

    if not hasattr(env, "mode_flags"):
        env.mode_flags = torch.zeros(N, dtype=torch.long, device=device)

    # Compute Y-axis offset relative to goal
    needle_offset_y = needle_center[:, 1] - goal_point[:, 1]

    # +1 if needle is LEFT of goal (positive y), -1 if RIGHT (negative y)
    if not hasattr(env, "needle_side_flag"):
        env.needle_side_flag = torch.zeros(N, dtype=torch.long, device=device)

    env.needle_side_flag = torch.where(
        needle_offset_y < 0.0,
        torch.ones_like(env.needle_side_flag),     # needle is LEFT of goal
        -torch.ones_like(env.needle_side_flag)     # needle is RIGHT of goal
    )

    # Handle env_ids safely
    env_ids = torch.arange(N, device=device)  # use all envs

    # --- Reward shaping ---
    reward = torch.zeros(env.num_envs, device=env.device)
    cost = torch.zeros_like(reward)

    mode_0 = (env.mode_flags[env_ids] == 0)
    mode_1 = (env.mode_flags[env_ids] == 1)

    # --- Phase 0 logic ---
    if mode_0.any():
        active_ids_0 = env_ids[mode_0]
        final_reached = (env.current_path_index[active_ids_0] == max_step)
        reward[active_ids_0] += 1.0 * torch.exp(-center_dist[active_ids_0] / 0.0095).squeeze(-1)
        reward[active_ids_0] += 0.5 * torch.exp(-blue_dist[active_ids_0] / 0.011).squeeze(-1)
        reward[active_ids_0] += 0.5 * torch.exp(-yellow_dist[active_ids_0] / 0.011).squeeze(-1)
        reward[active_ids_0] += 1.0 * advance[active_ids_0].float()
        reward[active_ids_0] += 3.0 * final_reached.float()

        reward[active_ids_0] += 3.0 * torch.exp(-live_center_dist[active_ids_0] / 0.0075).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 3.0 * torch.exp(-live_blue_dist[active_ids_0] / 0.01).squeeze(-1) * final_reached.float()
        reward[active_ids_0] += 3.0 * torch.exp(-live_yellow_dist[active_ids_0] / 0.01).squeeze(-1) * final_reached.float()

        final_step = live_align[active_ids_0] & final_reached
        reward[active_ids_0] += 5.0 * final_step.float()

        needle_drift = torch.norm(obj.data.root_lin_vel_w[active_ids_0], dim=1)
        drift_penalty_mask = (needle_drift > 0.05) & live_align_2[active_ids_0]
        move_needle = active_ids_0[drift_penalty_mask]
        reward[move_needle] -= 1.0

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

        side_flag = env.needle_side_flag

        if needs_path_2.any():
            env.path_generator_2 = getattr(env, "path_generator_2", LinearPathGenerator(num_steps=3))

            left_need = needs_path_2 & (side_flag == 1)
            right_need = needs_path_2 & (side_flag == -1)

            # --- Ensure secondary path buffers exist ---
            if not hasattr(env, "reference_path_2"):
                env.reference_path_2 = torch.zeros((N, 3, 3), device=device)  # 3 steps default
            if not hasattr(env, "current_path_index_2"):
                env.current_path_index_2 = torch.zeros(N, dtype=torch.long, device=device)
            if not hasattr(env, "path_initialized_2"):
                env.path_initialized_2 = torch.zeros(N, dtype=torch.bool, device=device)

            if left_need.any():
                left_indices = torch.nonzero(left_need, as_tuple=False).squeeze(-1)  # global IDs
                new_paths_left = env.path_generator_2.generate(contact_blue[left_indices], goal_point[left_indices])
                env.reference_path_2[left_indices] = new_paths_left
                env.current_path_index_2[left_indices] = 0
                env.path_initialized_2[left_indices] = True

            if right_need.any():
                right_indices = torch.nonzero(right_need, as_tuple=False).squeeze(-1)  # global IDs
                new_paths_right = env.path_generator_2.generate(contact_yellow[right_indices], goal_point[right_indices])
                env.reference_path_2[right_indices] = new_paths_right
                env.current_path_index_2[right_indices] = 0
                env.path_initialized_2[right_indices] = True

            # for i in range(3):
            #     delete_prim(f"/Visuals/PathDotStep_orient_{i}")

            # visualize_reference_path_2(env)

        # then use active_ids normally
        idx_2 = torch.clamp(env.current_path_index_2, max=2)
        ref_center_2 = env.reference_path_2[torch.arange(N), idx_2]

        # --- Correct pushing tip (decided by needle_side_flag) ---
        left_tip_dist_to_path = torch.norm(blue_tip - ref_center_2, dim=1)[active_ids_1]
        right_tip_dist_to_path = torch.norm(yellow_tip - ref_center_2, dim=1)[active_ids_1]
        needle_dist_to_path = torch.norm(needle_center - ref_center_2, dim=1)[active_ids_1]
        pushing_left_mask = (env.needle_side_flag[active_ids_1] == 1)
        tip_dist_to_path = torch.where(
            pushing_left_mask,
            left_tip_dist_to_path,
            right_tip_dist_to_path
        )

        # center_close = goal_dist_center < 0.003
        tip_close = (tip_dist_to_path < 0.005)
        advance_2 = tip_close

        # --- Path index increment ---
        max_step_2 = env.reference_path_2.shape[1] - 1
        env.current_path_index_2[active_ids_1] = torch.minimum(
            env.current_path_index_2[active_ids_1] + advance_2.long(),
            torch.full_like(env.current_path_index_2[active_ids_1], max_step_2)
        )

        # --- Phase 1 reward shaping ---
        # 1) Alignment Reward
        blue_align = torch.exp(-live_blue_dist[active_ids_1] / 0.01)
        yellow_align = torch.exp(-live_yellow_dist[active_ids_1] / 0.01)

        alignment_reward = torch.where(
            pushing_left_mask,
            7.0 * blue_align + 5.0 * yellow_align,  # left: blue primary
            5.0 * blue_align + 7.0 * yellow_align   # right: yellow primary
        )

        alignment_reward_2 = torch.where(
            pushing_left_mask,
            15.0 * yellow_align,  # left: blue primary
            15.0 * blue_align  # right: yellow primary
        )

        center_align = torch.exp(-live_center_dist[active_ids_1] / 0.01)

        reward[active_ids_1] += 7.0 * center_align
        reward[active_ids_1] += alignment_reward

        reward[active_ids_1] += 25.0 * torch.exp(-tip_dist_to_path / 0.005)

        # 2) Zone bonus when tip is within threshold
        zone_bonus = torch.zeros_like(tip_dist_to_path)
        zone_bonus[tip_close] = 20.0
        reward[active_ids_1] += zone_bonus

        # 3) Advance bonus for stepping along the path
        reward[active_ids_1] += 25.0 * advance_2.float()

        # 4a) One-time alignment_reward_2 for any env that has reached the final index (ignoring advance_2)
        completed_mask = (env.current_path_index_2[active_ids_1] >= 1)  # shape (M,)
        if completed_mask.any():
            completed_ids = active_ids_1[completed_mask]          # global env IDs
            reward[completed_ids] += alignment_reward_2[completed_mask]

        # 4) Terminal bonus when final step reached
        final_reached_2 = (env.current_path_index_2[active_ids_1] == max_step_2) & advance_2
        reward[active_ids_1] += 30.0 * final_reached_2.float()

        yaw_phase = yaw_vals[active_ids_1]                     # (M,)
        side_mask = pushing_left_mask                          # (M,)

        # Normalize reward: 1.0 at 0, taper to 0.0 at ±0.5
        norm = (1.0 - torch.abs(yaw_phase) / 0.5).clamp(min=0.0)  # (M,)

        # Only reward in the correct half‐interval
        valid_left = (yaw_phase <= 0.0) & (yaw_phase >= -0.5)
        valid_right = (yaw_phase >= 0.0) & (yaw_phase <= 0.5)
        valid = torch.where(side_mask, valid_left, valid_right)

        joint_reward = norm * valid.float()  # (M,)

        # Add it (scale as desired)
        reward[active_ids_1] += 10.0 * joint_reward

        if final_reached_2.any():
            to2 = active_ids_1[final_reached_2]
            if not hasattr(env, "check_orient"):
                env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
            env.check_orient[to2] = 1

        # print(live_align_2, live_center_dist, live_blue_dist, live_yellow_dist)
        # print(env.current_path_index_2, tip_dist_to_path, final_reached_2, live_center_dist, reward)

    # print them
    # print("psm_tool_yaw_joint positions:", yaw_vals)
    # print("psm_tool_pitch_joint positions:", pitch_vals)
    # print("psm_tool_roll_joint positions:", roll_vals)

    # if not hasattr(env, "check_reached"):
    #     env.check_reached = torch.zeros(N, dtype=torch.long, device=device)
    # if not hasattr(env, "check_orient"):
    #     env.check_orient = torch.zeros(N, dtype=torch.long, device=device)
    # print(env.check_reached, env.check_orient)

    reward += 0.35 * gripper_open.float()
    pitch_norm = ((pitch_vals + 0.5) / 0.5).clamp(0.0, 1.0)      # 1.0 at pitch=0, 0.0 at pitch=–0.5

    sigma = 0.1
    roll_reward = torch.exp(- (roll_vals / sigma) ** 2)

    reward += 5.0 * roll_reward

    # scale reward as desired (e.g. max +5 when perfectly at 0)
    reward += 0.5 * pitch_norm

    # wrong_side_penalty = live_align_2 & ((wrong_dist_blue > live_blue_dist) | (wrong_dist_yellow > live_yellow_dist))
    # reward[wrong_side_penalty] += 0.5

    # Z penalty for dropping
    z_drop_orient = (needle_z < 0.13) | (needle_z > 0.145)
    drop_ids = torch.nonzero(z_drop_orient, as_tuple=False).squeeze(-1)
    reward[drop_ids] -= 2.0

    # print(env.mode_flags)

    # Optional debug
    # print(f"Step {step[0].item()}, dist_c: {center_dist[0].item():.4f}, dist_b: {blue_dist[0].item():.4f}, dist_y: {yellow_dist[0].item():.4f}")
    # visualize_ee_contact_point(env)
    # visualize_gripper_tips(env)
    # visualize_needle_center_contact(env)
    # visualize_needle_contact_left_right(env)
    # visualize_goal_point(env)

    return reward


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular difference (in radians) between two quaternions.
    """
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(torch.abs(dot))
