# phased_orientation_reward.py


import torch
from .visualization import _contact_point_world, ee_contact_point_world, quat_to_rot_matrix, visualize_contact_point, visualize_ee_contact_point, visualize_goal_point, visualize_gripper_tips, visualize_needle_center_contact, visualize_needle_contact_left_right, visualize_needle_fixed_contact_relative, visualize_reference_path, visualize_gripper_links, visualize_reference_path_2
from .shared_phase_flags import get_mode_flags
from .path_generator import LinearPathGenerator
from isaacsim.core.utils.prims import delete_prim




def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
   """
   Compute angular difference (in radians) between two quaternions.
   """
   dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
   return 2.0 * torch.acos(torch.abs(dot))




def path_following_reward(env):
   # --- Generate path on first use per env ---
   needs_path = ~env.path_initialized
   N = env.num_envs
   device = env.device
   _ = get_mode_flags(env)


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
       # print(ee_center, needle_center)


       # print("Path points:", env.reference_path[0])  # Should print 10 distinct 3D points


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
   blue_grip = robot_data.joint_pos[:, blue_idx]
   yellow_grip = robot_data.joint_pos[:, yellow_idx]
   gripper_open = (blue_grip > 0.45) & (yellow_grip < -0.45)


   goal_point = torch.tensor([-0.1863, 0.1419, 0.1296], device=device).expand(N, 3)
   needle_z = obj.data.root_pos_w[:, 2]


   # --- Advancement logic based on phase ---
   advance = torch.where(
       pre_phase,
       center_good,
       torch.where(
           mid_phase,
           center_good & blue_good & yellow_good,
           final_phase
       )
   )


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


   live_align = (live_center_dist < 0.006) & (live_blue_dist < 0.0075) & (live_yellow_dist < 0.0075)
   live_align_2 = (live_center_dist < 0.012) & (live_blue_dist < 0.012) & (live_yellow_dist < 0.012)


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
       reward[active_ids_0] += 1.0 * torch.exp(-center_dist[active_ids_0] / 0.0095).squeeze(-1)
       reward[active_ids_0] += 0.5 * torch.exp(-blue_dist[active_ids_0] / 0.011).squeeze(-1)
       reward[active_ids_0] += 0.5 * torch.exp(-yellow_dist[active_ids_0] / 0.011).squeeze(-1)
       reward[active_ids_0] += 1.0 * advance[active_ids_0].float()
       final_reached = (env.current_path_index[active_ids_0] == max_step)
       reward[active_ids_0] += 2.0 * final_reached.float()


       reward[active_ids_0] += 1.0 * torch.exp(-live_center_dist[active_ids_0] / 0.01).squeeze(-1) * final_reached.float()
       reward[active_ids_0] += 0.5 * torch.exp(-live_blue_dist[active_ids_0] / 0.012).squeeze(-1) * final_reached.float()
       reward[active_ids_0] += 0.5 * torch.exp(-live_yellow_dist[active_ids_0] / 0.012).squeeze(-1) * final_reached.float()


       final_step = live_align[active_ids_0] & final_reached
       reward[active_ids_0] += 5.0 * final_step.float()


       # print(env.current_path_index, live_align, live_center_dist, live_blue_dist, live_yellow_dist, final_reached, reward)


       # # Cost
       # needle_drift = torch.norm(obj.data.root_lin_vel_w[active_ids_0], dim=1)
       # drift_penalty_mask = (needle_drift > 0.05) & (center_dist[active_ids_0].squeeze(-1) < 0.01)
       # cost[active_ids_0[drift_penalty_mask]] += 1.0


       # --- Ensure secondary path buffers exist ---
       if not hasattr(env, "reference_path_2"):
           env.reference_path_2 = torch.zeros((N, 3, 3), device=device)  # 3 steps default
       if not hasattr(env, "current_path_index_2"):
           env.current_path_index_2 = torch.zeros(N, dtype=torch.long, device=device)
       if not hasattr(env, "path_initialized_2"):
           env.path_initialized_2 = torch.zeros(N, dtype=torch.bool, device=device)


       # Build needs_path_2 only for active_ids_0
       needs_path_sub = (~env.path_initialized_2[active_ids_0]) & (
           live_align[active_ids_0] & (env.current_path_index[active_ids_0] == max_step)
       )


       # Expand it back to full mask
       needs_path_2 = torch.zeros(N, dtype=torch.bool, device=device)
       needs_path_2[active_ids_0] = needs_path_sub


       if needs_path_2.any():
           # Directly use the global mask to pick the correct env indices
           phase0_ids = torch.where(needs_path_2)[0]


           env.path_generator_2 = getattr(env, "path_generator_2", LinearPathGenerator(num_steps=3))
           new_paths_2 = env.path_generator_2.generate(
               needle_center[phase0_ids],
               goal_point[phase0_ids]
           )


           env.reference_path_2[phase0_ids] = new_paths_2
           env.current_path_index_2[phase0_ids] = 0
           env.path_initialized_2[phase0_ids] = True


           # for i in range(3):
           #     delete_prim(f"/Visuals/PathDotStep_orient_{i}")


           # visualize_reference_path_2(env)


       if final_step.any():
           # restrict final_step mask to only active_ids_0
           active_ids = active_ids_0[final_step]   # now safe


           # then use active_ids normally
           idx_2 = torch.clamp(env.current_path_index_2[active_ids], max=2)


           # If path not initialized, fallback to goal_point to avoid NaNs
           ref_center_2 = torch.where(
               env.path_initialized_2[active_ids].unsqueeze(-1),
               env.reference_path_2[active_ids, idx_2],
               goal_point[active_ids]
           )


           # Compute distances ONLY for active envs
           goal_dist_center = torch.norm(needle_center[active_ids] - ref_center_2, dim=1)
           left_tip_dist_to_path = torch.norm(blue_tip[active_ids] - ref_center_2, dim=1)
           right_tip_dist_to_path = torch.norm(yellow_tip[active_ids] - ref_center_2, dim=1)


           # Side mask only for these envs
           pushing_left_mask = (env.needle_side_flag[active_ids] == 1)
           tip_dist_to_path = torch.where(
               pushing_left_mask,
               left_tip_dist_to_path,
               right_tip_dist_to_path
           )
           tip_dist_to_needle = torch.where(
               pushing_left_mask,
               live_blue_dist[active_ids],
               live_yellow_dist[active_ids]
           )


           # # Thresholds
           # tip_close_path = tip_dist_to_path < 0.0075
           # tip_close_needle = tip_dist_to_needle < 0.01
           # transition_ready = tip_close_path & tip_close_needle


           # # Reward only for those active envs
           # reward[active_ids] += 2.0 * transition_ready.float()


           # # Transition only valid envs
           # env.mode_flags[active_ids[transition_ready]] = 1


           # if not hasattr(env, "prev_mode_flags"):
           #     # initialize on very first call
           #     env.prev_mode_flags = torch.zeros_like(env.mode_flags)


           # prev_flags = env.prev_mode_flags  # flags *before* this update
           # just_switched = (prev_flags == 0) & (env.mode_flags == 1)
           # reward[just_switched] += 10.0
           # # now update prev_mode_flags for next step
           # env.prev_mode_flags = env.mode_flags.clone()


   # # --- Phase 1 logic ---
   # if mode_1.any():
   #     active_ids_1 = env_ids[mode_1]


   #     # print(env.needle_side_flag, tip_dist_to_path, left_tip_dist_to_path, right_tip_dist_to_path)


   #     # --- ADVANCE CONDITION ---
   #     # Must satisfy BOTH:
   #     # 1) Needle center close to path
   #     # 2) Correct gripper tip close to same path point


   #     # Distance from needle center to path target (safe even if no path)


   #     idx_2 = torch.clamp(env.current_path_index_2, max=2)


   #     # If path not initialized, fallback to goal_point to avoid NaNs
   #     ref_center_2 = torch.where(
   #         env.path_initialized_2.unsqueeze(-1),
   #         env.reference_path_2[torch.arange(N), idx_2],
   #         goal_point
   #     )


   #     goal_dist_center = torch.norm(needle_center - ref_center_2, dim=1)


   #     # --- Correct pushing tip (decided by needle_side_flag) ---
   #     left_tip_dist_to_path = torch.norm(blue_tip - ref_center_2, dim=1)
   #     right_tip_dist_to_path = torch.norm(yellow_tip - ref_center_2, dim=1)


   #     pushing_left_mask = (env.needle_side_flag == 1)
   #     pushing_right_mask = (env.needle_side_flag == -1)


   #     tip_dist_to_path = torch.where(
   #         pushing_left_mask,
   #         left_tip_dist_to_path,
   #         right_tip_dist_to_path
   #     )
   #     tip_dist_to_needle = torch.where(
   #         pushing_left_mask,
   #         live_blue_dist,
   #         live_yellow_dist
   #     )


   #     center_close = goal_dist_center < 0.003
   #     tip_close = (tip_dist_to_path < 0.0025) & (tip_dist_to_needle < 0.02)


   #     advance_2 = center_close & tip_close  # BOTH must be true


   #     # --- Path index increment ---
   #     max_step_2 = env.reference_path_2.shape[1] - 1
   #     env.current_path_index_2 = torch.minimum(
   #         env.current_path_index_2 + advance_2.long(),
   #         torch.full_like(env.current_path_index_2, max_step_2)
   #     )


   #     # --- Phase 1 reward shaping ---
   #     # Baseline reward: stay aligned, but SMALL
   #     reward[active_ids_1] += 0.5 * live_align_2[active_ids_1].float()


   #     # Extra reward when aligned *and* close to path
   #     reward[active_ids_1] += 1.0 * torch.exp(-goal_dist_center[active_ids_1] / 0.012).squeeze(-1) * tip_close[active_ids_1].float()
   #     reward[active_ids_1] += 2.0 * torch.exp(-tip_dist_to_path[active_ids_1] / 0.012).squeeze(-1)


   #     # Binary bonus for really close
   #     reward[active_ids_1] += 2.0 * (tip_dist_to_path[active_ids_1] < 0.002).float()
   #     reward[active_ids_1] += 2.0 * (goal_dist_center[active_ids_1] < 0.0012).float() * tip_close[active_ids_1].float()


   #     # Progress reward: only if actually pushing forward
   #     if not hasattr(env, "prev_tip_dist"):
   #         env.prev_tip_dist = tip_dist_to_path.clone().detach()
   #     if not hasattr(env, "prev_center_dist"):
   #         env.prev_center_dist = goal_dist_center.clone().detach()


   #     prev_tip = env.prev_tip_dist[active_ids_1]
   #     prev_center = env.prev_center_dist[active_ids_1]


   #     curr_tip = tip_dist_to_path[active_ids_1]
   #     curr_center = goal_dist_center[active_ids_1]


   #     # Positive if moving closer
   #     progress_tip = (prev_tip - curr_tip).clamp(min=0.0)
   #     progress_center = (prev_center - curr_center).clamp(min=0.0)


   #     # BIGGEST reward → making actual progress toward pushing the needle
   #     reward[active_ids_1] += 6.0 * progress_tip
   #     reward[active_ids_1] += 6.0 * progress_center


   #     # Update for next step
   #     env.prev_tip_dist[active_ids_1] = tip_dist_to_path[active_ids_1].clone().detach()
   #     env.prev_center_dist[active_ids_1] = goal_dist_center[active_ids_1].clone().detach()


   #     # Reward advancing BOTH center + tip
   #     reward[active_ids_1] += 5.0 * advance_2[active_ids_1].float() * live_align_2[active_ids_1].float()


   #     # Final reached condition (center + tip reached last path step)
   #     final_reached_2 = (env.current_path_index_2[active_ids_1] == max_step_2) & advance_2[active_ids_1]
   #     reward[active_ids_1] += 12.0 * final_reached_2.float()


   #     needle_drift = torch.norm(obj.data.root_lin_vel_w[active_ids_1], dim=1)
   #     drift_penalty_mask = (needle_drift > 0.05) & (center_dist[active_ids_1] < 0.01)
   #     reward[active_ids_1][drift_penalty_mask] += 0.75


   #     # print(live_align_2, live_center_dist, live_blue_dist, live_yellow_dist)
   #     # print(env.current_path_index_2, goal_dist_center, tip_dist_to_path, tip_dist_to_needle, final_reached_2, reward)


   # reward += 0.35 * gripper_open.float()


   # Z penalty for dropping
   z_drop_orient = (needle_z < 0.13) | (needle_z > 0.145)
   cost[z_drop_orient] += 1.0


   # print(env.mode_flags)


   # print(drift_penalty_mask, needle_drift)


   # Optional debug
   # print(f"Step {step[0].item()}, dist_c: {center_dist[0].item():.4f}, dist_b: {blue_dist[0].item():.4f}, dist_y: {yellow_dist[0].item():.4f}")
   # visualize_ee_contact_point(env)
   # visualize_gripper_tips(env)
   # visualize_needle_center_contact(env)
   # visualize_needle_contact_left_right(env)
   # visualize_goal_point(env)
   # print(env.mode_flags)


   return reward, cost




def phased_orientation_reward_and_cost(env, ee_frame_cfg, object_cfg, robot_cfg):
   device = env.device
   N = env.num_envs


   # --- Entities ---
   obj = env.scene[object_cfg.name]
   robot = env.scene[robot_cfg.name]
   ee_frame = env.scene[ee_frame_cfg.name]
   mode_flags = get_mode_flags(env)


   # --- Needle Pose ---
   obj_pos_w = obj.data.root_pos_w
   obj_quat_w = obj.data.root_quat_w
   needle_rot = quat_to_rot_matrix(obj_quat_w)
   contact_center = _contact_point_world(obj_pos_w, obj_quat_w, device)


   # --- Contact Points ---
   offset_blue_local = torch.tensor([0.0, 0.0, 0.0045], device=device).expand(N, 3)
   offset_yellow_local = torch.tensor([0.0, 0.0, -0.0045], device=device).expand(N, 3)
   offset_center_local = torch.tensor([0.0005, 0.001, 0.0], device=device).expand(N, 3)
   contact_blue = contact_center + torch.bmm(needle_rot, offset_blue_local.unsqueeze(-1)).squeeze(-1)
   contact_yellow = contact_center + torch.bmm(needle_rot, offset_yellow_local.unsqueeze(-1)).squeeze(-1)
   contact_center_ee = contact_center + torch.bmm(needle_rot, offset_center_local.unsqueeze(-1)).squeeze(-1)


   # --- Gripper Tips ---
   base_grip_pos = ee_contact_point_world(ee_frame, device)
   grip_quat = ee_frame.data.target_quat_w[..., 0, :]
   grip_rot = quat_to_rot_matrix(grip_quat)
   blue_tip = base_grip_pos + torch.bmm(grip_rot, torch.tensor([0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
   yellow_tip = base_grip_pos + torch.bmm(grip_rot, torch.tensor([-0.0045, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)
   center_ee_pt = base_grip_pos + torch.bmm(grip_rot, torch.tensor([0.0, 0.0, 0.0], device=device).expand(N, 3).unsqueeze(-1)).squeeze(-1)


   # --- Distances ---
   blue_dist = torch.norm(blue_tip - contact_blue, dim=1)
   yellow_dist = torch.norm(yellow_tip - contact_yellow, dim=1)
   center_dist = torch.norm(center_ee_pt - contact_center_ee, dim=1)
   blue_to_yellow = torch.norm(blue_tip - contact_yellow, dim=1)
   yellow_to_blue = torch.norm(yellow_tip - contact_blue, dim=1)


   # --- Phase Logic ---
   # Phase 0: Reach, Phase 1: Orient, Phase 2: Withdraw
   reach_mask = center_dist < 0.025
   fine_mask = (blue_dist < 0.012) & (yellow_dist < 0.012) & (center_dist < 0.008)
   success_mask = (blue_dist < 0.007) & (yellow_dist < 0.007) & (center_dist < 0.006)


   if not hasattr(env, "mode_phase"):
       env.mode_phase = torch.zeros(N, dtype=torch.long, device=device)


   newly_orient = success_mask & (env.mode_phase == 0)
   # env.mode_phase = torch.where(newly_orient, torch.ones_like(env.mode_phase), env.mode_phase)


   # simulate withdrawal after orient
   goal_point = torch.tensor([-0.1863, 0.1419, 0.1296], device=device).expand(N, 3)
   goal_dist = torch.norm(contact_center_ee - goal_point, dim=1)
   at_goal = goal_dist < 0.0025
   env.mode_phase = torch.where((env.mode_phase == 1) & at_goal, torch.full_like(env.mode_phase, 2), env.mode_phase)


   # --- Reward Buffer ---
   reward = torch.zeros(N, device=device)
   cost = torch.zeros(N, device=device)


   # PHASE 0: REACH
   reach_phase = env.mode_phase == 0
   reward[reach_phase] += 1.0 * torch.exp(-center_dist[reach_phase] / 0.02)
   reward[reach_phase] += 2.0 * torch.exp(-(blue_dist[reach_phase] + yellow_dist[reach_phase]) / 0.02) * fine_mask.float()
   reward[reach_phase] += 3.5 * success_mask[reach_phase].float()
   reward[reach_phase] += 0.25 * torch.exp(-(blue_dist[reach_phase] + yellow_dist[reach_phase] + center_dist[reach_phase]) / 0.1)


   # open gripper bonus
   robot_data = robot.data
   joint_names = robot_data.joint_names
   blue_idx = joint_names.index("psm_tool_gripper2_joint")
   yellow_idx = joint_names.index("psm_tool_gripper1_joint")
   blue_grip = robot_data.joint_pos[:, blue_idx]
   yellow_grip = robot_data.joint_pos[:, yellow_idx]
   gripper_open = (blue_grip > -0.48) & (yellow_grip < 0.48)
   reward[reach_phase] += 0.35 * gripper_open[reach_phase].float()


   # penalize needle movement in phase 0
   needle_drift = torch.norm(obj.data.root_lin_vel_w, dim=1)
   drift_penalty_mask = (needle_drift > 0.05) & reach_mask
   cost[drift_penalty_mask] += 0.1
   # print(needle_drift)


   # penalize Z drop
   needle_z = obj.data.root_pos_w[:, 2]
   z_drop = ((needle_z < 0.13) | (needle_z > 0.142)) & reach_mask
   cost[z_drop] += 0.05


   # PHASE 1: ORIENT
   orient_phase = env.mode_phase == 1
   reward[orient_phase] += 3.0 * success_mask[orient_phase].float()
   goal_bonus = 4.0 * torch.exp(-goal_dist[orient_phase] / 0.005)
   reward[orient_phase] += goal_bonus


   z_drop_orient = ((needle_z < 0.13) | (needle_z > 0.142))
   cost[orient_phase & z_drop_orient] += 0.025


   # contact point deviation
   fixed_offset = torch.tensor([0.003, 0.0102, 0.0007], device=device).expand(N, 3)
   current_contact_point = _contact_point_world(obj_pos_w, obj_quat_w, device, offset=fixed_offset)
   target_world_pos = torch.tensor([-0.2, 0.1435, 0.1505], device=device).expand(N, 3)
   contact_drift = torch.norm(current_contact_point - target_world_pos, dim=1)
   cost[orient_phase] += 0.01 * torch.tanh(contact_drift[orient_phase])


   # # PHASE 2: WITHDRAW
   # withdraw_phase = env.mode_phase == 2
   # dist_from_needle = torch.norm(base_grip_pos - obj.data.root_pos_w, dim=1)
   # reward[withdraw_phase] += 4.0 * (dist_from_needle[withdraw_phase] > 0.04).float()
   # cost[withdraw_phase] += 0.005 * (dist_from_needle[withdraw_phase] < 0.02).float()


   # General penalties (misaligned tip)
   cross_penalty_mask = ((blue_to_yellow < blue_dist) | (yellow_to_blue < yellow_dist)) & reach_mask
   cost[cross_penalty_mask] += 0.035


   return reward, cost