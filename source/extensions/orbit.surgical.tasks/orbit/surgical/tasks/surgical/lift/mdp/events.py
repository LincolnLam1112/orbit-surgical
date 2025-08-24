import torch
from isaaclab.envs.mdp import reset_scene_to_default as base_reset_scene_to_default
from isaaclab.envs import ManagerBasedEnv


def reset_scene_with_reward_reset(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Custom Orbit Surgical reset: base scene reset + gripper reset + phase flag reset."""

    # 1. Run base scene reset (from Isaac Lab)
    base_reset_scene_to_default(env, env_ids)

    # 2. Set grippers to be open for PSM tool, if present
    for articulation in env.scene.articulations.values():
        joint_names = articulation.joint_names
        if "psm_tool_gripper1_joint" in joint_names and "psm_tool_gripper2_joint" in joint_names:
            j1 = joint_names.index("psm_tool_gripper1_joint")
            j2 = joint_names.index("psm_tool_gripper2_joint")
            default_joint_pos = articulation.data.default_joint_pos[env_ids].clone()
            default_joint_vel = articulation.data.default_joint_vel[env_ids].clone()
            default_joint_pos[:, j1] = -0.5
            default_joint_pos[:, j2] = 0.5
            articulation.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    # 3. Reset phase flags if using stateful reward
    if hasattr(env.reward_manager, "pickup"):
        reward_fn = env.reward_manager.pickup.func
        if hasattr(reward_fn, "reset"):
            reward_fn.reset(env_ids)

