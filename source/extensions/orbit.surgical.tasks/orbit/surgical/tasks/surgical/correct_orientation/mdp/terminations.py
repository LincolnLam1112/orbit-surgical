from isaaclab.envs import ManagerBasedEnv


def needle_below_table(env: ManagerBasedEnv, env_ids=None):
    z = env.scene["object"].data.root_pos_w[:, 2]
    is_below = z < 0.07
    return is_below


def reach_phase_success(env: ManagerBasedEnv, env_ids=None):
    # Phase 1 in mode_flags means reach completed
    return env.mode_flags == 1
