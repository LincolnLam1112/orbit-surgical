from isaaclab.envs import ManagerBasedEnv


def needle_below_table(env: ManagerBasedEnv, env_ids=None):
    z = env.scene["object"].data.root_pos_w[:, 2]
    is_below = z < 0.07
    return is_below
