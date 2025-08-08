import torch

from isaaclab.managers import SceneEntityCfg
from .rewards import path_following_reward


class PhasedOrientationCMORewardWrapper:
    def __init__(self, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
        self.ee_frame_cfg = ee_frame_cfg
        self.object_cfg = object_cfg
        self.robot_cfg = robot_cfg

    def __call__(self, env):
        reward = path_following_reward(env)
        
        # # Make cost available to runner
        # env.extras["costs"] = cost

        return reward
