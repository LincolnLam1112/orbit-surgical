import gymnasium as gym

from . import agents, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Correct-Orientation-Needle-Dual-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleOrientationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CorrOrientationPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Correct-Orientation-Needle-Dual-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleOrientationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CorrOrientationPPORunnerCfg,
    },
    disable_env_checker=True,
)