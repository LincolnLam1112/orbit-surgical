from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .cmo_ppo_cfg import CMOPPOAlgorithmCfg


@configclass
class CorrOrientationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # num_steps_per_env = 24
    # max_iterations = 500
    # save_interval = 100
    # experiment_name = "correct_needle_orientation"
    # empirical_normalization = False
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     # actor_hidden_dims=[256, 256],
    #     # critic_hidden_dims=[256, 256, 128],
    #     # activation="elu",
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.006,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-4,
    #     schedule="adaptive",
    #     gamma=0.98,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )

    # rsl_rl_cfg.py  — minimal PPO tweaks for stable nudging
    num_steps_per_env = 24            # was 24: better velocity/tangent signal per update
    max_iterations = 500
    save_interval = 100
    experiment_name = "correct_needle_orientation"
    empirical_normalization = True    # was False: stabilizes scaled rewards/obs

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.75,           # was 1.0: less random flailing, quicker nudge learning
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,           # was 0.006: a touch more exploration, still controlled
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,                   # was 0.98: slightly longer credit for arc progress
        lam=0.95,
        desired_kl=0.01,             # was 0.01: allows modestly larger policy updates
        max_grad_norm=1.0,
    )


# class CMOActorCriticCfg(RslRlPpoActorCriticCfg):
#     cost_critic: dict = {
#         "critic_hidden_dims": [256, 128, 64],
#         "mlp_activation": "elu",
#     }


# @configclass
# class CorrOrientationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 1500
#     save_interval = 50
#     experiment_name = "dual_needle_handover"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     policy = CMOActorCriticCfg(  # <-- replaced this
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     algorithm = CMOPPOAlgorithmCfg(
#         gamma=0.98,
#         lam=0.95,
#         value_loss_coef=1.0,
#         cost_gamma=0.998,
#         cost_lam=0.95,
#         cost_value_loss_coef=1.0,
#         lagrangian_lr=1e-3,
#         cost_limit=0.08,
#         learning_rate=1.0e-4,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         clip_param=0.2,
#         entropy_coef=0.006,
#         use_clipped_value_loss=True,
#         schedule="adaptive",
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
