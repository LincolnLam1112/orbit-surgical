from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .cmo_ppo_cfg import CMOPPOAlgorithmCfg


@configclass
class CorrOrientationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "correct_needle_orientation"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
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
