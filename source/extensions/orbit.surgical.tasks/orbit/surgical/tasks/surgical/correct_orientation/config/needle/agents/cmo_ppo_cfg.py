from __future__ import annotations
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class CMOPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    cost_gamma: float = 0.99
    cost_lam: float = 0.95
    cost_value_loss_coef: float = 1.0
    lagrangian_lr: float = 0.01
    cost_limit: float = 0.007
    