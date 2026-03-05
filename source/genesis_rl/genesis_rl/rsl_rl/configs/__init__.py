"""Config namespace for GenesisLab + RSL-RL integration.

This mirrors the structure of IsaacLab's ``isaaclab_rl.rsl_rl.rl_cfg`` by
exposing configclasses for:

- PPO actor-critic networks
- PPO algorithm hyperparameters
- On-policy runner configuration
"""

from .policy_cfg import RslRlPpoActorCriticCfg
from .algo_cfg import RslRlPpoAlgorithmCfg
from .runner_cfg import RslRlOnPolicyRunnerCfg

__all__ = [
    "RslRlPpoActorCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlOnPolicyRunnerCfg",
]

