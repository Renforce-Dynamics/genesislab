"""Go2 velocity tracking task for GenesisLab."""

from .go2_flat_env_cfg import Go2FlatVelocityEnvCfg
from .go2_rough_env_cfg import Go2RoughVelocityEnvCfg

__all__ = [
    "Go2FlatVelocityEnvCfg",
    "Go2RoughVelocityEnvCfg",
    "Go2VelocityEnv",
]

