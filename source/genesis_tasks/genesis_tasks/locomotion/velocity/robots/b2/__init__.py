"""B2 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import UnitreeB2FlatEnvCfg, UnitreeB2FlatEnvCfg_PLAY
from .rough_env_cfg import UnitreeB2RoughEnvCfg, UnitreeB2RoughEnvCfg_PLAY

__all__ = [
    "UnitreeB2FlatEnvCfg",
    "UnitreeB2FlatEnvCfg_PLAY",
    "UnitreeB2RoughEnvCfg",
    "UnitreeB2RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Unitree-B2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2FlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Unitree-B2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Unitree-B2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Unitree-B2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughEnvCfg_PLAY",
    },
)
