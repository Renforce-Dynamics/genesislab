"""H1 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import UnitreeH1FlatEnvCfg, UnitreeH1FlatEnvCfg_PLAY
from .rough_env_cfg import UnitreeH1RoughEnvCfg, UnitreeH1RoughEnvCfg_PLAY

__all__ = [
    "UnitreeH1FlatEnvCfg",
    "UnitreeH1FlatEnvCfg_PLAY",
    "UnitreeH1RoughEnvCfg",
    "UnitreeH1RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Unitree-H1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeH1FlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Unitree-H1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeH1FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Unitree-H1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeH1RoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Unitree-H1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeH1RoughEnvCfg_PLAY",
    },
)
