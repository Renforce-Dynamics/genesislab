"""G1 23Dof velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import UnitreeG1_23DOFFlatEnvCfg, UnitreeG1_23DOFFlatEnvCfg_PLAY
from .rough_env_cfg import UnitreeG1_23DOFRoughEnvCfg, UnitreeG1_23DOFRoughEnvCfg_PLAY

__all__ = [
    "UnitreeG1_23DOFFlatEnvCfg",
    "UnitreeG1_23DOFFlatEnvCfg_PLAY",
    "UnitreeG1_23DOFRoughEnvCfg",
    "UnitreeG1_23DOFRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Unitreeg1-23Dof-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1_23DOFFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Unitreeg1-23Dof-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1_23DOFFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Unitreeg1-23Dof-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1_23DOFRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Unitreeg1-23Dof-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1_23DOFRoughEnvCfg_PLAY",
    },
)
