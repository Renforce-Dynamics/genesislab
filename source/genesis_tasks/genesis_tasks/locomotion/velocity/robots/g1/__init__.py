"""G1 Beyondmimic velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .flat_env_cfg import G1BeyondMimicFlatEnvCfg, G1BeyondMimicFlatEnvCfg_PLAY
from .rough_env_cfg import G1BeyondMimicRoughEnvCfg, G1BeyondMimicRoughEnvCfg_PLAY

__all__ = [
    "G1BeyondMimicFlatEnvCfg",
    "G1BeyondMimicFlatEnvCfg_PLAY",
    "G1BeyondMimicRoughEnvCfg",
    "G1BeyondMimicRoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-G1Beyondmimic-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1BeyondMimicFlatEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Flat-G1Beyondmimic-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1BeyondMimicFlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-G1Beyondmimic-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1BeyondMimicRoughEnvCfg",
    },
)

gym.register(
    id="Genesis-Velocity-Rough-G1Beyondmimic-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1BeyondMimicRoughEnvCfg_PLAY",
    },
)
