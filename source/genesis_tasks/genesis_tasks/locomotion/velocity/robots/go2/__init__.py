"""Go2 velocity tracking task registration for GenesisLab."""

from __future__ import annotations

import gymnasium as gym

from .go2_flat_env_cfg import Go2FlatVelocityEnvCfg
from .go2_rough_env_cfg import Go2RoughVelocityEnvCfg

from . import agents  # type: ignore

__all__ = [
    "Go2FlatVelocityEnvCfg",
    "Go2RoughVelocityEnvCfg",
]


def _get_rsl_rl_cfg_entry_point(suffix: str) -> str | None:
    """Return the rsl_rl runner config entry point for a given suffix, if available.

    This mirrors the IsaacLab convention of defining task-specific RL configs
    inside an ``agents`` module.
    """
    if agents is None:
        return None
    return f"{agents.__name__}.rsl_rl_ppo_cfg:{suffix}"


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_env_cfg:Go2FlatVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": agents.Go2FlatPPORunnerCfg(),
    },
)

gym.register(
    id="Genesis-Velocity-Flat-Go2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_env_cfg:Go2FlatVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": _get_rsl_rl_cfg_entry_point("Go2FlatPPORunnerCfg"),
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_rough_env_cfg:Go2RoughVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": _get_rsl_rl_cfg_entry_point("Go2RoughPPORunnerCfg"),
    },
)

gym.register(
    id="Genesis-Velocity-Rough-Go2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_rough_env_cfg:Go2RoughVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": _get_rsl_rl_cfg_entry_point("Go2RoughPPORunnerCfg"),
    },
)
 