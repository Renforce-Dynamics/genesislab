"""RSL-RL integration utilities for GenesisLab.

This package mirrors IsaacLab's ``isaaclab_rl.rsl_rl`` helpers and exposes:

- ``GenesisRslRlVecEnv``: adapter from :class:`ManagerBasedRlEnv` to
  :class:`rsl_rl.env.VecEnv` used by :class:`rsl_rl.runners.OnPolicyRunner`.
"""

from .env_wrappers import GenesisRslRlVecEnv, ObsGroupMapping

__all__ = ["GenesisRslRlVecEnv", "ObsGroupMapping"]

