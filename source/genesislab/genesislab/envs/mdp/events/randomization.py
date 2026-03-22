"""Domain-randomization style event stubs for Genesis backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def randomize_rigid_body_material(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | Dict[str, Tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str = None,
) -> None:
    """Placeholder for material randomization (no-op on Genesis backend)."""
    _ = (env, env_ids, scale_range, asset_cfg, relative_child_path)
    return


def randomize_rigid_body_mass(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str = "add",
) -> None:
    """Placeholder for mass randomization (no-op on Genesis backend)."""
    _ = (env, env_ids, asset_cfg, mass_distribution_params, operation)
    return


def randomize_rigid_body_com(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Placeholder for center-of-mass randomization (no-op on Genesis backend)."""
    _ = (env, env_ids, com_range, asset_cfg)
    return


__all__ = [
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_com",
]

