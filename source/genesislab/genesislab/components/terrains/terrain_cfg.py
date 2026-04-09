"""Unified scene-level terrain configuration for GenesisLab.

This module is the single source of truth for terrain configuration and merges
the previously split semantics from:

- importer-style terrain config (plane / generator / usd)
- Genesis native terrain config (genesisbase + surface)
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg

logger = logging.getLogger(__name__)

TerrainType = Literal["plane", "genesisbase", "generator", "usd"]

from .genesis_sub_terrain_cfg import SubTerrainBaseCfg, FlatSubTerrainCfg


# -----------------------------------------------------------------------------
# Terrain surface configuration
# -----------------------------------------------------------------------------


@configclass
class TerrainSurfaceCfg:
    """Configuration for the terrain surface (gs.surfaces.Default)."""

    diffuse_color: Tuple[float, float, float] = None
    """Optional RGB diffuse color for the ground."""

    def build_surface(self):
        """Build and return a Genesis surface instance."""
        import genesis as gs  # local import to avoid hard dependency at import time

        kwargs: Dict[str, Any] = {}
        if self.diffuse_color is not None:
            kwargs["diffuse_color"] = tuple(self.diffuse_color)
        return gs.surfaces.Default(**kwargs)


# -----------------------------------------------------------------------------
# Genesis native terrain morph config (origin + grid + subterrain types/params)
# -----------------------------------------------------------------------------


@configclass
class GenesisTerrainMorphCfg:
    """Configuration for gs.morphs.Terrain (Genesis native heightfield terrain)."""

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_subterrains: Tuple[int, int] = (1, 1)
    subterrain_size: Tuple[float, float] = (24.0, 24.0)
    horizontal_scale: float = 0.25
    vertical_scale: float = 0.005
    uv_scale: float = 1.0
    randomize: bool = False
    subterrain_types: List[List[str]] = [["flat_terrain"]]
    subterrain_parameters: Dict[str, SubTerrainBaseCfg] = {"flat_terrain": FlatSubTerrainCfg()}

    def get_subterrain_params(self):
        return {
            k: v.to_genesis_dict() for k, v in self.subterrain_parameters.items()
        }
        
    def to_genesis_dict(self):
        return dict(
            pos=tuple(self.pos),
            n_subterrains=tuple(self.n_subterrains),
            subterrain_size=tuple(self.subterrain_size),
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            uv_scale=self.uv_scale,
            randomize=self.randomize,
            subterrain_types=self.subterrain_types,
            subterrain_parameters=self.get_subterrain_params(),
        )


# -----------------------------------------------------------------------------
# Top-level terrain config (scene-level)
# -----------------------------------------------------------------------------


@configclass
class TerrainCfg:
    """Unified scene terrain configuration.

    Supported ``terrain_type`` values:

    - ``"plane"``: flat plane terrain
    - ``"genesisbase"``: Genesis native heightfield terrain
    - ``"generator"``: procedural terrain via TerrainGeneratorCfg
    - ``"usd"``: USD terrain (reserved)
    """

    terrain_type: TerrainType = "plane"
    """Primary terrain mode."""

    terrain_generator: "TerrainGeneratorCfg" = None
    """Generator terrain config used when ``terrain_type == 'generator'``."""

    usd_path: str = None
    """USD path used when ``terrain_type == 'usd'``."""

    usd_decompose_error_threshold: float = float("inf")
    """CoACD convex decomposition threshold for USD terrain meshes.

    Controls whether/how precisely Genesis decomposes the terrain mesh into convex hulls
    for collision detection (via the CoACD library).

    - ``float("inf")`` (default): Skip decomposition entirely — fastest loading, uses
      each mesh's convex hull as-is. Suitable for flat terrain or simple geometry.
    - ``0.0``: Force full decomposition on every mesh — most accurate but very slow.
    - ``0.15``: Decompose only if convex-hull volume error exceeds 15% (Genesis default
      for rigid objects). Good balance for complex furniture/architectural meshes.

    Set to ``float("inf")`` for flat terrain, lower values for terrain with complex
    overhangs or concave features that require accurate collision.
    """

    env_spacing: float = None
    """Grid spacing fallback when per-subterrain origins are not used."""

    use_terrain_origins: bool = True
    """Whether to use terrain sub-grid origins for env placement."""

    max_init_terrain_level: int = None
    """Maximum initial terrain level for curriculum placement."""

    debug_vis: bool = False
    """Whether to enable terrain debug visualization."""

    terrain_details_cfg: GenesisTerrainMorphCfg = None
    """Genesis native terrain morph options for ``terrain_type='genesisbase'``."""

    surface_cfg: TerrainSurfaceCfg = TerrainSurfaceCfg()
    """Surface configuration for the terrain (shared across terrain types)."""

    # ------------------------------------------------------------------
    # Backward compatibility fields
    # ------------------------------------------------------------------

    terrain_generator_cfg: Any = None
    """Deprecated alias for ``terrain_generator``."""

    type: str = None
    """Deprecated alias for ``terrain_type``."""

    def __post_init__(self) -> None:  # noqa: D105
        if self.terrain_generator is None and self.terrain_generator_cfg is not None:
            warnings.warn(
                "TerrainCfg.terrain_generator_cfg is deprecated. Use TerrainCfg.terrain_generator instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.terrain_generator = self.terrain_generator_cfg

        if self.type is not None:
            warnings.warn(
                "TerrainCfg.type is deprecated. Use TerrainCfg.terrain_type instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._resolve_legacy_type()

    def _resolve_legacy_type(self) -> None:
        legacy = self.type

        if legacy == "plane":
            self.terrain_type = "plane"
        elif legacy == "genesisbase":
            self.terrain_type = "genesisbase"
        elif legacy == "rough":
            self.terrain_type = "generator"
            if self.terrain_generator is None:
                from genesislab.components.terrains.config.rough import (
                    ROUGH_TERRAINS_CFG,
                )

                self.terrain_generator = ROUGH_TERRAINS_CFG
            else:
                logger.debug(
                    "TerrainCfg.type='rough' but terrain_generator is already set; keeping user-provided generator."
                )
        elif legacy in ("generator", "usd"):
            self.terrain_type = legacy
        else:
            raise ValueError(
                f"Unknown legacy terrain type '{legacy}'. "
                "Accepted values are 'plane', 'genesisbase', 'rough', 'generator', 'usd'."
            )
