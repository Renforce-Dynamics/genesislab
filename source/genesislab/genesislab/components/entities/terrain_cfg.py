"""Scene configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genesislab.utils.configclass import configclass

@configclass
class TerrainCfg:
    """Configuration for terrain in the scene."""

    type: str = "plane"
    """Terrain type (e.g., 'plane', 'rough')."""