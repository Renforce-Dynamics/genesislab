"""Scene configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genesislab.utils.configclass import configclass
from .robot_cfg import RobotCfg


@configclass
class TerrainCfg:
    """Configuration for terrain in the scene."""

    type: str = "plane"
    """Terrain type (e.g., 'plane', 'rough')."""


@configclass
class SceneCfg:
    """Configuration for a Genesis scene used by GenesisLab.

    This describes the physical scene including robots, terrain, sensors and
    basic simulation options. It is intentionally minimal and focused on what
    the binding layer and RL environments require.
    """

    # Parallel environments
    num_envs: int = 1
    """Number of parallel environments to simulate."""

    env_spacing: tuple[float, float] = (2.0, 2.0)
    """Spacing between environments in the visualization grid (x, y)."""

    n_envs_per_row: int = None
    """Number of environments per row in the visualization grid. If None, computed automatically."""

    center_envs_at_origin: bool = True
    """Whether to center the environment grid at the origin."""

    # Simulation options (mapped to Genesis SimOptions)
    dt: float = 0.005
    """Physics timestep in seconds."""

    substeps: int = 1
    """Number of physics substeps per timestep."""

    backend: str = "cuda"
    """Backend to use: typically 'cuda' or 'cpu'."""

    requires_grad: bool = False
    """Whether to enable gradient tracking for differentiable simulation."""

    # Additional Genesis options
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector (x, y, z)."""

    # Viewer / visualization options
    viewer: bool = False
    """Whether to show the Genesis viewer window for this scene."""

    # Optional path for recording a video from a default camera.
    record_video_path: str = None
    """If set, GenesisBinding will attach a camera and start a VideoFile recorder."""

    # Entity configurations
    robots: dict[str, "RobotCfg"] = {}
    """Dictionary of robot configurations keyed by logical entity name."""

    terrain: TerrainCfg = None
    """Terrain configuration. If None, no terrain is added."""

    sensors: dict[str, Any] = {}
    """Sensor configurations keyed by sensor name."""

    def to_genesis_options(self) -> dict[str, Any]:
        """Convert this config to keyword arguments for ``gs.options.SimOptions``.
        
        Note: The `backend` field is not included here because it is set during
        `gs.init()`, not in `SimOptions`.
        """
        return {
            "dt": self.dt,
            "substeps": self.substeps,
            "requires_grad": self.requires_grad,
            "gravity": self.gravity,
        }

