"""Scene configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any, Literal

import genesis as gs

from genesislab.utils.configclass import configclass
from genesislab.engine.assets.robot import RobotCfg

from genesislab.engine.sim import \
    ViewerOptionsCfg, VisOptionsCfg, RigidOptionsCfg, SimOptionsCfg
    
from genesislab.components.terrains import TerrainCfg

@configclass
class SceneCfg:
    """Configuration for a Genesis scene used by GenesisLab.

    This describes the physical scene including robots, terrain, sensors and
    basic simulation options. It is intentionally minimal and focused on what
    the scene layer and RL environments require.
    """

    # Parallel environments
    num_envs: int = 32
    """Number of parallel environments to simulate."""

    env_spacing: tuple[float, float] = (2.0, 2.0)
    """Spacing between environments in the visualization grid (x, y)."""

    n_envs_per_row: int = None
    """Number of environments per row in the visualization grid. If None, computed automatically."""

    center_envs_at_origin: bool = True
    """Whether to center the environment grid at the origin."""

    # Backend configuration (set during gs.init(), not in SimOptions)
    backend: str = "cuda"
    """Backend to use: typically 'cuda' or 'cpu'."""

    # Viewer / visualization options
    viewer: bool = False
    """Whether to show the Genesis viewer window for this scene."""
    
    viewer_options: ViewerOptionsCfg = ViewerOptionsCfg()
    """Viewer options configuration. If None, uses default ViewerOptionsCfg()."""
    
    vis_options: VisOptionsCfg = VisOptionsCfg()
    """Visualization options configuration. If None, uses default VisOptionsCfg()."""
    
    rigid_options: RigidOptionsCfg = RigidOptionsCfg()
    """Rigid body simulation options configuration. If None, uses default RigidOptionsCfg()."""

    sim_options: SimOptionsCfg = SimOptionsCfg()
    """Simulation options configuration. If None, uses default SimOptionsCfg()."""

    terrain: TerrainCfg = TerrainCfg()
    """Terrain configuration. If None, no terrain is added."""

    # Optional path for recording a video from a default camera.
    record_video_path: str = None
    """If set, LabScene will attach a camera and start a VideoFile recorder."""

    # Entity configurations
    robots: dict[str, "RobotCfg"] = {}
    """Dictionary of robot configurations keyed by logical entity name."""

    sensors: dict[str, Any] = {}
    """Sensor configurations keyed by sensor name."""

