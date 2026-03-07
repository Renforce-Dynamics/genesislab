"""Scene configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any, Literal

import genesis as gs

from genesislab.utils.configclass import configclass
from .robot_cfg import RobotCfg
from .terrain_cfg import TerrainCfg

@configclass
class SceneCfg:
    """Configuration for a Genesis scene used by GenesisLab.

    This describes the physical scene including robots, terrain, sensors and
    basic simulation options. It is intentionally minimal and focused on what
    the binding layer and RL environments require.
    """

    @configclass
    class SimOptionsCfg:
        """Configuration for Genesis simulation options (gs.options.SimOptions).
        
        This config class stores parameters that map to Genesis' SimOptions.
        """

        dt: float = 0.005
        """Physics timestep in seconds."""

        substeps: int = 1
        """Number of physics substeps per timestep."""

        requires_grad: bool = False
        """Whether to enable gradient tracking for differentiable simulation."""

        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        """Gravity vector (x, y, z)."""

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


    @configclass
    class ViewerOptionsCfg:
        """Configuration for Genesis viewer options (gs.options.ViewerOptions).
        
        This config class stores parameters that map to Genesis' ViewerOptions.
        """

        max_FPS: int = None
        """Maximum FPS for the viewer. If None, uses Genesis default."""
        
        camera_pos: tuple[float, float, float] = None
        """Initial camera position (x, y, z). If None, uses Genesis default."""
        
        camera_lookat: tuple[float, float, float] = None
        """Initial camera look-at target (x, y, z). If None, uses Genesis default."""
        
        camera_fov: float = None
        """Camera field of view in degrees. If None, uses Genesis default."""

        def to_genesis_options(self) -> dict[str, Any]:
            """Convert this config to keyword arguments for ``gs.options.ViewerOptions``."""
            kwargs = {}
            if self.max_FPS is not None:
                kwargs["max_FPS"] = self.max_FPS
            if self.camera_pos is not None:
                kwargs["camera_pos"] = self.camera_pos
            if self.camera_lookat is not None:
                kwargs["camera_lookat"] = self.camera_lookat
            if self.camera_fov is not None:
                kwargs["camera_fov"] = self.camera_fov
            return kwargs


    @configclass
    class VisOptionsCfg:
        """Configuration for Genesis visualization options (gs.options.VisOptions).
        
        This config class stores parameters that map to Genesis' VisOptions.
        """

        rendered_envs_idx: list[int] = None
        """List of environment indices to render. If None, all environments are rendered."""

        def to_genesis_options(self) -> dict[str, Any] | None:
            """Convert this config to keyword arguments for ``gs.options.VisOptions``.
            
            Returns None if rendered_envs_idx is None (no vis options needed).
            """
            if self.rendered_envs_idx is None:
                return None
            return {"rendered_envs_idx": self.rendered_envs_idx}


    @configclass
    class RigidOptionsCfg:
        """Configuration for Genesis rigid body simulation options (gs.options.RigidOptions).
        
        This config class stores parameters that map to Genesis' RigidOptions.
        """

        dt: float = None
        """Time step for rigid body simulation. If None, uses scene.dt from SimOptionsCfg."""
        
        constraint_solver: Literal = ["Newton", "GaussSeidel"]
        """Constraint solver type: 'Newton' or 'GaussSeidel'. Defaults to 'Newton'."""
        
        enable_collision: bool = True
        """Whether to enable collision detection."""
        
        enable_joint_limit: bool = True
        """Whether to enable joint limits."""
        
        max_collision_pairs: int = None
        """Maximum number of collision pairs. If None, uses Genesis default."""

        def to_genesis_options(self, scene_dt: float) -> dict[str, Any]:
            """Convert this config to keyword arguments for ``gs.options.RigidOptions``.
            
            Args:
                scene_dt: The scene's dt from SimOptionsCfg, used as fallback if self.dt is None.
            """

            kwargs = {
                "dt": self.dt if self.dt is not None else scene_dt,
                "enable_collision": self.enable_collision,
                "enable_joint_limit": self.enable_joint_limit,
            }
            
            # Map constraint solver string to Genesis enum
            if self.constraint_solver == "Newton":
                kwargs["constraint_solver"] = gs.constraint_solver.Newton
            elif self.constraint_solver == "GaussSeidel":
                kwargs["constraint_solver"] = gs.constraint_solver.GaussSeidel
            else:
                raise ValueError(
                    f"Unknown constraint_solver '{self.constraint_solver}'. "
                    f"Expected 'Newton' or 'GaussSeidel'."
                )
            
            if self.max_collision_pairs is not None:
                kwargs["max_collision_pairs"] = self.max_collision_pairs
            
            return kwargs

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

    # Genesis options - organized into sub-configs
    sim: SimOptionsCfg = SimOptionsCfg()
    """Simulation options configuration. If None, uses default SimOptionsCfg()."""
    
    # Viewer / visualization options
    viewer: bool = False
    """Whether to show the Genesis viewer window for this scene."""
    
    viewer_options: ViewerOptionsCfg = ViewerOptionsCfg()
    """Viewer options configuration. If None, uses default ViewerOptionsCfg()."""
    
    vis_options: VisOptionsCfg = VisOptionsCfg()
    """Visualization options configuration. If None, uses default VisOptionsCfg()."""
    
    rigid_options: RigidOptionsCfg = RigidOptionsCfg()
    """Rigid body simulation options configuration. If None, uses default RigidOptionsCfg()."""

    terrain: TerrainCfg = TerrainCfg()
    """Terrain configuration. If None, no terrain is added."""

    # Optional path for recording a video from a default camera.
    record_video_path: str = None
    """If set, GenesisBinding will attach a camera and start a VideoFile recorder."""

    # Entity configurations
    robots: dict[str, "RobotCfg"] = {}
    """Dictionary of robot configurations keyed by logical entity name."""

    sensors: dict[str, Any] = {}
    """Sensor configurations keyed by sensor name."""

