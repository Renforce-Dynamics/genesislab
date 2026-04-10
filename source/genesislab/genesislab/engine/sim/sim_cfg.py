from typing import Literal
from genesislab.utils.configclass import configclass
import genesis as gs

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

    def to_genesis_options(self) -> dict:
        return self.to_dict()


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

    def to_genesis_options(self) -> dict[str, object]:
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

    lights: list[dict] = None
    """List of light configurations. Each light is a dict with keys:
    - type: 'directional' or 'point'
    - For directional: dir (tuple), color (tuple), intensity (float)
    - For point: pos (tuple), color (tuple), intensity (float)
    If None, uses Genesis default lighting."""

    ambient_light: tuple[float, float, float] = None
    """Ambient light color (r, g, b). If None, uses Genesis default."""

    background_color: tuple[float, float, float] = None
    """Background color (r, g, b). If None, uses Genesis default."""

    # HDRI environment lighting
    env_surface: str = None
    """Path to HDRI file (e.g., 'sky.hdr') for image-based lighting.
    If a relative path is provided, it will be resolved relative to data/assets/hdri/.
    Requires RayTracer renderer. If None, no HDRI environment is used."""

    env_radius: float = 1000.0
    """Radius of the environment sphere for HDRI. Only used when env_surface is set."""

    env_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the environment sphere for HDRI. Only used when env_surface is set."""

    def to_genesis_options(self) -> dict[str, object]:
        """Convert this config to keyword arguments for ``gs.options.VisOptions``."""
        if self.rendered_envs_idx is None and self.lights is None and \
           self.ambient_light is None and self.background_color is None:
            return None

        kwargs = {}
        if self.rendered_envs_idx is not None:
            kwargs["rendered_envs_idx"] = self.rendered_envs_idx
        if self.lights is not None:
            kwargs["lights"] = self.lights
        if self.ambient_light is not None:
            kwargs["ambient_light"] = self.ambient_light
        if self.background_color is not None:
            kwargs["background_color"] = self.background_color

        return kwargs if kwargs else None


@configclass
class RigidOptionsCfg:
    """Configuration for Genesis rigid body simulation options (gs.options.RigidOptions).
    
    This config class stores parameters that map to Genesis' RigidOptions.
    """

    dt: float = None
    """Time step for rigid body simulation. If None, uses scene.dt from SimOptionsCfg."""
    
    constraint_solver: Literal["Newton", "GaussSeidel"] = "Newton"
    """Constraint solver type: 'Newton' or 'GaussSeidel'. Defaults to 'Newton'."""
    
    enable_collision: bool = True
    """Whether to enable collision detection."""
    
    enable_joint_limit: bool = True
    """Whether to enable joint limits."""
    
    max_collision_pairs: int = None
    """Maximum number of collision pairs. If None, uses Genesis default."""

    def to_genesis_options(self, scene_dt: float) -> dict:
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