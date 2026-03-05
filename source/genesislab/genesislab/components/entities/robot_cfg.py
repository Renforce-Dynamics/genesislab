"""Robot configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Any, Literal

from genesislab.utils.configclass import configclass


@configclass
class PoseCfg:
    """Initial pose configuration for a robot."""

    pos: list[float] = [0.0, 0.0, 0.0]
    """Initial position (x, y, z)."""

    quat: list[float] = [0.0, 0.0, 0.0, 1.0]
    """Initial orientation quaternion (x, y, z, w)."""


@configclass
class MaterialCfg:
    """Material configuration wrapper.

    This is a lightweight wrapper around material parameters. The actual Genesis
    material object is typically constructed in the engine layer based on these
    settings.
    """

    type: str | None = None
    """Material type identifier or import path (e.g., 'Rigid', 'MPM.Muscle')."""

    params: dict[str, Any] = {}
    """Backend-specific parameters for the material."""


@configclass
class SurfaceCfg:
    """Surface configuration wrapper."""

    type: str | None = None
    """Surface preset/type identifier (e.g., 'Smooth', 'Rough')."""

    params: dict[str, Any] = {}
    """Backend-specific parameters for the surface."""


@configclass
class RobotCfg:
    """Configuration for a robot entity to be added to a Genesis scene.

    This config is deliberately minimal and focused on rigid-body robots loaded
    from URDF/MJCF/USD. More advanced options (materials, surfaces, soft bodies)
    can be added incrementally as needed.
    """

    # Required fields: must be annotated *and* have a class member.
    # We use `dataclasses.MISSING` as the default value to signal "no default".
    morph_type: Literal['URDF', 'MJCF', 'USD'] = MISSING
    """Type of morph to use: 'URDF', 'MJCF', 'USD', etc."""

    morph_path: str = MISSING
    """Path to the robot asset file (URDF, MJCF, USD, etc.)."""

    initial_pose: PoseCfg = PoseCfg()
    """Initial pose of the robot."""

    material: MaterialCfg | None = None
    """Material configuration. If None, uses a default rigid material."""

    surface: SurfaceCfg | None = None
    """Surface configuration for contact. If None, uses a default surface."""

    fixed_base: bool = False
    """Whether the robot base is fixed (non-floating)."""

    # Control configuration
    control_dofs: list[str] = None
    """List of joint names to control. If None, all actuated joints are controlled."""

    # Optional per-joint PD gains: joint name -> (kp, kd).
    pd_gains: dict[str, tuple[float, float]] = None
    """Optional PD gains per joint name as (kp, kd).

    If specified, these gains can be used by the engine binding or higher-level
    controllers to initialize or override default PD gains on the robot's DOFs.
    """

    # Optional global/default PD gains applied to all controlled DOFs.
    default_pd_kp: float | None = None
    """Default proportional gain applied to all controlled DOFs if set.

    When not None, the engine binding will apply this gain uniformly across all
    DOFs for the robot after the scene is built. This is a simple alternative
    to specifying per-joint gains in :attr:`pd_gains`.
    """

    default_pd_kd: float | None = None
    """Default derivative gain applied to all controlled DOFs if set.

    When not None, the engine binding will apply this gain uniformly across all
    DOFs for the robot after the scene is built.
    """

    # Additional morph options
    morph_options: dict[str, Any] = {}
    """Additional options passed to the Genesis morph constructor."""

