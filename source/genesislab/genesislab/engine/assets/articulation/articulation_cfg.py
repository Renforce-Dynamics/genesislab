from dataclasses import MISSING
from typing import Any, Literal, Sequence

from genesislab.utils.configclass import configclass

from genesislab.engine.assets.lab_asset_base import LabAssetBase

@configclass
class InitialPoseCfg:
    """Initial pose configuration for a robot."""

    pos: list[float] = [0.0, 0.0, 0.0]
    """Initial position (x, y, z)."""

    quat: list[float] = [0.0, 0.0, 0.0, 1.0]
    """Initial orientation quaternion (x, y, z, w)."""

@configclass
class ArticulationCfg:
    """Configuration for a Genesis articulation asset.

    This is intentionally lightweight and aligned with :class:`RobotCfg` so
    that environments can easily construct either scene-level robots or
    explicit asset wrappers.
    """

    name: str = MISSING
    """Logical name of the articulation asset."""

    morph_type: Literal["URDF", "MJCF", "USD"] = MISSING
    """Type of Genesis morph to construct."""

    morph_path: str = ""
    """File path to the robot description (URDF/MJCF/USD)."""

    initial_pose: InitialPoseCfg = InitialPoseCfg()
    """Initial pose of the articulation root."""

    fixed_base: bool = False
    """Whether the base of the articulation is fixed."""

    control_dofs: list[str] = None
    """List of joint names to control. If None, all actuated joints are controlled."""

    morph_options: dict = {}
    """Additional keyword arguments forwarded to the Genesis morph constructor."""
