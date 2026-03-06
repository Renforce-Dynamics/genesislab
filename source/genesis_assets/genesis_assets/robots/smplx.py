"""Configuration for SMPLX humanoid robot."""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_USD_DIR
except ImportError:
    ROBOTLIB_USD_DIR = None

##
# Configuration
##

SMPLX_HUMANOID_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ROBOTLIB_USD_DIR}/smplx/smplx_humanoid.usda" if ROBOTLIB_USD_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.95],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
