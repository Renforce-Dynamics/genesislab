"""Configuration for SMPL humanoid robot."""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_ASSETS_DIR as ASSET_DIR
except ImportError:
    ASSET_DIR = None

##
# Configuration
##

SMPL_HUMANOID_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/smpl/smpl_humanoid.usda" if ASSET_DIR else "",
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
