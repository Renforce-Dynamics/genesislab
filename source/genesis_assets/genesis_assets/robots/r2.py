"""Configuration for R2 humanoid robot.

Reference: R2 wholebody humanoid robot.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_ASSETLIB_DIR
except ImportError:
    ROBOTLIB_ASSETLIB_DIR = None

##
# Configuration
##

R2_WHOLEBODY_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ROBOTLIB_ASSETLIB_DIR}/thrid_party/r2_wholebody/usd/r2_wb.usd" if ROBOTLIB_ASSETLIB_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.0],  # Will be set based on actual robot configuration
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
