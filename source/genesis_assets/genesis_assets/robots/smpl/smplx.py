"""Configuration for SMPLX humanoid robot."""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_USD_DIR

##
# Configuration
##

SMPLX_HUMANOID_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{GENESIS_ASSETS_USD_DIR}/smplx/smplx_humanoid.usda",
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
