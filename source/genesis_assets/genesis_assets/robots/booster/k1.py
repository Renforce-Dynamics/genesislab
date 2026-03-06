"""Configuration for Booster K1 humanoid robot.

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_ASSETLIB_DIR as ASSET_DIR

##
# Configuration
##

BOOSTER_K1_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/robots/K1/K1_22dof.urdf",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.57],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={
        "replace_cylinders_with_capsules": False,
    },
)
