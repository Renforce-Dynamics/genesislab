"""Configuration for Booster robots (K1 and T1).

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Try to import robotlib asset path, but allow it to be None if not available
try:
    from robotlib import ROBOTLIB_ASSETLIB_DIR as ASSET_DIR, ROBOTLIB_USD_DIR
except ImportError:
    ASSET_DIR = None
    ROBOTLIB_USD_DIR = None

##
# Configuration
##

BOOSTER_K1_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/robots/K1/K1_22dof.urdf" if ASSET_DIR else "",
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

BOOSTER_T1_CFG = RobotCfg(
    morph_type="URDF",
    morph_path=f"{ASSET_DIR}/robots/T1/T1_23dof.urdf" if ASSET_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.70],
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

# K1 Serial variant (USD format)
BOOSTER_K1SERIAL_22DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ROBOTLIB_USD_DIR}/booster_k1_rev/usd/K1_serial.usd" if ROBOTLIB_USD_DIR else "",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.53],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    morph_options={},
)
