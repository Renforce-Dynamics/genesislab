"""Configuration for Booster K1 Serial humanoid robot (USD variant).

Reference: Booster humanoid robots from BeyondMimic.
"""

from __future__ import annotations

from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_USD_DIR

##
# Configuration
##

BOOSTER_K1SERIAL_22DOF_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{GENESIS_ASSETS_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
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
