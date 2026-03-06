"""Configuration for SMPL humanoid robot."""

from __future__ import annotations

from genesislab.components.actuators import ImplicitActuatorCfg
from genesislab.components.entities.robot_cfg import PoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_ASSETS_DIR as ASSET_DIR

##
# Configuration
##

SMPL_HUMANOID_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/smpl/smpl_humanoid.usda",
    initial_pose=PoseCfg(
        pos=[0.0, 0.0, 0.95],
        quat=[0.0, 0.0, 0.0, 1.0],
    ),
    fixed_base=False,
    control_dofs=None,
    pd_gains=None,
    default_pd_kp=None,
    default_pd_kd=None,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,  # Use default from USD
            damping=None,  # Use default from USD
        ),
    },
    morph_options={},
)
