"""Configuration for SMPL humanoid robot."""

from __future__ import annotations

from genesislab.components.actuators import ImplicitActuatorCfg
from genesislab.engine.assets.robot import InitialPoseCfg, RobotCfg

# Import asset paths from genesis_assets
from genesis_assets import GENESIS_ASSETS_ASSETS_DIR as ASSET_DIR

##
# Configuration
##

SMPL_HUMANOID_CFG = RobotCfg(
    morph_type="USD",
    morph_path=f"{ASSET_DIR}/smpl/smpl_humanoid.usda",
    initial_pose=InitialPoseCfg(
        pos=[0.0, 0.0, 0.95],
        quat=[1.0, 0.0, 0.0, 0.0]  # wxyz: identity,
    ),
    fixed_base=False,
    control_dofs=None,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,  # Use default from USD
            damping=None,  # Use default from USD
        ),
    },
    morph_options={},
)
