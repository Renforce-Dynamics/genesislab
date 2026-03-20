from __future__ import annotations

from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from ..velocity_env_cfg import HumanVelocityEnvCfg


@configclass
class RobotEnvCfg(HumanVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robots["robot"] = G1_FULL_ACT_CFG
        self.actions.joint_pos.actuator_name = "full"

        # G1 humanoid base link mapping.
        if getattr(self.events, "add_base_mass", None) is not None:
            self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        if getattr(self.events, "base_external_force_torque", None) is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        if getattr(self.terminations, "base_contact", None) is not None:
            self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="pelvis")


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if hasattr(self.scene, "terrain") and getattr(self.scene.terrain, "terrain_generator", None) is not None:
            terrain_gen = self.scene.terrain.terrain_generator
            if hasattr(terrain_gen, "num_rows"):
                terrain_gen.num_rows = 2
            if hasattr(terrain_gen, "num_cols"):
                terrain_gen.num_cols = 10
