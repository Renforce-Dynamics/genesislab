from __future__ import annotations

from genesis_assets.robots import BOOSTER_K1_CFG
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from ..velocity_env_cfg import HumanVelocityEnvCfg


@configclass
class RobotEnvCfg(HumanVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robots["robot"] = BOOSTER_K1_CFG

        # Source K1 task uses trunk as base body in randomization terms.
        if getattr(self.events, "add_base_mass", None) is not None:
            self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")
        if getattr(self.events, "base_external_force_torque", None) is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")

        # K1-specific reward/body remapping from SoccerLab config.
        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
                r"^(?!left_foot_link$)(?!right_foot_link$).+$"
            ]


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
