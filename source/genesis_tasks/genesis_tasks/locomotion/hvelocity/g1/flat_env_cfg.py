"""Unitree G1 humanoid — flat-ground human velocity task."""

from __future__ import annotations

from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG
from genesislab.components.terrains import TerrainCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from ..velocity_env_cfg import HumanVelocityEnvCfg


@configclass
class RobotEnvCfg(HumanVelocityEnvCfg):
    """G1 on plane terrain: SoccerLab-style human velocity with GenesisLab MDP."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robots["robot"] = G1_FULL_ACT_CFG
        self.actions.joint_pos.actuator_name = "full"

        self.scene.terrain = TerrainCfg(terrain_type="plane")
        if self.curriculum is not None:
            self.curriculum.terrain_levels = None

        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play / eval: full command limits, fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
