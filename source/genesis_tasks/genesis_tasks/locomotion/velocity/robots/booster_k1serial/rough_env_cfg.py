"""Configuration for Booster K1Serial velocity tracking task on rough terrain."""

from dataclasses import MISSING

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from ...base_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from genesis_assets.robots import BOOSTER_K1SERIAL_22DOF_CFG
import genesis_tasks.locomotion.velocity.mdp as mdp


@configclass
class BoosterK1SerialRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Booster K1Serial velocity tracking on rough terrain."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Scene: Set robot and terrain
        if isinstance(self.scene, type(MISSING)) or self.scene is MISSING or self.scene is None:
            self.scene = SceneCfg(
                num_envs=4096,
                env_spacing=(2.5, 2.5),
                dt=0.005,
                substeps=1,
                backend="cuda",
                viewer=False,
                robots={"robot": BOOSTER_K1SERIAL_22DOF_CFG},
                terrain=TerrainCfg(type="rough"),
            )

        # Scale down terrains for small robot
        if hasattr(self.scene.terrain, "terrain_generator") and self.scene.terrain.terrain_generator is not None:
            if hasattr(self.scene.terrain.terrain_generator, "sub_terrains"):
                sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
                if "boxes" in sub_terrains:
                    sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
                if "random_rough" in sub_terrains:
                    sub_terrains["random_rough"].noise_range = (0.01, 0.06)
                    sub_terrains["random_rough"].noise_step = 0.01

        # Actions: Reduce action scale
        self.actions.joint_pos.scale = 0.25

        # Rewards
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Terminations
        self.terminations.base_height.params["asset_cfg"] = SceneEntityCfg("robot")


@configclass
class BoosterK1SerialRoughEnvCfg_PLAY(BoosterK1SerialRoughEnvCfg):
    """Configuration for Booster K1Serial velocity tracking on rough terrain (play mode)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = (2.5, 2.5)
        # Spawn robot randomly in grid
        if hasattr(self.scene.terrain, "max_init_terrain_level"):
            self.scene.terrain.max_init_terrain_level = None
        # Reduce number of terrains
        if hasattr(self.scene.terrain, "terrain_generator") and self.scene.terrain.terrain_generator is not None:
            terrain_gen = self.scene.terrain.terrain_generator
            if hasattr(terrain_gen, "num_rows"):
                terrain_gen.num_rows = 5
            if hasattr(terrain_gen, "num_cols"):
                terrain_gen.num_cols = 5
            if hasattr(terrain_gen, "curriculum"):
                terrain_gen.curriculum = False

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
