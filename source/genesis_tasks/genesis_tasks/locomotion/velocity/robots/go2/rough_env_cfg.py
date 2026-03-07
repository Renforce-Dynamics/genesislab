"""Configuration for Go2 velocity tracking task on rough terrain."""

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.managers import SceneEntityCfg
from genesislab.utils.configclass import configclass

from ...base_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from genesis_assets.robots import UNITREE_GO2_CFG
import genesis_tasks.locomotion.velocity.mdp as mdp


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Unitree Go2 velocity tracking on rough terrain."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Scene: Set robot and terrain
        from dataclasses import MISSING
        if isinstance(self.scene, type(MISSING)) or self.scene is MISSING or self.scene is None:
            self.scene = SceneCfg(
                num_envs=4096,
                env_spacing=(2.5, 2.5),
                dt=0.005,
                substeps=1,
                backend="cuda",
                viewer=False,
                robots={"robot": UNITREE_GO2_CFG},
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

        # Actions: Align with genesis-forge example
        # - scale: 0.25 (same as genesis-forge)
        # - use_default_offset: True (use default joint positions as offset)
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True

        # Rewards: align with IsaacLab's Unitree Go2 rough config where applicable.
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Feet air-time and undesired contacts
        # Note: contact-based rewards are currently implemented as no-ops until contact
        # sensors are available, but we still mirror IsaacLab's configuration.
        if hasattr(self.rewards, "feet_air_time") and self.rewards.feet_air_time is not None:
            # Use the shared contact-forces sensor; body selection is handled inside the reward.
            self.rewards.feet_air_time.params["sensor_cfg"] = "contact_forces"
            self.rewards.feet_air_time.params["command_name"] = "base_velocity"
            self.rewards.feet_air_time.params["threshold"] = 0.5
            self.rewards.feet_air_time.weight = 0.01

        if hasattr(self.rewards, "undesired_contacts"):
            # Disable undesired_contacts for Go2 rough task (IsaacLab sets this to None).
            self.rewards.undesired_contacts = None

        # Terminations: keep base-height termination, but also configure contact-based
        # termination in line with IsaacLab naming (currently a no-op without contacts).
        self.terminations.base_height.params["asset_cfg"] = SceneEntityCfg("robot")
        if hasattr(self.terminations, "base_contact") and self.terminations.base_contact is not None:
            self.terminations.base_contact.params["sensor_cfg"] = "contact_forces"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    """Configuration for Unitree Go2 velocity tracking on rough terrain (play mode)."""

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
