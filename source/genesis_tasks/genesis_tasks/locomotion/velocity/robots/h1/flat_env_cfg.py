"""Configuration for Unitreeh1 velocity tracking task on flat terrain."""

from genesislab.components.entities.scene_cfg import TerrainCfg
from genesislab.utils.configclass import configclass

from .rough_env_cfg import UnitreeH1RoughEnvCfg


@configclass
class UnitreeH1FlatEnvCfg(UnitreeH1RoughEnvCfg):
    """Configuration for Unitreeh1 velocity tracking on flat terrain."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = 0.25

        # Change terrain to flat
        self.scene.terrain = TerrainCfg(type="plane")
        # No terrain curriculum
        if self.curriculum is not None:
            self.curriculum.terrain_levels = None


@configclass
class UnitreeH1FlatEnvCfg_PLAY(UnitreeH1FlatEnvCfg):
    """Configuration for Unitreeh1 velocity tracking on flat terrain (play mode)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = (2.5, 2.5)
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
