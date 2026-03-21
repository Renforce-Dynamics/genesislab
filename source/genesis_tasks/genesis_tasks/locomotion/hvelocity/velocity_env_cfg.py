"""Human velocity task entry configs (shared across humanoid variants)."""

from genesislab.utils.configclass import configclass

from .base_hvelocity_env_cfg import BaseHumanVelocityEnvCfg


@configclass
class HumanVelocityEnvCfg(BaseHumanVelocityEnvCfg):
    """Defaults for human velocity tracking live in :mod:`genesis_tasks.locomotion.hvelocity.components`."""


@configclass
class HumanVelocityPlayEnvCfg(HumanVelocityEnvCfg):
    """Smaller rollout; full command envelope; optional obs corruption off."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
