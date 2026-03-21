"""Base RL environment configuration for human velocity (hvelocity) tasks."""

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.utils.configclass import configclass

from .components import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    HumanVelocitySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class HumanVelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Base config: scene, observations, actions, commands, rewards, terminations, events, curriculum."""

    scene: HumanVelocitySceneCfg = HumanVelocitySceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
            tg = getattr(self.scene.terrain, "terrain_generator", None)
            if tg is not None and hasattr(tg, "curriculum"):
                tg.curriculum = True
