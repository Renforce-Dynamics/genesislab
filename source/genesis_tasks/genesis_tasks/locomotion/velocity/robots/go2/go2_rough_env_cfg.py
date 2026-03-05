"""Configuration for Go2 velocity tracking task on rough terrain."""

from genesislab.components.entities.robot_cfg import RobotCfg
from genesislab.components.entities.scene_cfg import SceneCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesis_tasks.locomotion.velocity.velocity_env_cfg import (
    VelocityEnvCfg,
    ObservationsCfg,
    ActionsCfg,
    RewardsCfg,
    TerminationsCfg,
    CommandsCfg,
)
from genesis_tasks.locomotion.velocity.mdp import Go2ActionTermCfg, VelocityCommandCfg
from .go2_flat_env_cfg import Go2FlatVelocityEnvCfg


class Go2RoughVelocityEnvCfg(Go2FlatVelocityEnvCfg):
    """Configuration for Go2 velocity tracking task on rough terrain.

    This config extends the flat terrain config to add rough terrain challenges.
    The Go2 robot must track a forward velocity command while maintaining stability
    on rough/uneven terrain.
    """

    # Override scene to use rough terrain
    scene: SceneCfg = SceneCfg(
        num_envs=4096,
        dt=0.002,  # 2ms physics timestep
        substeps=1,
        backend="cuda",
        robots={
            "go2": RobotCfg(
                morph_type="MJCF",
                morph_path="./data/assets/assetslib/unitree/unitree_go2/mjcf/go2.xml",
                initial_pose={"pos": [0.0, 0.0, 0.5], "quat": [0.0, 0.0, 0.0, 1.0]},
                fixed_base=False,
                control_dofs=None,  # Control all actuated joints
                pd_gains=None,  # Will be set in env
            )
        },
        terrain={"type": "rough"},  # Use rough terrain instead of plane
    )
