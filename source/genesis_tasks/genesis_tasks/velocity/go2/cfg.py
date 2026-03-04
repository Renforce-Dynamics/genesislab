"""Configuration for Go2 velocity tracking task."""

from dataclasses import field

from genesislab.envs.manager_based_rl_env import ManagerBasedGenesisEnvCfg
from genesislab.configs.robot_cfg import RobotCfg
from genesislab.configs.scene_cfg import SceneCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.utils.configclass import configclass
from genesis_tasks.velocity.go2.mdp.actions import Go2ActionTermCfg
from genesis_tasks.velocity.go2.mdp.commands import VelocityCommandCfg


@configclass
class Go2VelocityEnvCfg(ManagerBasedGenesisEnvCfg):
    """Minimal configuration for Go2 velocity tracking task.

    This config implements a simple velocity tracking task where the Go2 robot
    must track a forward velocity command while maintaining stability.
    """

    # Scene / Simulation configuration
    scene: SceneCfg = SceneCfg(
        num_envs=4096,
        dt=0.002,  # 2ms physics timestep
        substeps=1,
        backend="cuda",
        robots={
            "go2": RobotCfg(
                morph_type="MJCF",
                morph_path="assetslib/unitree/unitree_go2/mjcf/go2.xml",
                initial_pose={"pos": [0.0, 0.0, 0.5], "quat": [0.0, 0.0, 0.0, 1.0]},
                fixed_base=False,
                control_dofs=None,  # Control all actuated joints
                pd_gains=None,  # Will be set in env
            )
        },
        terrain={"type": "plane"},
    )

    # Environment timing
    decimation: int = 10  # 50Hz control (0.002 * 10 = 0.02s control dt)
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    # Observations
    observations: dict[str, ObservationGroupCfg] = field(
        default_factory=lambda: {
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func="genesis_tasks.velocity.go2.mdp.observations.joint_pos",
                    ),
                    "joint_vel": ObservationTermCfg(
                        func="genesis_tasks.velocity.go2.mdp.observations.joint_vel",
                    ),
                    "base_lin_vel": ObservationTermCfg(
                        func="genesis_tasks.velocity.go2.mdp.observations.base_lin_vel",
                    ),
                    "base_ang_vel": ObservationTermCfg(
                        func="genesis_tasks.velocity.go2.mdp.observations.base_ang_vel",
                    ),
                    "command": ObservationTermCfg(
                        func="genesis_tasks.velocity.go2.mdp.observations.command",
                    ),
                },
                concatenate=True,
            )
        }
    )

    # Actions - joint position targets
    actions: dict[str, Go2ActionTermCfg] = field(
        default_factory=lambda: {
            "go2": Go2ActionTermCfg(
                entity_name="go2",
            )
        }
    )

    # Rewards
    rewards: dict[str, RewardTermCfg] = field(
        default_factory=lambda: {
            "velocity_tracking": RewardTermCfg(
                func="genesis_tasks.velocity.go2.mdp.rewards.velocity_tracking",
                weight=1.0,
            ),
            "action_penalty": RewardTermCfg(
                func="genesis_tasks.velocity.go2.mdp.rewards.action_penalty",
                weight=-0.01,
            ),
            "upright": RewardTermCfg(
                func="genesis_tasks.velocity.go2.mdp.rewards.upright",
                weight=0.5,
            ),
        }
    )

    # Terminations
    terminations: dict[str, TerminationTermCfg] = field(
        default_factory=lambda: {
            "base_height": TerminationTermCfg(
                func="genesis_tasks.velocity.go2.mdp.terminations.base_height",
                time_out=False,
            ),
            "time_out": TerminationTermCfg(
                func="genesis_tasks.velocity.go2.mdp.terminations.time_out",
                time_out=True,
            ),
        }
    )

    # Commands - velocity command generation
    commands: dict[str, VelocityCommandCfg] | None = field(
        default_factory=lambda: {
            "lin_vel": VelocityCommandCfg(
                resampling_time_range=(5.0, 10.0),  # Resample every 5-10 seconds
                velocity_range=(0.0, 1.5),  # Forward velocity range in m/s
            )
        }
    )

    # PD control gains
    pd_kp: float = 40.0
    """Position gain for PD control."""

    pd_kd: float = 0.5
    """Velocity gain for PD control."""

    # Termination thresholds
    base_height_threshold: float = 0.15
    """Minimum base height before termination (meters)."""
