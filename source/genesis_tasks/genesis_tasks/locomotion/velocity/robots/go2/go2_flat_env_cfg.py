"""Configuration for Go2 velocity tracking task on flat terrain."""

from dataclasses import MISSING

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


class Go2FlatVelocityEnvCfg(VelocityEnvCfg):
    """Configuration for Go2 velocity tracking task on flat terrain.

    This config implements a simple velocity tracking task where the Go2 robot
    must track a forward velocity command while maintaining stability on flat terrain.
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
                morph_path="./data/assets/assetslib/unitree/unitree_go2/mjcf/go2.xml",
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

    # Observations - using configclass
    observations: ObservationsCfg = ObservationsCfg(
        policy=ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(
                    func="genesis_tasks.locomotion.velocity.mdp.observations.joint_pos",
                ),
                "joint_vel": ObservationTermCfg(
                    func="genesis_tasks.locomotion.velocity.mdp.observations.joint_vel",
                ),
                "base_lin_vel": ObservationTermCfg(
                    func="genesis_tasks.locomotion.velocity.mdp.observations.base_lin_vel",
                ),
                "base_ang_vel": ObservationTermCfg(
                    func="genesis_tasks.locomotion.velocity.mdp.observations.base_ang_vel",
                ),
                "command": ObservationTermCfg(
                    func="genesis_tasks.locomotion.velocity.mdp.observations.command",
                ),
            },
            concatenate_terms=True,
        )
    )

    # Actions - using configclass
    actions: ActionsCfg = ActionsCfg(
        go2=Go2ActionTermCfg(
            entity_name="go2",
        )
    )

    # Rewards - using configclass
    rewards: RewardsCfg = RewardsCfg(
        velocity_tracking=RewardTermCfg(
            func="genesis_tasks.locomotion.velocity.mdp.rewards.velocity_tracking",
            weight=1.0,
        ),
        action_penalty=RewardTermCfg(
            func="genesis_tasks.locomotion.velocity.mdp.rewards.action_penalty",
            weight=-0.01,
        ),
        upright=RewardTermCfg(
            func="genesis_tasks.locomotion.velocity.mdp.rewards.upright",
            weight=0.5,
        ),
    )

    # Terminations - using configclass
    terminations: TerminationsCfg = TerminationsCfg(
        base_height=TerminationTermCfg(
            func="genesis_tasks.locomotion.velocity.mdp.terminations.base_height",
            time_out=False,
        ),
        time_out=TerminationTermCfg(
            func="genesis_tasks.locomotion.velocity.mdp.terminations.time_out",
            time_out=True,
        ),
    )

    # Commands - using configclass
    commands: CommandsCfg = CommandsCfg(
        lin_vel=VelocityCommandCfg(
            resampling_time_range=(5.0, 10.0),  # Resample every 5-10 seconds
            velocity_range=(0.0, 1.5),  # Forward velocity range in m/s
        )
    )
