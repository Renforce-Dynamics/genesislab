"""Configuration for Go2 velocity tracking task on flat terrain."""

from genesislab.components.entities.robot_cfg import RobotCfg
from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.scene_entity_config import SceneEntityCfg
from genesis_tasks.locomotion.velocity.velocity_env_cfg import VelocityEnvCfg
from genesis_tasks.locomotion.velocity.mdp import (
    JointPositionActionCfg,
    UniformVelocityCommandCfg,
    observations as mdp_obs,
    rewards as mdp_rewards,
    terminations as mdp_terminations,
)
from genesislab.utils.configclass import configclass


@configclass
class PolicyGroupCfg(ObservationGroupCfg):
    """Policy observation group for Go2 velocity tracking."""
    
    joint_pos = ObservationTermCfg(
        func=mdp_obs.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )
    joint_vel = ObservationTermCfg(
        func=mdp_obs.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )
    base_lin_vel = ObservationTermCfg(
        func=mdp_obs.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )
    base_ang_vel = ObservationTermCfg(
        func=mdp_obs.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )
    projected_gravity = ObservationTermCfg(
        func=mdp_obs.projected_gravity,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )
    velocity_commands = ObservationTermCfg(
        func=mdp_obs.generated_commands,
        params={"command_name": "base_velocity"},
    )


@configclass
class ObservationsCfg:
    """Observation groups configuration for Go2 velocity tracking."""
    
    policy: PolicyGroupCfg = PolicyGroupCfg()


@configclass
class ActionsCfg:
    """Action terms configuration for Go2 velocity tracking."""
    
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="go2",
        joint_names=[".*"],  # Control all joints
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Reward terms configuration for Go2 velocity tracking."""
    
    velocity_tracking = RewardTermCfg(
        func=mdp_rewards.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("go2")},
    )
    action_penalty = RewardTermCfg(
        func=mdp_rewards.action_rate_l2,
        weight=-0.01,
    )
    upright = RewardTermCfg(
        func=mdp_rewards.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("go2")},
    )


@configclass
class TerminationsCfg:
    """Termination terms configuration for Go2 velocity tracking."""
    
    base_height = TerminationTermCfg(
        func=mdp_terminations.base_height,
        time_out=False,
        params={"threshold": 0.15, "asset_cfg": SceneEntityCfg("go2")},
    )
    time_out = TerminationTermCfg(
        func=mdp_terminations.time_out,
        time_out=True,
    )


@configclass
class CommandsCfg:
    """Command terms configuration for Go2 velocity tracking."""
    
    base_velocity: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="go2",
        resampling_time_range=(5.0, 10.0),  # Resample every 5-10 seconds
        heading_command=False,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),  # Forward velocity range in m/s
            lin_vel_y=(-0.5, 0.5),  # Lateral velocity range
            ang_vel_z=(-1.0, 1.0),  # Angular velocity range
        ),
    )


class Go2FlatVelocityEnvCfg(VelocityEnvCfg):
    """Configuration for Go2 velocity tracking task on flat terrain.

    This config implements a simple velocity tracking task where the Go2 robot
    must track a forward velocity command while maintaining stability on flat terrain.
    """

    def __post_init__(self):
        """Post initialization to ensure scene is properly set."""
        super().__post_init__()
        # Ensure scene is set (workaround for configclass inheritance issue with MISSING)
        from dataclasses import MISSING
        if isinstance(self.scene, type(MISSING)) or self.scene is MISSING:
            self.scene = SceneCfg(
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
                        # Apply uniform PD gains to all DOFs directly from the robot config.
                        default_pd_kp=40.0,
                        default_pd_kd=0.5,
                    )
                },
                terrain=TerrainCfg(type="plane"),
            )
        # Ensure we use the local manager config classes rather than the base ones.
        # This avoids inheritance issues with configclass and ensures that all
        # task-specific parameters (like asset_cfg, command ranges, etc.) are used.
        if not isinstance(self.observations, ObservationsCfg):
            self.observations = ObservationsCfg()
        if not isinstance(self.rewards, RewardsCfg):
            self.rewards = RewardsCfg()
        if not isinstance(self.terminations, TerminationsCfg):
            self.terminations = TerminationsCfg()
        if not isinstance(self.commands, CommandsCfg):
            self.commands = CommandsCfg()

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
                # Apply uniform PD gains to all DOFs directly from the robot config.
                default_pd_kp=40.0,
                default_pd_kd=0.5,
            )
        },
        terrain=TerrainCfg(type="plane"),
    )

    # Environment timing
    decimation: int = 10  # 50Hz control (0.002 * 10 = 0.02s control dt)
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    # Observations - using configclass with direct field definitions
    # Note: We need to use the local ObservationsCfg class, not the base class one
    observations: "ObservationsCfg" = ObservationsCfg()

    # Actions - using configclass with direct field definitions
    actions: ActionsCfg = ActionsCfg()

    # Rewards - using configclass with direct field definitions
    rewards: RewardsCfg = RewardsCfg()

    # Terminations - using configclass with direct field definitions
    terminations: TerminationsCfg = TerminationsCfg()

    # Commands - using configclass with direct field definitions
    commands: CommandsCfg = CommandsCfg()
