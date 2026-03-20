"""Base human-velocity task configuration.

This module ports the SoccerLab human velocity task settings to GenesisLab's
manager-based locomotion framework while reusing the local velocity MDP stack.
"""

from __future__ import annotations

from genesis_tasks.locomotion.velocity.base_velocity_env_cfg import BaseVelocityEnvCfg
from genesislab.utils.configclass import configclass


@configclass
class HumanVelocityEnvCfg(BaseVelocityEnvCfg):
    """Shared config for human velocity tasks."""

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.episode_length_s = 20.0

        # SoccerLab-style command behavior.
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
        self.commands.base_velocity.ranges.heading = None

        # Action scale from source task.
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True

        # Source task reward scaling mapped to local reward terms.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -0.001
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.alive.weight = 0.15

        if hasattr(self.rewards, "dof_pos_limits"):
            self.rewards.dof_pos_limits.weight = -5.0

        # Match source task's reset/randomization cadence.
        if getattr(self, "events", None) is not None:
            if getattr(self.events, "add_base_mass", None) is not None:
                self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
            if getattr(self.events, "push_robot", None) is not None:
                self.events.push_robot.interval_range_s = (5.0, 5.0)
                self.events.push_robot.params["velocity_range"] = {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }


@configclass
class HumanVelocityPlayEnvCfg(HumanVelocityEnvCfg):
    """Play config for human velocity tasks."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if hasattr(self.scene, "terrain") and getattr(self.scene.terrain, "terrain_generator", None) is not None:
            terrain_gen = self.scene.terrain.terrain_generator
            if hasattr(terrain_gen, "num_rows"):
                terrain_gen.num_rows = 2
            if hasattr(terrain_gen, "num_cols"):
                terrain_gen.num_cols = 10
