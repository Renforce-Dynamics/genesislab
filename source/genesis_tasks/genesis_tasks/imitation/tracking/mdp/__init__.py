"""MDP utilities for whole-body tracking tasks on GenesisLab.

Re-exports motion-specific terms from this package and shared terms (base_lin_vel,
joint_pos_rel, time_out, action_rate_l2, push_by_setting_velocity, etc.) from
genesis_tasks.locomotion.velocity.mdp so that a single ``import genesis_tasks.imitation.tracking.mdp as mdp``
provides everything needed for tracking env configs.
"""

from genesislab.envs.mdp.actions import JointActionCfg, JointPositionActionCfg

from genesis_tasks.locomotion.velocity.mdp import (
    base_lin_vel,
    base_ang_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from genesis_tasks.locomotion.velocity.mdp.rewards import (
    action_rate_l2,
    joint_pos_limits,
    undesired_contacts,
)
from genesis_tasks.locomotion.velocity.mdp.terminations import time_out
from genesis_tasks.locomotion.velocity.mdp import (
    push_by_setting_velocity,
    randomize_rigid_body_material,
)

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

__all__ = [
    "JointActionCfg",
    "JointPositionActionCfg",
    "base_lin_vel",
    "base_ang_vel",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "action_rate_l2",
    "joint_pos_limits",
    "undesired_contacts",
    "time_out",
    "push_by_setting_velocity",
    "randomize_rigid_body_material",
]
