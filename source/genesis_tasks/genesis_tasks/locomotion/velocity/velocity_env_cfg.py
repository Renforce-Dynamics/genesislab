"""Base configuration for velocity tracking locomotion tasks."""

from dataclasses import MISSING

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers.observation_manager import ObservationGroupCfg
from genesislab.managers.action_manager import ActionTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.command_manager import CommandTermCfg
from genesislab.utils.configclass import configclass


@configclass
class VelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Base configuration for velocity tracking locomotion tasks.
    
    This base class uses configclass for all manager configurations instead of dict.
    Subclasses should override the configclass fields to customize the task.
    
    To define observations, actions, rewards, etc., subclass the respective config
    classes and add term fields directly:
    
    .. code-block:: python
    
        @configclass
        class MyObservationsCfg:
            policy: ObservationGroupCfg = PolicyGroupCfg()
        
        @configclass
        class PolicyGroupCfg(ObservationGroupCfg):
            joint_pos = ObservationTermCfg(func=mdp.joint_pos)
            joint_vel = ObservationTermCfg(func=mdp.joint_vel)
    """
    
    # Common velocity task parameters
    pd_kp: float = 40.0
    """Position gain for PD control."""
    
    pd_kd: float = 0.5
    """Velocity gain for PD control."""
    
    base_height_threshold: float = 0.15
    """Minimum base height before termination (meters)."""
