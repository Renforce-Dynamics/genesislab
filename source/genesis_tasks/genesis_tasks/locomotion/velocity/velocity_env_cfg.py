"""Base configuration for velocity tracking locomotion tasks."""

from dataclasses import MISSING, field
from typing import Dict

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers.observation_manager import ObservationGroupCfg
from genesislab.managers.action_manager import ActionTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.command_manager import CommandTermCfg
from genesislab.utils.configclass import configclass


@configclass
class ObservationsCfg:
    """Configuration for observation groups.
    
    This configclass groups all observation configurations together.
    The manager will extract the dict from this configclass.
    """
    
    def to_dict(self) -> Dict[str, ObservationGroupCfg]:
        """Convert to dictionary format expected by managers."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not MISSING:
                result[key] = value
        return result


@configclass
class ActionsCfg:
    """Configuration for action terms.
    
    This configclass groups all action configurations together.
    The manager will extract the dict from this configclass.
    """
    
    def to_dict(self) -> Dict[str, ActionTermCfg]:
        """Convert to dictionary format expected by managers."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not MISSING:
                result[key] = value
        return result


@configclass
class RewardsCfg:
    """Configuration for reward terms.
    
    This configclass groups all reward configurations together.
    The manager will extract the dict from this configclass.
    """
    
    def to_dict(self) -> Dict[str, RewardTermCfg]:
        """Convert to dictionary format expected by managers."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not MISSING:
                result[key] = value
        return result


@configclass
class TerminationsCfg:
    """Configuration for termination terms.
    
    This configclass groups all termination configurations together.
    The manager will extract the dict from this configclass.
    """
    
    def to_dict(self) -> Dict[str, TerminationTermCfg]:
        """Convert to dictionary format expected by managers."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not MISSING:
                result[key] = value
        return result


@configclass
class CommandsCfg:
    """Configuration for command terms.
    
    This configclass groups all command configurations together.
    The manager will extract the dict from this configclass.
    """
    
    def to_dict(self) -> Dict[str, CommandTermCfg]:
        """Convert to dictionary format expected by managers."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not MISSING:
                result[key] = value
        return result


@configclass
class VelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Base configuration for velocity tracking locomotion tasks.
    
    This base class uses configclass for all manager configurations instead of dict.
    Subclasses should override the configclass fields to customize the task.
    """
    
    # Override manager configs to use configclass instead of dict
    observations: ObservationsCfg = ObservationsCfg()
    """Observation groups configuration as a configclass."""
    
    actions: ActionsCfg = ActionsCfg()
    """Action terms configuration as a configclass."""
    
    rewards: RewardsCfg = RewardsCfg()
    """Reward terms configuration as a configclass."""
    
    terminations: TerminationsCfg = TerminationsCfg()
    """Termination terms configuration as a configclass."""
    
    commands: CommandsCfg = None
    """Command terms configuration as a configclass. If None, no command manager is created."""
    
    # Common velocity task parameters
    pd_kp: float = 40.0
    """Position gain for PD control."""
    
    pd_kd: float = 0.5
    """Velocity gain for PD control."""
    
    base_height_threshold: float = 0.15
    """Minimum base height before termination (meters)."""
