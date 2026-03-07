"""Base environment for manager-based Genesis RL environments."""

from __future__ import annotations

import random
from typing import Any, ClassVar

import torch

from genesislab.components.entities.scene_cfg import SceneCfg
from genesislab.utils.configclass import configclass

from genesislab.engine.binding import GenesisBinding
from genesislab.envs.common import VecEnvObs, VecEnvStepReturn
from genesislab.envs.entity import Entity
from genesislab.managers.action_manager import ActionManager
from genesislab.managers.command_manager import CommandManager, NullCommandManager
from genesislab.managers.observation_manager import ObservationManager
from genesislab.managers.reward_manager import RewardManager
from genesislab.managers.termination_manager import TerminationManager


class ManagerBasedGenesisEnv:
    """Manager-based RL environment for Genesis.

    This environment orchestrates managers (action, observation, reward, termination,
    command) and provides a clean interface for RL training. It maintains batched
    state buffers and handles decimation between physics and control steps.

    The step pipeline is:
    1. Process actions via ActionManager
    2. Apply actions and step physics (with decimation)
    3. Compute observations via ObservationManager
    4. Compute rewards via RewardManager
    5. Compute terminations via TerminationManager
    6. Reset terminated environments
    """

    # Vectorized env metadata, aligned with IsaacLab/mjlab.
    is_vector_env: ClassVar[bool] = True
    """Whether this environment manages a batch of parallel sub-environments."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "rgb_array"],
    }
    """Environment metadata (render modes, fps, etc.)."""
    cfg: "ManagerBasedGenesisEnvCfg"

    def __init__(self, cfg: "ManagerBasedGenesisEnvCfg", device: str = "cuda"):
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
            device: Device to use for tensors ('cuda' or 'cpu').
        """
        self.cfg = cfg
        self.device = device

        # Set random seed if provided
        if cfg.seed is not None:
            self.seed(cfg.seed)

        # Build engine binding
        self._binding = GenesisBinding(cfg.scene, device=device)
        self._binding.build()

        # Compute step timing
        self.physics_dt: float = cfg.scene.sim_options.dt
        """Physics simulation step size."""

        self.step_dt: float = self.physics_dt * cfg.decimation
        """Environment step size (physics_dt * decimation)."""

        # Episode management
        self._max_episode_length_s: float = cfg.episode_length_s
        self.is_finite_horizon: bool = cfg.is_finite_horizon
        if self._max_episode_length_s is not None:
            self._max_episode_length: int = int(self._max_episode_length_s / self.step_dt)
        else:
            self._max_episode_length = None

        # Initialize buffers
        self.num_envs: int = self._binding.num_envs
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.common_step_counter = 0

        # Load managers
        self._load_managers()

        # Configure Gym-style spaces
        self._configure_spaces()

    def seed(self, seed: int = None) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed. If None, uses a random seed.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.cfg.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        # Note: Genesis may have its own RNG that should be seeded separately

    def _load_managers(self) -> None:
        """Load and initialize all managers."""
        # Managers can accept either configclass instances or dictionaries
        # Pass configclass instances directly to preserve type information
        
        # Command manager (optional) - must be initialized first as observations may depend on it
        if self.cfg.commands is not None:
            self.command_manager = CommandManager(cfg=self.cfg.commands, env=self)
        else:
            self.command_manager = NullCommandManager()

        # Action manager
        self.action_manager = ActionManager(cfg=self.cfg.actions, env=self)

        # Observation manager
        self.observation_manager = ObservationManager(cfg=self.cfg.observations, env=self)

        # Reward manager
        self.reward_manager = RewardManager(
            cfg=self.cfg.rewards,
            env=self,
        )

        # Termination manager
        self.termination_manager = TerminationManager(cfg=self.cfg.terminations, env=self)

        # Report initialized managers (IsaacLab-style summary).
        print("[ManagerBasedGenesisEnv] Command manager: %s", self.command_manager)
        print("[ManagerBasedGenesisEnv] Action manager: %s", self.action_manager)
        print("[ManagerBasedGenesisEnv] Observation manager: %s", self.observation_manager)
        print("[ManagerBasedGenesisEnv] Reward manager: %s", self.reward_manager)
        print("[ManagerBasedGenesisEnv] Termination manager: %s", self.termination_manager)

    def _configure_spaces(self) -> None:
        """Configure Gym-style observation and action spaces."""
        # Action space
        action_dim = self.action_manager.total_action_dim
        self.action_space = torch.zeros((self.num_envs, action_dim), device=self.device)

        # Observation space
        # This would be configured based on observation_manager output shapes
        # For now, we'll compute it dynamically
        self.observation_space = None  # Will be set after first observation

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds, if configured."""
        return self._max_episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps, if configured."""
        return self._max_episode_length

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    @property
    def scene(self) -> Any:
        """The Genesis Scene instance."""
        return self._binding.scene

    @property
    def entities(self) -> dict[str, Entity]:
        """Dictionary of entity wrappers keyed by name.

        Each entity provides a `data` property for accessing state:
        - `env.entities["go2"].data.joint_pos` - joint positions
        - `env.entities["go2"].data.root_pos_w` - root position in world frame
        - etc.
        """
        # Lazy initialization: create Entity wrappers on first access
        if not hasattr(self, "_entity_wrappers"):
            self._entity_wrappers: dict[str, Entity] = {}
            for entity_name, raw_entity in self._binding.entities.items():
                self._entity_wrappers[entity_name] = Entity(self, entity_name, raw_entity)
        return self._entity_wrappers

    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity."""
        return self._binding.get_joint_state(entity_name)

    def get_root_state(
        self, entity_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocities for an entity."""
        return self._binding.get_root_state(entity_name)

    def get_body_positions(self, entity_name: str) -> torch.Tensor:
        """Get positions of all bodies/links for an entity.
        
        Args:
            entity_name: Name of the entity.
            
        Returns:
            Tensor of shape (num_envs, num_bodies, 3) containing the world frame
            positions of all bodies/links.
        """
        return self._binding.get_body_positions(entity_name)

    def set_joint_targets(
        self, entity_name: str, targets: torch.Tensor, control_type: str = "position"
    ) -> None:
        """Set joint targets for an entity."""
        self._binding.set_joint_targets(entity_name, targets, control_type=control_type)

    def reset(
        self,
        seed: int = None,
        env_ids: torch.Tensor = None,
        options: dict[str, Any] = None,
    ) -> tuple[VecEnvObs, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed for this reset.
            env_ids: Environment indices to reset. If None, resets all environments.
            options: Additional reset options.

        Returns:
            Tuple of (observations, info) where observations is a dict of tensors
            and info contains additional information.
        """
        # Re-seed if provided
        if seed is not None:
            self.seed(seed)

        # Determine which environments to reset
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # Reset engine binding
        self._binding.reset(env_ids=env_ids)

        # Reset episode counters
        self.episode_length_buf[env_ids] = 0

        # Reset managers and collect extras
        manager_extras = {}
        manager_extras.update(self.action_manager.reset(env_ids=env_ids))
        manager_extras.update(self.observation_manager.reset(env_ids=env_ids))
        manager_extras.update(self.reward_manager.reset(env_ids=env_ids))
        manager_extras.update(self.termination_manager.reset(env_ids=env_ids))
        manager_extras.update(self.command_manager.reset(env_ids=env_ids))

        # Compute initial observations
        obs_buf = self.observation_manager.compute(update_history=True)

        # For alignment with IsaacLab/mjlab, expose manager extras at top level.
        info = dict(manager_extras)
        return obs_buf, info

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment.

        Args:
            action: Action tensor of shape (num_envs, action_dim).

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info) following
            Gymnasium API conventions.
        """
        # Process actions
        self.action_manager.process_action(action.to(self.device))

        # Physics stepping loop (decimation)
        for _ in range(self.cfg.decimation):
            # Apply actions
            self.action_manager.apply_action()

            # Step physics
            self._binding.step()

            # Update sensors (if any) at physics rate.
            self._update_sensors()

        # Update episode counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Compute terminations
        reset_buf = self.termination_manager.compute()
        reset_terminated = self.termination_manager.terminated
        reset_time_outs = self.termination_manager.time_outs

        # Compute rewards
        reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Reset terminated/timed-out environments and collect any episode metrics.
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        manager_extras = {}
        if len(reset_env_ids) > 0:
            maybe_extras = self._reset_idx(reset_env_ids)
            if isinstance(maybe_extras, dict):
                manager_extras.update(maybe_extras)

        # Update commands
        self.command_manager.compute(dt=self.step_dt)

        # Debug visualization (if enabled)
        if hasattr(self, "command_manager") and self.command_manager is not None:
            self.command_manager.debug_vis(self._binding.scene)

        # Compute observations
        obs_buf = self.observation_manager.compute(update_history=True)

        # Build info dict, including any episode-level metrics produced at reset.
        info: dict[str, Any] = {
            "time_outs": reset_time_outs,
            "terminated": reset_terminated,
            "log": manager_extras
        }

        return obs_buf, reward_buf, reset_terminated, reset_time_outs, info

    def _update_sensors(self) -> None:
        """Update any scene-attached sensors after a physics step."""
        scene = self._binding.scene
        if not hasattr(scene, "sensors"):
            raise AttributeError(
                "Scene does not have 'sensors' attribute. "
                "Sensors dict should be initialized during scene building."
            )
        
        sensors = scene.sensors
        for sensor_name, sensor in sensors.items():
            if not hasattr(sensor, "update"):
                raise AttributeError(
                    f"Sensor '{sensor_name}' does not have 'update' method. "
                    f"All sensors must implement the update() method."
                )
            sensor.update(dt=self.physics_dt)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments.

        Args:
            env_ids: Environment indices to reset.
        """
        # Reset engine binding
        self._binding.reset(env_ids=env_ids)

        # Reset episode counters
        self.episode_length_buf[env_ids] = 0

        # Reset managers and collect extras (episode-level summaries, metrics, etc.).
        manager_extras: dict[str, Any] = {}
        manager_extras.update(self.action_manager.reset(env_ids=env_ids))
        manager_extras.update(self.observation_manager.reset(env_ids=env_ids))
        manager_extras.update(self.reward_manager.reset(env_ids=env_ids))
        manager_extras.update(self.termination_manager.reset(env_ids=env_ids))
        manager_extras.update(self.command_manager.reset(env_ids=env_ids))

        return manager_extras


@configclass
class ManagerBasedGenesisEnvCfg:
    """Configuration for a manager-based Genesis environment.
    
    This base config is agnostic to any particular MDP or RL algorithm. It captures
    the high-level environment settings:
    
    - Scene configuration (robots, terrain, sensors, simulation dt, etc.).
    - Manager configurations (observation, action, reward, termination, commands).
    - Generic episode timing and reward scaling options that higher-level RL
      wrappers may interpret according to their own semantics.
    """

    # Base environment configuration
    decimation: int = 1
    """Number of physics steps per environment step. Environment dt = scene.dt * decimation."""

    scene: SceneCfg = SceneCfg()
    """Scene configuration describing robots, terrain, sensors and simulation options."""

    # Manager configs are kept intentionally untyped here; task configs are expected
    # to populate them with the appropriate term config objects from the managers.
    observations: dict[str, object] = {}
    """Observation groups configuration (typically `ObservationGroupCfg` instances)."""

    actions: dict[str, object] = {}
    """Action term configurations."""

    rewards: dict[str, object] = {}
    """Reward term configurations."""

    terminations: dict[str, object] = {}
    """Termination term configurations."""

    commands: dict[str, object] = None
    """Command term configurations. If None, no command manager is created."""

    # RL-specific configuration
    seed: int = None
    """Random seed for reproducibility."""

    episode_length_s: float = None
    """Episode length in seconds. If None, horizon is infinite unless terminated by terms."""

    is_finite_horizon: bool = False
    """Whether episodes are treated as finite-horizon (timeouts counted as truncations)."""

