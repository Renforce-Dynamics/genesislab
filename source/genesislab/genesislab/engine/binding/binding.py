"""Genesis engine binding layer for GenesisLab.

This module provides a thin binding layer that exposes a stable, RL-centric
interface over a Genesis Scene and entities. It encapsulates engine-specific
details and provides batched state access and control methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs
import torch

if TYPE_CHECKING:
    from genesislab.components.entities.scene_cfg import SceneCfg

from genesislab.engine.binding.actuators import ActuatorManager
from genesislab.engine.binding.control import Controller
from genesislab.engine.binding.dof_resolver import DOFResolver
from genesislab.engine.binding.scene import SceneBuilder
from genesislab.engine.binding.state import StateQuerier


class GenesisBinding:
    """Binding layer for Genesis Scene and entities.

    This class provides a clean interface for RL environments to interact with
    Genesis without directly accessing solver internals. It manages:
    - Scene construction and building
    - Entity loading and indexing
    - State queries (joint states, root poses, velocities)
    - Control target setting
    - Reset operations
    - DOF indexing

    All operations are batched over num_envs environments.
    """

    def __init__(self, cfg: SceneCfg, device: str = "cuda"):
        """Initialize the Genesis binding.

        Args:
            cfg: Scene configuration.
            device: Device to use for tensors ('cuda' or 'cpu').
        """
        self.cfg = cfg
        self.device = device
        self._scene: gs.Scene = None
        self._entities: dict[str, Any] = {}
        self._dof_indices: dict[str, torch.Tensor] = {}
        self._actuators: dict[str, dict[str, Any]] = {}  # entity_name -> {actuator_name -> actuator_instance}
        self._num_envs = cfg.num_envs

        # Initialize helper components
        self._scene_builder         = SceneBuilder(self)
        self._dof_resolver          = DOFResolver(self)
        self._actuator_manager      = ActuatorManager(self)
        self._state_querier         = StateQuerier(self)
        self._controller            = Controller(self)

    @property
    def scene(self) -> gs.Scene:
        """The Genesis Scene instance."""
        if self._scene is None:
            raise RuntimeError("Scene not built. Call build() first.")
        return self._scene

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def entities(self) -> dict[str, Any]:
        """Dictionary of entity objects keyed by name."""
        return self._entities

    def build(self) -> None:
        """Build the Genesis scene and entities.

        This method:
        1. Creates a Genesis Scene with appropriate options
        2. Adds robots and terrain according to cfg
        3. Builds the scene with num_envs
        4. Resolves DOF indices for controlled joints
        """
        # Create scene
        self._scene = self._scene_builder.create_scene()

        # Initialize sensors dict (even if empty) so that MDP code can safely access env.scene.sensors
        if not hasattr(self._scene, "sensors"):
            self._scene.sensors = {}

        # Add terrain if specified
        if self.cfg.terrain is not None:
            self._scene_builder.add_terrain(self._scene)

        # Add robots
        for entity_name, robot_cfg in self.cfg.robots.items():
            entity = self._scene_builder.add_robot(self._scene, entity_name, robot_cfg)
            self._entities[entity_name] = entity

        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._scene_builder.add_sensor(self._scene, sensor_name, sensor_cfg)

        # Optional: attach a simple camera and start video recording before
        # building the scene, if requested via the scene config.
        video_path = getattr(self.cfg, "record_video_path", None)
        if video_path is not None:
            from genesislab.engine.visualize import attach_video_recorder
            attach_video_recorder(self._scene, str(video_path))

        # Build the scene
        self._scene_builder.build_scene(self._scene)

        # Resolve DOF indices for controlled joints
        self._dof_resolver.resolve_dof_indices()

        # Process actuator configurations (IsaacLab-style)
        # This takes precedence over legacy PD gains
        self._actuator_manager.process_actuators_cfg()

        # Apply default PD gains from robot configs, if specified
        # (only if actuators are not configured)
        self._actuator_manager.apply_robot_pd_gains()

    def reset(self, env_ids: torch.Tensor = None) -> None:
        """Reset specified environments to initial state.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self.scene.reset()
        else:
            # Convert to list if tensor
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
            self.scene.reset(envs_idx=env_ids)

    def step(self) -> None:
        """Step the physics simulation by one timestep."""
        self.scene.step()

    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (positions, velocities) tensors of shape (num_envs, num_dofs).
        """
        return self._state_querier.get_joint_state(entity_name)

    def get_root_state(
        self, entity_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocity for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (position, quaternion, linear_velocity, angular_velocity) tensors.
            Positions and velocities have shape (num_envs, 3).
            Quaternions have shape (num_envs, 4).
        """
        return self._state_querier.get_root_state(entity_name)

    def get_body_positions(self, entity_name: str) -> torch.Tensor:
        """Get positions of all bodies/links for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tensor of shape ``(num_envs, num_links, 3)`` containing the world-frame
            positions of all links/bodies.
        """
        return self._state_querier.get_body_positions(entity_name)

    def set_joint_targets(
        self, entity_name: str, targets: torch.Tensor, control_type: str = "position"
    ) -> None:
        """Set joint control targets for an entity.

        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        self._controller.set_joint_targets(entity_name, targets, control_type)

    def set_pd_gains(self, entity_name: str, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """Set PD gains for an entity's joints.

        Args:
            entity_name: Name of the entity.
            kp: Position gains of shape (num_dofs,) or (num_envs, num_dofs).
            kd: Velocity gains of shape (num_dofs,) or (num_envs, num_dofs).
        """
        self._controller.set_pd_gains(entity_name, kp, kd)
