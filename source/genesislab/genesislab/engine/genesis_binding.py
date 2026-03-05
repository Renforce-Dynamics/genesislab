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

from genesislab.engine.assets.articulation import GenesisArticulation, GenesisArticulationCfg


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
        self._num_envs = cfg.num_envs

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
        # Create scene with simulation options
        sim_options = gs.options.SimOptions(**self.cfg.to_genesis_options())

        self._scene = gs.Scene(sim_options=sim_options)

        # Add terrain if specified
        if self.cfg.terrain is not None:
            self._add_terrain()

        # Add robots
        for entity_name, robot_cfg in self.cfg.robots.items():
            self._add_robot(entity_name, robot_cfg)

        # Add sensors if specified
        for sensor_name, sensor_cfg in self.cfg.sensors.items():
            self._add_sensor(sensor_name, sensor_cfg)

        # Build the scene
        self._scene.build(
            n_envs=self.cfg.num_envs,
            env_spacing=self.cfg.env_spacing,
            n_envs_per_row=self.cfg.n_envs_per_row,
            center_envs_at_origin=self.cfg.center_envs_at_origin,
        )

        # Resolve DOF indices for controlled joints
        self._resolve_dof_indices()

        # Apply default PD gains from robot configs, if specified
        self._apply_robot_pd_gains()

    def _apply_robot_pd_gains(self) -> None:
        """Apply default PD gains specified in the robot configs.

        This applies uniform PD gains per robot using ``RobotCfg.default_pd_kp``
        and ``RobotCfg.default_pd_kd`` if they are set. It mirrors the simple
        task-level behavior where all DOFs share the same gains, but moves the
        configuration into the robot description.
        """
        import torch  # Local import to avoid any circular import issues.

        for entity_name, robot_cfg in self.cfg.robots.items():
            kp = getattr(robot_cfg, "default_pd_kp", None)
            kd = getattr(robot_cfg, "default_pd_kd", None)
            if kp is None or kd is None:
                continue

            # Infer number of DOFs from joint state.
            dof_pos, _ = self.get_joint_state(entity_name)
            num_dofs = dof_pos.shape[-1]

            kp_tensor = torch.full((num_dofs,), float(kp), device=self.device)
            kd_tensor = torch.full((num_dofs,), float(kd), device=self.device)

            self.set_pd_gains(entity_name, kp_tensor, kd_tensor)

    def _add_terrain(self) -> None:
        """Add terrain entity to the scene."""
        terrain_cfg = self.cfg.terrain
        if terrain_cfg is None:
            return

        # Support both dict-based and configclass-based terrain configs.
        if isinstance(terrain_cfg, dict):
            terrain_type = terrain_cfg.get("type", "plane")
        else:
            terrain_type = getattr(terrain_cfg, "type", "plane")
        if terrain_type == "plane":
            # Create a simple plane
            plane = gs.morphs.Box(
                size=[100.0, 100.0, 0.1],
                pos=[0.0, 0.0, -0.05],
                fixed=True,
            )
            self._scene.add_entity(plane, name="terrain")
        else:
            # Handle other terrain types (e.g., heightfield, mesh)
            raise NotImplementedError(f"Terrain type '{terrain_type}' not yet implemented")

    def _add_robot(self, entity_name: str, robot_cfg: Any) -> None:
        """Add a robot entity to the scene using the Genesis-native asset layer.

        Args:
            entity_name: Name to assign to the entity.
            robot_cfg: Robot configuration.
        """
        if self._scene is None:
            raise RuntimeError("Scene must be created before adding robots.")

        asset_cfg = GenesisArticulationCfg(
            name=entity_name,
            morph_type=robot_cfg.morph_type,
            morph_path=robot_cfg.morph_path,
            initial_pose=robot_cfg.initial_pose,
            fixed_base=robot_cfg.fixed_base,
            control_dofs=robot_cfg.control_dofs,
            morph_options=robot_cfg.morph_options,
        )
        asset = GenesisArticulation(asset_cfg, device=self.device)
        entity = asset.build_into_scene(self._scene)

        # Store only the underlying entity for now; environments can optionally
        # keep track of the wrapper if they need richer behaviour.
        self._entities[entity_name] = entity

    def _add_sensor(self, sensor_name: str, sensor_cfg: dict[str, Any]) -> None:
        """Add a sensor to the scene.

        Args:
            sensor_name: Name to assign to the sensor.
            sensor_cfg: Sensor configuration.
        """
        # Sensor creation logic would go here
        # This is a placeholder for future sensor support
        pass

    def _resolve_dof_indices(self) -> None:
        """Resolve DOF indices for controlled joints in each robot."""
        for entity_name, robot_cfg in self.cfg.robots.items():
            entity = self._entities[entity_name]
            control_dofs = robot_cfg.control_dofs

            if control_dofs is None:
                # If no specific joints specified, get all actuated joints
                # This would require querying the entity for joint names
                # For now, we'll store None and resolve later when needed
                self._dof_indices[entity_name] = None
            else:
                # Resolve joint names to DOF indices
                dof_indices = []
                for joint_name in control_dofs:
                    joint = entity.get_joint(joint_name)
                    if joint is not None:
                        # Genesis entities expose dof_start for each joint
                        dof_start = joint.dof_start
                        dof_count = joint.dof_count if hasattr(joint, "dof_count") else 1
                        dof_indices.extend(range(dof_start, dof_start + dof_count))
                    else:
                        raise ValueError(f"Joint '{joint_name}' not found in entity '{entity_name}'")

                self._dof_indices[entity_name] = torch.tensor(dof_indices, dtype=torch.long, device=self.device)

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
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

        # Get joint positions
        if dof_indices is not None:
            positions = entity.get_dofs_position(dof_indices)
        else:
            # Get all DOF positions
            positions = entity.get_dofs_position()

        # Get joint velocities
        if dof_indices is not None:
            velocities = entity.get_dofs_velocity(dof_indices)
        else:
            # Get all DOF velocities
            velocities = entity.get_dofs_velocity()

        return positions, velocities

    def get_root_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocity for an entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            Tuple of (position, quaternion, linear_velocity, angular_velocity) tensors.
            Positions and velocities have shape (num_envs, 3).
            Quaternions have shape (num_envs, 4).
        """
        entity = self._entities[entity_name]

        # Get root pose
        position = entity.get_pos()
        quaternion = entity.get_quat()

        # Get root velocity
        linear_velocity = entity.get_vel()
        angular_velocity = entity.get_ang()

        return position, quaternion, linear_velocity, angular_velocity

    def set_joint_targets(self, entity_name: str, targets: torch.Tensor, control_type: str = "position") -> None:
        """Set joint control targets for an entity.

        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

        if control_type == "position":
            if dof_indices is not None:
                entity.control_dofs_position(targets, dof_indices)
            else:
                entity.control_dofs_position(targets)
        elif control_type == "velocity":
            if dof_indices is not None:
                entity.control_dofs_velocity(targets, dof_indices)
            else:
                entity.control_dofs_velocity(targets)
        elif control_type == "torque":
            # Torque control would be implemented here
            raise NotImplementedError("Torque control not yet implemented")
        else:
            raise ValueError(f"Unknown control type: {control_type}")

    def set_pd_gains(self, entity_name: str, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """Set PD gains for an entity's joints.

        Args:
            entity_name: Name of the entity.
            kp: Position gains of shape (num_dofs,) or (num_envs, num_dofs).
            kd: Velocity gains of shape (num_dofs,) or (num_envs, num_dofs).
        """
        entity = self._entities[entity_name]
        dof_indices = self._dof_indices.get(entity_name)

        if dof_indices is not None:
            entity.set_dofs_kp(kp, dof_indices)
            entity.set_dofs_kv(kd, dof_indices)
        else:
            entity.set_dofs_kp(kp)
            entity.set_dofs_kv(kd)
