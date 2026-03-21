"""Actuator management for LabScene.

This module provides the ActuatorManager class for processing and managing
actuator configurations for robots in the scene.
"""

from __future__ import annotations

import torch
from genesislab.components.actuators import ActuatorBase

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .lab_scene import LabScene
    from genesislab.engine.entity import LabEntity


class ActuatorManager:
    """Helper class for managing actuators and PD gains."""

    def __init__(self, scene: "LabScene"):
        """Initialize the actuator manager.

        Args:
            scene: Reference to the LabScene instance.
        """
        self._scene = scene

    def process_actuators_cfg(self) -> None:
        """Process and apply actuator configurations for robots (IsaacLab-style).

        This method processes actuator configurations from RobotCfg.actuators and:
        1. Creates actuator instances for each actuator group
        2. Syncs engine ``kp`` / ``kv`` to each actuator's resolved ``stiffness`` / ``damping``
        3. Actuators may still compute torques explicitly and apply them via ``control_dofs_force()``
        """

        for entity_name, robot_cfg in self._scene.cfg.robots.items():
            actuators_cfg = getattr(robot_cfg, "actuators", None)
            if actuators_cfg is None: continue
            lab_entity = self._scene.entities[entity_name]
            self.process_actuator(actuators_cfg, lab_entity)

    def process_actuator(self, actuators_cfg, lab_entity: "LabEntity"):
        # Get raw entity for direct access
        entity = lab_entity.raw_entity
        # Initialize actuators dictionary in the entity
        lab_entity._actuators = {}

        # Get Robot asset from LabEntity (required)
        robot_asset = lab_entity.robot_asset
        if robot_asset is None:
            raise RuntimeError(
                f"Robot asset for entity '{lab_entity.entity_name}' is None. "
                f"This indicates that the robot was not properly initialized with GenesisArticulationRobot."
            )
        # Get actuated joint names (filtered by Robot asset)
        raw_joint_names = robot_asset.get_actuated_joint_names(normalized=False)
        
        if not raw_joint_names:
            raise RuntimeError(f"Robot '{lab_entity.entity_name}': No actuated joints found. Skipping actuator processing.")

        normalized_joint_names = robot_asset.get_normalized_joint_names()

        # Get joint state to infer number of DOFs directly from entity
        dof_pos = entity.get_dofs_position()
        num_dofs = dof_pos.shape[-1]
        num_envs = dof_pos.shape[0]

        # Build joint name to DOF index mapping (using raw names for indexing)
        joint_name_to_dof_indices = {}
        for raw_joint_name in raw_joint_names:
            joint = entity.get_joint(raw_joint_name)
            if joint is not None and hasattr(joint, "dof_start") and joint.dof_start is not None:
                dof_start = joint.dof_start
                dof_count = getattr(joint, "dof_count", 1) if hasattr(joint, "dof_count") else 1
                joint_name_to_dof_indices[raw_joint_name] = list(range(dof_start, dof_start + dof_count))

        # Process each actuator group
        for actuator_name, actuator_cfg in actuators_cfg.items():
            # Find matching joints using normalized names (for consistent pattern matching)
            try:
                matched_indices, matched_normalized_names = robot_asset.match_joints(
                    actuator_cfg.joint_names_expr
                )
                # Convert normalized names back to raw names for indexing
                matched_raw_names = [raw_joint_names[idx] for idx in matched_indices]
            except ValueError as e:
                raise ValueError(
                    f"Robot '{lab_entity.entity_name}': Actuator '{actuator_name}': {e}\n"
                    f"Available normalized joint names: {normalized_joint_names}"
                )

            if not matched_raw_names:
                raise ValueError(
                    f"Robot '{lab_entity.entity_name}': Actuator '{actuator_name}': "
                    f"No joints matched expression {actuator_cfg.joint_names_expr}. "
                    f"Available normalized joint names: {normalized_joint_names}"
                )

            # Resolve DOF indices for matched joints (using raw names, full DOF space)
            matched_dof_indices_full = []
            for raw_joint_name in matched_raw_names:
                if raw_joint_name in joint_name_to_dof_indices:
                    matched_dof_indices_full.extend(joint_name_to_dof_indices[raw_joint_name])
            num_actuator_joints = len(matched_dof_indices_full)

            # Convert to tensor or slice for efficiency
            if len(matched_raw_names) == len(raw_joint_names):
                joint_ids_tensor = slice(None)
            else:
                joint_ids_tensor = torch.tensor(matched_indices, dtype=torch.long, device=self._scene.device)

            # Get default joint properties from entity (for now, use zeros as defaults)
            # In a full implementation, these would be read from the USD/URDF file
            default_stiffness = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_damping = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_armature = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_friction = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_dynamic_friction = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_viscous_friction = torch.zeros(num_envs, num_actuator_joints, device=self._scene.device)
            default_effort_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._scene.device)
            default_velocity_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._scene.device)

            # Create actuator instance (use normalized names for actuator's internal use)
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=matched_normalized_names,  # Use normalized names for consistency
                joint_ids=joint_ids_tensor,
                num_envs=num_envs,
                device=self._scene.device,
                stiffness=default_stiffness,
                damping=default_damping,
                armature=default_armature,
                friction=default_friction,
                dynamic_friction=default_dynamic_friction,
                viscous_friction=default_viscous_friction,
                effort_limit=default_effort_limit,
                velocity_limit=default_velocity_limit,
            )

            # Store actuator instance in the entity
            lab_entity._actuators[actuator_name] = actuator
            
            self.setup_actuator(actuator, entity, matched_dof_indices_full)
            
    def setup_actuator(self, actuator, entity, matched_dof_indices_full):
        full_dof_tensor = torch.tensor(matched_dof_indices_full, dtype=torch.long, device=self._scene.device)

        # Store joint-space DOF indices (excluding base DOFs) in actuator for later use by action terms.
        # We assume the first 6 DOFs correspond to the floating-base motion; joint DOFs start from index 6.
        base_offset = 6
        joint_space_indices = [idx - base_offset for idx in matched_dof_indices_full if idx >= base_offset]
        actuator._dof_indices = torch.tensor(
            joint_space_indices,
            dtype=torch.long,
            device=self._scene.device,
        )

        # Align engine PD gains with the actuator's resolved stiffness / damping (cfg + asset defaults).
        kp, kv = actuator.stiffness, actuator.damping
        if kp.dim() == 2 and not getattr(entity._solver._options, "batch_dofs_info", False):
            kp = kp[0]
            kv = kv[0]
        entity.set_dofs_kp(kp, full_dof_tensor)
        entity.set_dofs_kv(kv, full_dof_tensor)