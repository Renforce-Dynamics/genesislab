"""Per-robot actuator management.

Each :class:`~genesislab.engine.entity.lab_entity.LabEntity` owns an
:class:`ActuatorManager` instance. Actuators are created and wired to the
simulation after the Genesis scene is built (``scene.build()``), when DOF
queries are valid.
"""

from __future__ import annotations

import torch
from genesislab.components.actuators import ActuatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesislab.engine.assets.robot import Robot, RobotCfg
    from genesislab.engine.entity import LabEntity


class ActuatorManager:
    """Builds and holds actuators for a single robot entity."""

    def __init__(self, lab_entity: "LabEntity", device: str):
        self._lab_entity = lab_entity
        self._device = device

    @property
    def lab_entity(self) -> "LabEntity":
        return self._lab_entity

    def setup(self, robot_cfg: "RobotCfg") -> None:
        """Create actuators from ``robot_cfg.actuators`` and apply engine PD gains.

        Must run after the Genesis scene has been built so joint / DOF state is available.
        """
        actuators_cfg = getattr(robot_cfg, "actuators", None)
        if actuators_cfg is None: return
        self._setup_actuators(actuators_cfg)

    def _setup_actuators(self, actuators_cfg) -> None:
        lab_entity = self._lab_entity
        entity = lab_entity.raw_entity
        lab_entity._actuators = {}

        robot_asset = lab_entity.robot_asset
        if robot_asset is None:
            raise RuntimeError(
                f"Robot asset for entity '{lab_entity.entity_name}' is None. "
                "The robot was not initialized with a Genesis articulation Robot asset."
            )

        joint_names = robot_asset.get_joint_names()
        if not joint_names:
            raise RuntimeError(
                f"Robot '{lab_entity.entity_name}': No actuated joints found; cannot set up actuators."
            )
        joint_name_to_dof_indices = robot_asset.get_all_joint_dof_indices()
        dof_pos = entity.get_dofs_position()
        num_dofs = dof_pos.shape[-1]
        num_envs = dof_pos.shape[0]

        for actuator_name, actuator_cfg in actuators_cfg.items():
            try:
                matched_indices, matched_joint_names = robot_asset.match_joints(
                    actuator_cfg.joint_names_expr
                )
            except ValueError as e:
                raise ValueError(
                    f"Robot '{lab_entity.entity_name}': Actuator '{actuator_name}': {e}\n"
                    f"Available joint names: {joint_names}"
                ) from e

            if not matched_joint_names:
                raise ValueError(
                    f"Robot '{lab_entity.entity_name}': Actuator '{actuator_name}': "
                    f"No joints matched expression {actuator_cfg.joint_names_expr}. "
                    f"Available joint names: {joint_names}"
                )

            matched_dof_indices_full: list[int] = []
            for name in matched_joint_names:
                if name in joint_name_to_dof_indices:
                    matched_dof_indices_full.extend(joint_name_to_dof_indices[name])
            num_actuator_joints = len(matched_dof_indices_full)

            if len(matched_joint_names) == len(joint_names):
                joint_ids_tensor = slice(None)
            else:
                joint_ids_tensor = torch.tensor(matched_indices, dtype=torch.long, device=self._device)

            default_stiffness = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_damping = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_armature = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_friction = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_dynamic_friction = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_viscous_friction = torch.zeros(num_envs, num_actuator_joints, device=self._device)
            default_effort_limit = torch.full((num_envs, num_actuator_joints), float("inf"), device=self._device)
            default_velocity_limit = torch.full((num_envs, num_actuator_joints), float("inf"), device=self._device)

            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=matched_joint_names,
                joint_ids=joint_ids_tensor,
                num_envs=num_envs,
                device=self._device,
                stiffness=default_stiffness,
                damping=default_damping,
                armature=default_armature,
                friction=default_friction,
                dynamic_friction=default_dynamic_friction,
                viscous_friction=default_viscous_friction,
                effort_limit=default_effort_limit,
                velocity_limit=default_velocity_limit,
            )

            lab_entity._actuators[actuator_name] = actuator
            self._finalize_actuator_state(actuator, entity, matched_dof_indices_full, num_dofs)

    def _finalize_actuator_state(
        self,
        actuator: ActuatorBase,
        entity,
        matched_dof_indices_full: list[int],
        num_dofs: int,
    ) -> None:
        """Set joint-space DOF indices on the actuator and sync engine ``kp`` / ``kv``."""
        full_dof_tensor = torch.tensor(matched_dof_indices_full, dtype=torch.long, device=self._device)

        base_offset = 6 if num_dofs > 6 else 0
        joint_space_indices = [idx - base_offset for idx in matched_dof_indices_full if idx >= base_offset]
        actuator._dof_indices = torch.tensor(joint_space_indices, dtype=torch.long, device=self._device)

        kp = actuator.stiffness
        kv = actuator.damping
        if kp.dim() == 2 and not getattr(entity._solver._options, "batch_dofs_info", False):
            kp = kp[0]
            kv = kv[0]
        entity.set_dofs_kp(kp, full_dof_tensor)
        entity.set_dofs_kv(kv, full_dof_tensor)
