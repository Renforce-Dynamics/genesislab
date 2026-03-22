"""Explicit actuator path: PD (or other model) in our code, torques via ``control_dofs_force``."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ArticulationActions
from genesislab.envs.mdp.actions._action_common import prepare_action_batch, warn_if_nonfinite_actions
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass
from genesislab.utils.configclass.string import resolve_matching_names_values

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


class JointPositionAction(ActionTerm):
    """Explicit torque command through a configured :class:`~genesislab.components.actuators.ActuatorBase`.

    Flow each physics substep:

    1. Build :class:`~genesislab.components.actuators.articulation_actions.ArticulationActions`
       with desired joint positions (actuator joint order).
    2. :meth:`~genesislab.components.actuators.actuator_base.ActuatorBase.compute` turns targets and
       current joint state into ``joint_efforts`` using the actuator's stored gains/limits.
    3. :meth:`~genesislab.components.actuators.actuator_base.ActuatorBase.apply_torques` writes
       forces on the correct full-DOF indices (base offset applied inside; see actuator base).

    Requires ``actuator_name`` to match an actuator on the entity. Controlled degrees of freedom are
    exactly that actuator's ``dof_indices`` (joint space without base), so base DOFs are never driven
    by this term.
    """

    cfg: "JointPositionActionCfg"

    def __init__(self, cfg: "JointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._entity_name = cfg.entity_name
        entity = env.entities[self._entity_name]

        if cfg.actuator_name not in entity.actuators:
            raise ValueError(
                f"Actuator {cfg.actuator_name!r} not found on entity {self._entity_name!r}. "
                f"Available: {list(entity.actuators.keys())}"
            )
        self._actuator = entity.actuators[cfg.actuator_name]
        self._action_dim = self._actuator.num_joints

        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)

        self._offset = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        if cfg.use_default_offset and entity.data.default_joint_pos is not None:
            self._offset[:] = entity.data.default_joint_pos[:, self._actuator.dof_indices]

        actuator_joint_names = self._actuator.joint_names
        if cfg.offset != 0.0:
            if isinstance(cfg.offset, dict):
                _, _, offset_values = resolve_matching_names_values(
                    cfg.offset, actuator_joint_names, preserve_order=False
                )
                self._offset += torch.tensor(
                    offset_values, dtype=self._offset.dtype, device=self.device
                ).unsqueeze(0)
            else:
                self._offset += float(cfg.offset)

        if isinstance(cfg.scale, dict):
            _, _, scale_values = resolve_matching_names_values(
                cfg.scale, actuator_joint_names, preserve_order=False
            )
            self._scale = torch.tensor(scale_values, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            self._scale = float(cfg.scale)

        self._build_clip_bounds(self._action_dim, cfg.clip, actuator_joint_names)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = prepare_action_batch(actions, self._raw_action, term_name="JointPositionAction")
        self._raw_action[:] = actions
        warn_if_nonfinite_actions(actions, "JointPositionAction")

        self._targets[:] = self._offset + self._scale * actions
        self._targets[:] = self._apply_clip(self._targets)

    def apply_actions(self) -> None:
        entity = self._env.entities[self._entity_name]
        raw_entity = entity.raw_entity
        joint_pos = entity.data.joint_pos[:, self._actuator.dof_indices]
        joint_vel = entity.data.joint_vel[:, self._actuator.dof_indices]

        control_action = ArticulationActions(
            joint_positions=self._targets,
            joint_velocities=None,
            joint_efforts=None,
            joint_indices=None,
        )
        control_action = self._actuator.compute(control_action, joint_pos=joint_pos, joint_vel=joint_vel)

        if control_action.joint_efforts is None:
            raise RuntimeError(
                "actuator.compute() must set joint_efforts for JointPositionAction "
                "(e.g. IdealPDActuator or ImplicitActuator)."
            )
        self._actuator.apply_torques(raw_entity, control_action.joint_efforts)


@configclass
class JointActionCfg(ActionTermCfg):
    """Base joint action cfg; ``joint_names`` is reserved for Isaac-style matchers / docs."""

    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Config for :class:`JointPositionAction` (explicit actuator compute + force application)."""

    class_type: type = JointPositionAction
    actuator_name: str = MISSING
    use_default_offset: bool = True
