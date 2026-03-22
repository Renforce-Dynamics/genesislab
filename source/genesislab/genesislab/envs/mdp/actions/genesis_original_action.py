"""Implicit joint position targets via Genesis ``control_dofs_position`` (built-in PD)."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ActuatorBase
from genesislab.envs.mdp.actions._action_common import prepare_action_batch, warn_if_nonfinite_actions
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _ensure_dof_pattern(value: float | dict[str, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {".*": float(value)}
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items()}
    raise TypeError(f"Expected float or dict, got {type(value)}")


def _genesis_rigid_entity(scene_entity) -> object:
    """Return the backing Genesis entity; :class:`~genesislab.engine.entity.lab_entity.LabEntity` uses ``raw_entity``."""
    return getattr(scene_entity, "raw_entity", scene_entity)


def _resolve_joint(scene_entity, joint_name: str):
    """Look up a Genesis joint by actuator joint name (normalized, from :class:`ActuatorBase`)."""
    robot_asset = getattr(scene_entity, "robot_asset", None)
    if robot_asset is not None:
        get_joint = getattr(robot_asset, "get_joint", None)
        if get_joint is None:
            return None
        try:
            return get_joint(joint_name)
        except Exception:
            return None
    rigid = _genesis_rigid_entity(scene_entity)
    get_joint = getattr(rigid, "get_joint", None)
    if get_joint is None:
        return None
    try:
        return get_joint(joint_name)
    except Exception:
        return None


def _layout_when_actuators_defined(
    entity_obj, actuators: dict[str, ActuatorBase]
) -> tuple[list[str], dict[str, int], list[int], int]:
    """Map each actuator joint to scene DOF indices; action width is sum of ``num_joints`` per actuator."""
    joint_names: list[str] = []
    joint_name_to_index: dict[str, int] = {}
    dofs_idx: list[int] = []
    idx = 0
    for actuator in actuators.values():
        for joint_name in actuator.joint_names:
            joint = _resolve_joint(entity_obj, joint_name)
            if joint is None or not hasattr(joint, "dof_start") or joint.dof_start is None:
                continue
            joint_names.append(joint_name)
            joint_name_to_index[joint_name] = idx
            dof_start = joint.dof_start
            dof_count = getattr(joint, "dof_count", 1)
            dofs_idx.extend(range(dof_start, dof_start + dof_count))
            idx += 1
    action_dim = sum(a.num_joints for a in actuators.values())
    return joint_names, joint_name_to_index, dofs_idx, action_dim


def _layout_from_articulation_joints(entity_obj) -> tuple[list[str], dict[str, int], list[int], int]:
    """All non-base joints with DOFs, in articulation order (no actuator grouping)."""
    joint_names: list[str] = []
    joint_name_to_index: dict[str, int] = {}
    dofs_idx: list[int] = []
    idx = 0
    rigid = _genesis_rigid_entity(entity_obj)
    joints = getattr(rigid, "joints", None)
    if joints is None:
        raise RuntimeError(
            "GenesisOriginalAction: Genesis entity has no `joints` attribute; "
            "cannot build layout without actuators. Use a robot entity or define actuators."
        )
    for joint in joints:
        if hasattr(joint, "name") and joint.name.lower() == "base":
            continue
        if not hasattr(joint, "dof_start") or joint.dof_start is None:
            continue
        name = joint.name
        joint_names.append(name)
        joint_name_to_index[name] = idx
        dof_start = joint.dof_start
        dof_count = getattr(joint, "dof_count", 1)
        dofs_idx.extend(range(dof_start, dof_start + dof_count))
        idx += 1
    return joint_names, joint_name_to_index, dofs_idx, idx


class GenesisOriginalAction(ActionTerm):
    """Implicit position control: targets are sent with Genesis PD (``control_dofs_position``).

    Policy commands are mapped with ``target = offset + scale * action`` (per-joint patterns for
    scale/offset), optionally clipped, then applied only on the listed articulation DOFs—never on
    the floating-base DOFs (the joint named ``base`` is skipped when enumerating from the entity).

    When the entity defines actuators, action dimension follows actuator joint counts and joint
    order matches actuator iteration; otherwise it follows all non-base DOF joints on the entity.
    """

    cfg: "GenesisOriginalActionCfg"

    def __init__(self, cfg: "GenesisOriginalActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        self._entity_name = cfg.entity_name
        scene_entity = env.scene.entities[self._entity_name]
        self._actuators: dict[str, ActuatorBase] = scene_entity.actuators

        if self._actuators:
            self._joint_names, self._joint_name_to_index, self._dofs_idx, self._action_dim = (
                _layout_when_actuators_defined(scene_entity, self._actuators)
            )
        else:
            self._joint_names, self._joint_name_to_index, self._dofs_idx, self._action_dim = (
                _layout_from_articulation_joints(scene_entity)
            )

        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)

        self._offset_cfg = _ensure_dof_pattern(cfg.offset)
        self._scale_cfg = _ensure_dof_pattern(cfg.scale)
        if cfg.clip is not None and not isinstance(cfg.clip, tuple):
            raise TypeError(
                f"GenesisOriginalAction clip must be tuple[float, float] or None, got {type(cfg.clip)}"
            )
        self._use_default_offset = cfg.use_default_offset

        if self._use_default_offset and self._offset_cfg is not None and self._offset_cfg.get(".*") != 0.0:
            raise ValueError("Cannot set both use_default_offset=True and a non-zero default-pattern offset")

        self._scale_values: torch.Tensor | None = None
        self._offset_values: torch.Tensor | None = None

        self._build_affine_tensors()
        self._build_clip_bounds(self._action_dim, cfg.clip, joint_names=None)

    def _build_affine_tensors(self) -> None:
        n = self._action_dim
        self._scale_values = torch.ones(n, device=self.device)
        self._offset_values = torch.zeros(n, device=self.device)

        if self._use_default_offset:
            entity = self._env.entities[self._entity_name]
            default_joint_pos = entity.data.default_joint_pos
            if default_joint_pos.shape[-1] >= n:
                self._offset_values[:] = default_joint_pos[0, :n].clone()
            else:
                self._offset_values[: default_joint_pos.shape[-1]] = default_joint_pos[0].clone()

        if self._scale_cfg is not None:
            self._apply_pattern_dict(self._scale_cfg, self._scale_values, default_value=1.0)

        if not self._use_default_offset and self._offset_cfg is not None:
            self._apply_pattern_dict(self._offset_cfg, self._offset_values, default_value=0.0)

    def _apply_pattern_dict(
        self,
        pattern_dict: dict[str, float],
        output: torch.Tensor,
        *,
        default_value: float,
    ) -> None:
        matched = [False] * len(self._joint_names)
        for pattern, value in pattern_dict.items():
            found = False
            for i, joint_name in enumerate(self._joint_names):
                if matched[i]:
                    continue
                if re.match(f"^{pattern}$", joint_name):
                    output[self._joint_name_to_index[joint_name]] = float(value)
                    matched[i] = True
                    found = True
            if not found and pattern != ".*":
                warnings.warn(
                    f"GenesisOriginalAction: no joints matched pattern {pattern!r}; "
                    f"available: {self._joint_names}",
                    stacklevel=2,
                )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = prepare_action_batch(actions, self._raw_action, term_name="GenesisOriginalAction")
        self._raw_action[:] = actions
        warn_if_nonfinite_actions(actions, "GenesisOriginalAction")

        self._targets[:] = self._offset_values.unsqueeze(0) + self._scale_values.unsqueeze(0) * actions
        self._targets[:] = self._apply_clip(self._targets)

    def apply_actions(self) -> None:
        lab = self._env.scene.entities[self._entity_name]
        raw = _genesis_rigid_entity(lab)
        dof_list = self._dofs_idx if self._dofs_idx else None
        raw.control_dofs_position(self._targets, dof_list)


@configclass
class GenesisOriginalActionCfg(ActionTermCfg):
    """Config for :class:`GenesisOriginalAction` (implicit Genesis PD position control)."""

    class_type: type = GenesisOriginalAction
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    use_default_offset: bool = True
    clip: tuple[float, float] | None = None
    """Uniform clip ``(low, high)`` on position targets after the affine map (no per-joint dict)."""
