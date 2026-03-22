"""Robot asset wrapper with name resolution (joints, links) across URDF/MJCF/USD."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

import genesis as gs

from genesislab.engine.assets.articulation import Articulation, ArticulationCfg
from genesislab.engine.assets.utils.name_normalizer import NameNormalizer
from genesislab.engine.assets.robot.actuator_manager import ActuatorManager

if TYPE_CHECKING:
    from genesislab.engine.gstype import KinematicEntity
    from genesislab.components.actuators import ActuatorBase
    from .robot_cfg import RobotCfg


class Robot(Articulation):
    """Articulation with normalized joint/link names (actuated joints only for joint API)."""

    def __init__(self, cfg: "RobotCfg", device: str | torch.device = None):
        super().__init__(cfg, device=device)
        self._joint_normalizer: Optional[NameNormalizer] = None
        self._body_normalizer: Optional[NameNormalizer] = None
        self._actuators: Dict[str, "ActuatorBase"] = {}
        self._actuator_manager: ActuatorManager | None = None
        self._actuators_initialized: bool = False

    def build_into_scene(self, scene: gs.Scene) -> Any:
        entity = super().build_into_scene(scene)
        self._initialize_name_normalizers(entity)
        return entity

    def _initialize_name_normalizers(self, entity: KinematicEntity) -> None:
        raw_joint_names = [joint.name for joint in entity.joints]
        # Index 0 is the floating-base joint; actuated API uses joints[1:] with the same NameNormalizer rules.
        actuated_raw = raw_joint_names[1:] if len(raw_joint_names) > 1 else []
        if actuated_raw:
            self._joint_normalizer = NameNormalizer(actuated_raw)

        raw_body_names = [link.name for link in entity.links]
        if raw_body_names:
            self._body_normalizer = NameNormalizer(raw_body_names)

    def initialize_actuators(self) -> None:
        """Create actuators after :meth:`gs.Scene.build` (DOF queries require a built entity)."""
        if self._actuators_initialized: return
        self._actuators = {}
        self._actuator_manager = ActuatorManager(self, device=self.device)
        self._actuator_manager.setup()
        self._actuators_initialized = True

    @property
    def actuators(self) -> Dict[str, "ActuatorBase"]:
        return self._actuators

    @property
    def actuator_manager(self) -> ActuatorManager | None:
        return self._actuator_manager

    # --- joints (normalized names only; raw names: joint_normalizer.raw_names) ---

    @property
    def joint_normalizer(self) -> Optional[NameNormalizer]:
        return self._joint_normalizer

    def get_joint_names(self) -> List[str]:
        if self._joint_normalizer is None:
            return []
        return self._joint_normalizer.normalized_names

    def get_joint(self, name: str) -> Optional[Any]:
        if self._entity is None or self._joint_normalizer is None:
            return None
        raw_name = self._joint_normalizer.get_raw_name(name)
        if raw_name is None:
            return None
        return self._entity.get_joint(raw_name)

    def match_joints(self, patterns: List[str]) -> Tuple[List[int], List[str]]:
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        return self._joint_normalizer.match_patterns(patterns)

    def resolve_joint_values(self, pattern_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        from genesislab.utils.configclass.string import resolve_matching_names_values

        names = self._joint_normalizer.normalized_names
        indices_list, _, values_list = resolve_matching_names_values(pattern_dict, names)
        return {names[i]: v for i, v in zip(indices_list, values_list)}

    def get_all_joint_dof_indices(self) -> Dict[str, List[int]]:
        if self._entity is None or self._joint_normalizer is None:
            raise ValueError("Robot not initialized. Call build_into_scene() first.")
        out: Dict[str, List[int]] = {}
        for raw_joint_name in self._joint_normalizer.raw_names:
            joint = self._entity.get_joint(raw_joint_name)
            if joint is None or not hasattr(joint, "dof_start") or joint.dof_start is None:
                continue
            n = self._joint_normalizer.get_normalized_name(raw_joint_name)
            if n is None:
                continue
            dof_start = joint.dof_start
            dof_count = getattr(joint, "dof_count", 1)
            out[n] = list(range(dof_start, dof_start + dof_count))
        return out

    def get_joint_dof_indices(self, name: str) -> Optional[List[int]]:
        joint = self.get_joint(name)
        if joint is None or not hasattr(joint, "dof_start") or joint.dof_start is None:
            return None
        dof_start = joint.dof_start
        dof_count = getattr(joint, "dof_count", 1)
        return list(range(dof_start, dof_start + dof_count))

    # --- bodies / links ---

    @property
    def body_normalizer(self) -> Optional[NameNormalizer]:
        return self._body_normalizer

    def get_body_names(self, normalized: bool = True) -> List[str]:
        if self._body_normalizer is None:
            return []
        if normalized:
            return self._body_normalizer.normalized_names
        return self._body_normalizer.raw_names

    def get_body(self, name: str, normalized: bool = True) -> Optional[Any]:
        if self._entity is None or self._body_normalizer is None:
            return None
        if normalized:
            raw_name = self._body_normalizer.get_raw_name(name)
            if raw_name is None:
                return None
        else:
            raw_name = name

        if hasattr(self._entity, "get_link") and hasattr(self._entity, "n_links"):
            for i in range(self._entity.n_links):
                link = self._entity.get_link(i)
                if hasattr(link, "name") and link.name == raw_name:
                    return link
        elif hasattr(self._entity, "get_body"):
            body = self._entity.get_body(raw_name)
            if body is not None:
                return body
        return None

    def match_bodies(self, patterns: List[str]) -> Tuple[List[int], List[str]]:
        if self._body_normalizer is None:
            raise ValueError("Body normalizer not initialized. Call build_into_scene() first.")
        return self._body_normalizer.match_patterns(patterns)
