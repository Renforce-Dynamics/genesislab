"""Robot asset wrapper with name resolution support.

This module provides a Robot class that extends GenesisArticulation with
unified name resolution for joints and bodies/links, handling the differences
between asset formats (URDF, MJCF, USD).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

import genesis as gs

from genesislab.engine.assets.articulation import Articulation, ArticulationCfg
from genesislab.engine.assets.utils.name_normalizer import NameNormalizer

if TYPE_CHECKING:
    from genesislab.engine.gstype import KinematicEntity
    from genesislab.engine.assets.robot_cfg import RobotCfg

class Robot(Articulation):
    """Robot asset wrapper with unified name resolution.
    
    This class extends GenesisArticulation to provide:
    - Unified name resolution for joints and bodies/links across asset formats
    - Pattern matching using normalized names
    - Convenient accessors for joints and bodies by name
    """

    def __init__(self, cfg: "RobotCfg", device: str | torch.device = None):
        """Initialize the robot asset.
        
        Args:
            cfg: Robot configuration.
            device: Device to use for tensors.
        """
        super().__init__(cfg, device=device)
        
        # Name normalizers (initialized after build_into_scene)
        self._joint_normalizer: Optional[NameNormalizer] = None
        self._body_normalizer: Optional[NameNormalizer] = None

    def build_into_scene(self, scene: gs.Scene) -> Any:
        """Instantiate the robot entity and initialize name normalizers.
        
        Args:
            scene: The Genesis scene.
            
        Returns:
            The created entity.
        """
        entity = super().build_into_scene(scene)
        # Initialize name normalizers after entity is built
        self._initialize_name_normalizers(entity)
        return entity

    def _initialize_name_normalizers(self, entity: KinematicEntity) -> None:
        # Initialize joint normalizer
        raw_joint_names = [joint.name for joint in entity.joints]
        if raw_joint_names: self._joint_normalizer = NameNormalizer(raw_joint_names)
        
        # Initialize body/link normalizer
        raw_body_names = [link.name for link in entity.links]
        if raw_body_names: self._body_normalizer = NameNormalizer(raw_body_names)

    # ------------------------------------------------------------------ #
    # Joint name resolution
    # ------------------------------------------------------------------ #

    @property
    def joint_normalizer(self) -> Optional[NameNormalizer]:
        return self._joint_normalizer

    def get_joint_names(self, normalized: bool = True) -> List[str]:
        """Get all joint names.
        
        Args:
            normalized: If True, return normalized names; otherwise return raw names.
            
        Returns:
            List of joint names.
        """
        if self._joint_normalizer is None:
            return []
        if normalized:
            return self._joint_normalizer.normalized_names
        return self._joint_normalizer.raw_names

    def get_actuated_joint_names(self, normalized: bool = False) -> List[str]:
        """Get actuated joint names (excluding base joint and joints without DOFs).
        
        This method filters out:
        - Base joint (name.lower() == "base")
        - Joints without DOFs (no dof_start attribute or dof_start is None)
        
        Args:
            normalized: If True, return normalized names; otherwise return raw names.
            
        Returns:
            List of actuated joint names.
            
        Raises:
            ValueError: If entity or joint normalizer is not initialized.
        """
        if self._entity is None:
            raise ValueError("Entity not initialized. Call build_into_scene() first.")
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        
        raw_joint_names = self._joint_normalizer.raw_names
        actuated_joint_names = []
        
        for raw_joint_name in raw_joint_names:
            # Filter out base joint
            if raw_joint_name.lower() == "base":
                continue
            
            # Check if joint has DOFs
            joint = self._entity.get_joint(raw_joint_name)
            if joint is not None and hasattr(joint, "dof_start") and joint.dof_start is not None:
                actuated_joint_names.append(raw_joint_name)
        
        # Convert to normalized names if requested
        if normalized:
            normalized_names = []
            for raw_name in actuated_joint_names:
                normalized = self._joint_normalizer.get_normalized_name(raw_name)
                if normalized is not None:
                    normalized_names.append(normalized)
            return normalized_names
        
        return actuated_joint_names

    def get_joint(self, name: str, normalized: bool = True) -> Optional[Any]:
        """Get a joint by name.
        
        Args:
            name: Joint name (normalized or raw depending on normalized flag).
            normalized: If True, treat name as normalized; otherwise treat as raw.
            
        Returns:
            Joint object, or None if not found.
        """
        if self._entity is None or self._joint_normalizer is None:
            return None
        
        # Convert to raw name if needed
        if normalized:
            raw_name = self._joint_normalizer.get_raw_name(name)
            if raw_name is None:
                return None
        else:
            raw_name = name
        
        # Get joint from entity
        return self._entity.get_joint(raw_name)

    def match_joints(self, patterns: List[str]) -> Tuple[List[int], List[str]]:
        """Match joint names using regex patterns.
        
        Args:
            patterns: List of regex patterns to match (e.g., ["FL_.*_joint", "FR_.*_joint"]).
            
        Returns:
            Tuple of (matched_indices, matched_normalized_names).
            
        Raises:
            ValueError: If normalizer is not initialized or patterns don't match.
        """
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        return self._joint_normalizer.match_patterns(patterns)

    def get_normalized_joint_names(self) -> List[str]:
        """Get all normalized joint names.
        
        Returns:
            List of normalized joint names (short names).
        """
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        return self._joint_normalizer.normalized_names

    def get_raw_joint_name(self, normalized_name: str) -> Optional[str]:
        """Get the raw joint name from a normalized name.
        
        Args:
            normalized_name: Normalized joint name.
            
        Returns:
            Raw joint name, or None if not found.
        """
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        return self._joint_normalizer.get_raw_name(normalized_name)

    def resolve_joint_values(self, pattern_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve joint values from a pattern dictionary.
        
        This method matches regex patterns against normalized joint names and returns
        a dictionary mapping raw joint names to their resolved values.
        
        Args:
            pattern_dict: Dictionary mapping regex patterns to values
                (e.g., {"FL_.*_joint": 0.5, ".*_calf_joint": -1.0}).
            
        Returns:
            Dictionary mapping raw joint names to resolved values.
            
        Raises:
            ValueError: If normalizer is not initialized or patterns don't match.
        """
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        
        from genesislab.utils.configclass.string import resolve_matching_names_values
        
        normalized_joint_names = self._joint_normalizer.normalized_names
        indices_list, _, values_list = resolve_matching_names_values(
            pattern_dict, normalized_joint_names
        )
        
        # Build mapping from raw joint names to values
        result = {}
        for idx, value in zip(indices_list, values_list):
            normalized_name = normalized_joint_names[idx]
            raw_joint_name = self._joint_normalizer.get_raw_name(normalized_name)
            if raw_joint_name is not None:
                result[raw_joint_name] = value
        
        return result

    def get_all_joint_dof_indices(self) -> Dict[str, List[int]]:
        """Get DOF indices mapping for all joints.
        
        Returns:
            Dictionary mapping raw joint names to their DOF indices.
            
        Raises:
            ValueError: If entity is not initialized.
        """
        if self._entity is None:
            raise ValueError("Entity not initialized. Call build_into_scene() first.")
        
        if self._joint_normalizer is None:
            raise ValueError("Joint normalizer not initialized. Call build_into_scene() first.")
        
        raw_joint_names = self._joint_normalizer.raw_names
        joint_name_to_dof_indices = {}
        
        for raw_joint_name in raw_joint_names:
            joint = self._entity.get_joint(raw_joint_name)
            if joint is not None and hasattr(joint, "dof_start") and joint.dof_start is not None:
                dof_start = joint.dof_start
                dof_count = getattr(joint, "dof_count", 1)
                joint_name_to_dof_indices[raw_joint_name] = list(range(dof_start, dof_start + dof_count))
        
        return joint_name_to_dof_indices

    def get_joint_dof_indices(self, name: str, normalized: bool = True) -> Optional[List[int]]:
        """Get DOF indices for a joint by name.
        
        Args:
            name: Joint name (normalized or raw depending on normalized flag).
            normalized: If True, treat name as normalized; otherwise treat as raw.
            
        Returns:
            List of DOF indices, or None if joint not found.
        """
        joint = self.get_joint(name, normalized=normalized)
        if joint is None:
            return None
        
        if not hasattr(joint, "dof_start") or joint.dof_start is None:
            return None
        
        dof_start = joint.dof_start
        dof_count = getattr(joint, "dof_count", 1)
        return list(range(dof_start, dof_start + dof_count))

    # ------------------------------------------------------------------ #
    # Body/Link name resolution
    # ------------------------------------------------------------------ #

    @property
    def body_normalizer(self) -> Optional[NameNormalizer]:
        """Get the body/link name normalizer.
        
        Returns:
            Body/link name normalizer, or None if not initialized.
        """
        return self._body_normalizer

    def get_body_names(self, normalized: bool = True) -> List[str]:
        """Get all body/link names.
        
        Args:
            normalized: If True, return normalized names; otherwise return raw names.
            
        Returns:
            List of body/link names.
        """
        if self._body_normalizer is None:
            return []
        if normalized:
            return self._body_normalizer.normalized_names
        return self._body_normalizer.raw_names

    def get_body(self, name: str, normalized: bool = True) -> Optional[Any]:
        """Get a body/link by name.
        
        Args:
            name: Body/link name (normalized or raw depending on normalized flag).
            normalized: If True, treat name as normalized; otherwise treat as raw.
            
        Returns:
            Body/link object, or None if not found.
        """
        if self._entity is None or self._body_normalizer is None:
            return None
        
        # Convert to raw name if needed
        if normalized:
            raw_name = self._body_normalizer.get_raw_name(name)
            if raw_name is None:
                return None
        else:
            raw_name = name
        
        # Try to get body/link from entity
        # Different asset formats may expose different APIs
        if hasattr(self._entity, "get_link"):
            # Try by index first (if we can enumerate)
            if hasattr(self._entity, "n_links"):
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
        """Match body/link names using regex patterns.
        
        Args:
            patterns: List of regex patterns to match.
            
        Returns:
            Tuple of (matched_indices, matched_normalized_names).
            
        Raises:
            ValueError: If normalizer is not initialized or patterns don't match.
        """
        if self._body_normalizer is None:
            raise ValueError("Body normalizer not initialized. Call build_into_scene() first.")
        return self._body_normalizer.match_patterns(patterns)
