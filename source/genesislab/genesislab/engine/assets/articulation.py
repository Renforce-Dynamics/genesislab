"""Genesis-native articulation asset wrapper.

This module provides a minimal articulation wrapper that mirrors the high-level
intent of :mod:`components.assets.articulation` but operates directly on
Genesis entities (``gs.morphs.URDF`` / ``MJCF`` / ``USD``) without any
dependency on Omniverse or Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import torch

import genesis as gs

from genesislab.engine.assets.base import GenesisAssetBase


@dataclass
class GenesisArticulationCfg:
    """Configuration for a Genesis articulation asset.

    This is intentionally lightweight and aligned with :class:`RobotCfg` so
    that environments can easily construct either binding-level robots or
    explicit asset wrappers.
    """

    name: str
    """Logical name of the articulation asset."""

    morph_type: Literal["URDF", "MJCF", "USD"] = "URDF"
    """Type of Genesis morph to construct."""

    morph_path: str = ""
    """File path to the robot description (URDF/MJCF/USD)."""

    initial_pose: dict[str, Any] = field(
        default_factory=lambda: {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}
    )
    """Initial pose of the articulation root."""

    fixed_base: bool = False
    """Whether the base of the articulation is fixed."""

    control_dofs: list[str] = None
    """List of joint names to control. If None, all actuated joints are controlled."""

    morph_options: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the Genesis morph constructor."""


class GenesisArticulation(GenesisAssetBase):
    """Genesis-native articulation asset.

    This class wraps a single Genesis articulation entity and provides
    batched reset, control and state update operations.
    """

    # NOTE:
    # The original implementation used `device: str | torch.device = gs.device`.
    # However, the public `genesis` PyPI package does not expose a `device` attribute,
    # which results in `AttributeError: module 'genesis' has no attribute 'device'`
    # at import time. To keep this class usable with the current `genesis`
    # distribution, we provide a safe default ("cuda:0" if available, else "cpu")
    # and still allow callers to pass an explicit device.
    def __init__(self, cfg: GenesisArticulationCfg, device: str | torch.device = None):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        if device is None:
            # Prefer CUDA when available; fall back to CPU otherwise.
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._entity: Any = None
        self._dof_indices: torch.Tensor = None

        # Runtime buffers (allocated lazily once entity is available)
        self._targets_pos: torch.Tensor = None
        self._targets_vel: torch.Tensor = None

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def build_into_scene(self, scene: gs.Scene) -> Any:
        """Instantiate the articulation entity and add it to the scene."""

        morph_type = self.cfg.morph_type.upper()
        morph_kwargs = dict(self.cfg.morph_options)
        
        # Handle both dict and PoseCfg object
        if isinstance(self.cfg.initial_pose, dict):
            pos = self.cfg.initial_pose.get("pos", [0.0, 0.0, 0.0])
            quat = self.cfg.initial_pose.get("quat", [0.0, 0.0, 0.0, 1.0])
        else:
            # PoseCfg object
            pos = getattr(self.cfg.initial_pose, "pos", [0.0, 0.0, 0.0])
            quat = getattr(self.cfg.initial_pose, "quat", [0.0, 0.0, 0.0, 1.0])

        if morph_type == "URDF":
            # URDF supports the 'fixed' parameter
            morph_kwargs.setdefault("fixed", self.cfg.fixed_base)
            morph = gs.morphs.URDF(file=self.cfg.morph_path, pos=pos, quat=quat, **morph_kwargs)
        elif morph_type == "MJCF":
            # MJCF does not support the 'fixed' parameter, so we don't pass it
            if "fixed" in morph_kwargs:
                morph_kwargs.pop("fixed")
            morph = gs.morphs.MJCF(file=self.cfg.morph_path, pos=pos, quat=quat, **morph_kwargs)
        elif morph_type == "USD":
            # USD may or may not support 'fixed', so we conditionally add it
            if self.cfg.fixed_base:
                morph_kwargs.setdefault("fixed", self.cfg.fixed_base)
            morph = gs.morphs.USD(file=self.cfg.morph_path, pos=pos, quat=quat, **morph_kwargs)
        else:
            raise ValueError(f"Unsupported morph type for GenesisArticulation: {morph_type}")

        entity = scene.add_entity(morph, name=self.name)
        self._entity = entity

        # Resolve DOF indices (if control_dofs provided)
        if self.cfg.control_dofs is not None:
            indices: list[int] = []
            for joint_name in self.cfg.control_dofs:
                joint = entity.get_joint(joint_name)
                if joint is None:
                    raise ValueError(f"Joint '{joint_name}' not found in articulation '{self.name}'.")
                dof_start = joint.dof_start
                dof_count = getattr(joint, "dof_count", 1)
                indices.extend(range(dof_start, dof_start + dof_count))
            self._dof_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            self._dof_indices = None

        return entity

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #
    def reset(self, env_ids: Sequence[int] | torch.Tensor = None) -> None:
        """Reset root pose and joint state for the given environments."""

        if self._entity is None:
            raise RuntimeError("GenesisArticulation.reset() called before build_into_scene().")

        # Convert env_ids to tensor mask or None
        if env_ids is None:
            env_mask = None
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.detach().to(self.device)
            else:
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            env_mask = env_ids

        # Reset root pose
        pos = torch.tensor(self.cfg.initial_pose.get("pos", [0.0, 0.0, 0.0]), dtype=gs.tc_float, device=gs.device)
        quat = torch.tensor(
            self.cfg.initial_pose.get("quat", [0.0, 0.0, 0.0, 1.0]),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self._entity.set_pos(pos, envs_idx=env_mask)
        self._entity.set_quat(quat, envs_idx=env_mask)

        # Reset joints if DOF indices known (set to zero)
        if self._dof_indices is not None:
            zeros = torch.zeros(
                (self._entity.n_envs, self._dof_indices.numel()),
                dtype=gs.tc_float,
                device=gs.device,
            )
            self._entity.set_dofs_position(zeros, self._dof_indices, envs_idx=env_mask)
            self._entity.set_dofs_velocity(zeros, self._dof_indices, envs_idx=env_mask)

        # Clear any pending targets
        self._targets_pos = None
        self._targets_vel = None

    def set_position_targets(self, targets: torch.Tensor) -> None:
        """Set batched joint position targets for the articulation.

        Parameters
        ----------
        targets:
            Tensor of shape ``(num_envs, num_dofs)`` in Genesis DOF order.
        """

        if self._entity is None:
            raise RuntimeError("GenesisArticulation.set_position_targets() called before build_into_scene().")
        self._targets_pos = targets.to(device=gs.device, dtype=gs.tc_float)

    def set_velocity_targets(self, targets: torch.Tensor) -> None:
        """Set batched joint velocity targets for the articulation."""

        if self._entity is None:
            raise RuntimeError("GenesisArticulation.set_velocity_targets() called before build_into_scene().")
        self._targets_vel = targets.to(device=gs.device, dtype=gs.tc_float)

    def write_data_to_sim(self) -> None:
        """Apply any buffered position / velocity targets to the Genesis entity."""

        if self._entity is None:
            raise RuntimeError("GenesisArticulation.write_data_to_sim() called before build_into_scene().")

        if self._targets_pos is not None:
            if self._dof_indices is not None:
                self._entity.control_dofs_position(self._targets_pos, self._dof_indices)
            else:
                self._entity.control_dofs_position(self._targets_pos)

        if self._targets_vel is not None:
            if self._dof_indices is not None:
                self._entity.control_dofs_velocity(self._targets_vel, self._dof_indices)
            else:
                self._entity.control_dofs_velocity(self._targets_vel)

    def update(self, dt: float) -> None:  # noqa: ARG002 - reserved for future use
        """Update internal state buffers after a simulation step.

        Currently this is a no-op placeholder; state queries should go through
        the underlying Genesis entity directly.
        """

        # In a more feature-complete implementation we would cache state here.
        return

