"""Tests for ``LabEntityData`` data-view semantics.

These tests use lightweight fakes — they avoid pulling in Genesis' real scene/solver
(which is too heavy for unit tests) and focus on the logic that has bitten us:

1. ``body_rot_w`` is **base Euler XYZ** (shape ``(num_envs, 3)``), NOT per-link quats,
   despite the name. This mirrors the NPZ motion format's ``body_rot_w`` key. The
   previous version had a docstring claiming ``(num_envs, num_links, 4)`` which was
   off-by-everything — regressing this will fail these tests.

2. ``projected_gravity_b`` reuses the cached ``GRAVITY_VEC_W`` and does not allocate
   a fresh tensor on every property access.

3. ``body_rot_w`` / ``root_euler_xyz_w`` raises a clear error when the entity is
   fixed-base (i.e., fewer than 6 DOFs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
import torch

from genesislab.engine.entity.lab_entity_data import LabEntityData


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class _FakeRawEntity:
    def __init__(self, dofs: torch.Tensor, quat: Optional[torch.Tensor] = None):
        self._dofs = dofs
        self._quat = quat

    def get_dofs_position(self) -> torch.Tensor:
        return self._dofs

    def get_quat(self) -> torch.Tensor:
        assert self._quat is not None
        return self._quat


@dataclass
class _FakeLabEntity:
    name: str
    raw_entity: _FakeRawEntity
    joint_names: List[str] = field(default_factory=list)
    link_names: List[str] = field(default_factory=list)
    raw_link_names: List[str] = field(default_factory=list)


@dataclass
class _FakeScene:
    cfg: Any = None


class _FakeEnv:
    def __init__(self, num_envs: int, raw_entity: _FakeRawEntity, entity_name: str = "robot"):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _FakeScene()
        self._lab = _FakeLabEntity(name=entity_name, raw_entity=raw_entity)
        self.entities: Dict[str, _FakeLabEntity] = {entity_name: self._lab}


def _make_data(num_envs: int = 3, num_joint_dofs: int = 2, quat: Optional[torch.Tensor] = None):
    """Build a ``LabEntityData`` on top of fakes. 6 base DOFs + N joint DOFs."""
    total_dofs = 6 + num_joint_dofs
    dofs = torch.zeros(num_envs, total_dofs)
    # Fill base Euler (idx 3:6) with recognizable values per env.
    for i in range(num_envs):
        dofs[i, 3] = 0.1 * (i + 1)  # roll
        dofs[i, 4] = 0.2 * (i + 1)  # pitch
        dofs[i, 5] = 0.3 * (i + 1)  # yaw
    raw = _FakeRawEntity(dofs=dofs, quat=quat)
    env = _FakeEnv(num_envs=num_envs, raw_entity=raw)
    data = LabEntityData.__new__(LabEntityData)
    data._env = env
    data._scene = env.scene
    data._lab_entity = env._lab
    data._entity_name = "robot"
    data._raw_entity = raw
    data._prev_joint_vel = None
    data._last_acc_step = -1
    data._gravity_vec_w = None
    return data


# ---------------------------------------------------------------------------
# body_rot_w / root_euler_xyz_w
# ---------------------------------------------------------------------------


def test_body_rot_w_is_base_euler_shape_and_values():
    data = _make_data(num_envs=3, num_joint_dofs=4)
    rot = data.body_rot_w
    # Must be base Euler XYZ — shape (num_envs, 3), NOT (num_envs, num_links, 4).
    assert rot.shape == (3, 3)
    # Values match the base DOFs we seeded (env i has (0.1i, 0.2i, 0.3i)).
    expected = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]]
    )
    assert torch.allclose(rot, expected)


def test_root_euler_xyz_w_alias():
    data = _make_data()
    assert torch.equal(data.root_euler_xyz_w, data.body_rot_w)


def test_body_rot_w_raises_when_no_floating_base():
    # Fixed-base entity: only 3 DOFs exposed (e.g. an arm), not enough for base Euler.
    raw = _FakeRawEntity(dofs=torch.zeros(2, 3))
    env = _FakeEnv(num_envs=2, raw_entity=raw)
    data = LabEntityData.__new__(LabEntityData)
    data._env = env
    data._scene = env.scene
    data._lab_entity = env._lab
    data._entity_name = "robot"
    data._raw_entity = raw
    data._prev_joint_vel = None
    data._last_acc_step = -1
    data._gravity_vec_w = None

    with pytest.raises(RuntimeError, match="floating-base"):
        _ = data.root_euler_xyz_w


# ---------------------------------------------------------------------------
# GRAVITY_VEC_W / projected_gravity_b
# ---------------------------------------------------------------------------


def test_gravity_vec_w_is_cached():
    data = _make_data()
    g1 = data.GRAVITY_VEC_W
    g2 = data.GRAVITY_VEC_W
    # Same object → caching works.
    assert g1 is g2
    assert g1.shape == (3, 3)
    assert torch.allclose(g1, torch.tensor([[0.0, 0.0, -1.0]] * 3))


def test_projected_gravity_b_identity_quat_equals_world_gravity():
    # Identity wxyz quaternion → body frame == world frame.
    num_envs = 4
    quat = torch.zeros(num_envs, 4)
    quat[:, 0] = 1.0
    data = _make_data(num_envs=num_envs, quat=quat)
    g_b = data.projected_gravity_b
    assert g_b.shape == (num_envs, 3)
    assert torch.allclose(g_b, torch.tensor([[0.0, 0.0, -1.0]] * num_envs), atol=1e-6)


def test_projected_gravity_b_rolled_90_points_sideways():
    # 90° roll about +X maps world +Z to body -Y, so world (0,0,-1) -> body (0, -1, 0).
    from genesislab.utils.math import quat_from_euler_xyz

    num_envs = 2
    roll = torch.full((num_envs,), math.pi / 2)
    zero = torch.zeros(num_envs)
    quat = quat_from_euler_xyz(roll, zero, zero)
    data = _make_data(num_envs=num_envs, quat=quat)
    g_b = data.projected_gravity_b
    expected = torch.tensor([[0.0, -1.0, 0.0]] * num_envs)
    assert torch.allclose(g_b, expected, atol=1e-6)
