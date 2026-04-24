"""Tests for wxyz (scalar-first) quaternion helpers.

All quaternions in genesislab use the same layout MuJoCo publishes in ``qpos``:
``q[..., 0]`` = w, ``q[..., 1:4]`` = (x, y, z). These tests lock that convention
in — they will fail if anyone silently flips to xyzw.
"""

from __future__ import annotations

import math

import pytest
import torch

from genesislab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_to_euler_xyz,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    yaw_quat,
)


def _identity(batch: int = 2) -> torch.Tensor:
    q = torch.zeros(batch, 4)
    q[:, 0] = 1.0
    return q


# ---------------------------------------------------------------------------
# Convention (wxyz) checks
# ---------------------------------------------------------------------------


def test_identity_has_w_first():
    """The scalar part lives at index 0, not index 3 (MuJoCo convention)."""
    q = _identity(batch=3)
    # w = 1.0, (x, y, z) = 0
    assert torch.allclose(q[:, 0], torch.ones(3))
    assert torch.allclose(q[:, 1:], torch.zeros(3, 3))


def test_quat_apply_identity_preserves_vector():
    q = _identity(batch=4)
    v = torch.tensor([[1.0, 2.0, 3.0]] * 4)
    assert torch.allclose(quat_apply(q, v), v, atol=1e-6)
    assert torch.allclose(quat_apply_inverse(q, v), v, atol=1e-6)


def test_wxyz_xyzw_roundtrip():
    q_wxyz = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # 90° roll
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    # xyzw scalar is at the end
    assert torch.allclose(q_xyzw, torch.tensor([[0.7071, 0.0, 0.0, 0.7071]]))
    assert torch.allclose(quat_xyzw_to_wxyz(q_xyzw), q_wxyz)


# ---------------------------------------------------------------------------
# quat_mul / quat_inv basics
# ---------------------------------------------------------------------------


def test_quat_mul_identity_left_and_right():
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # valid unit quat
    e = _identity(batch=1)
    assert torch.allclose(quat_mul(e, q), q, atol=1e-6)
    assert torch.allclose(quat_mul(q, e), q, atol=1e-6)


def test_quat_inv_then_mul_is_identity():
    roll = torch.tensor([0.3])
    pitch = torch.tensor([-0.4])
    yaw = torch.tensor([0.9])
    q = quat_from_euler_xyz(roll, pitch, yaw)
    prod = quat_mul(q, quat_inv(q))
    # w ≈ 1, xyz ≈ 0 (modulo sign flip on the whole quat)
    assert torch.allclose(prod[:, 0].abs(), torch.ones(1), atol=1e-5)
    assert torch.allclose(prod[:, 1:], torch.zeros(1, 3), atol=1e-5)


def test_quat_mul_is_not_commutative_for_generic_quats():
    # Basic sanity: compose two non-axis-aligned rotations and check ab != ba.
    q1 = quat_from_euler_xyz(torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([0.0]))
    q2 = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.7]), torch.tensor([0.0]))
    ab = quat_mul(q1, q2)
    ba = quat_mul(q2, q1)
    assert not torch.allclose(ab, ba, atol=1e-4)


# ---------------------------------------------------------------------------
# Rotation correctness — check against hand-computed axis rotations
# ---------------------------------------------------------------------------


def test_quat_apply_90deg_about_z_rotates_x_to_y():
    # 90° yaw (about world +Z): +X axis goes to +Y axis.
    q = yaw_quat(torch.tensor([math.pi / 2]))
    x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = quat_apply(q, x_axis)
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_quat_apply_inverse_is_inverse_of_quat_apply():
    q = quat_from_euler_xyz(torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([1.1]))
    v = torch.tensor([[0.7, -0.4, 1.2]])
    assert torch.allclose(quat_apply_inverse(q, quat_apply(q, v)), v, atol=1e-5)
    assert torch.allclose(quat_apply(q, quat_apply_inverse(q, v)), v, atol=1e-5)


def test_matrix_from_quat_identity():
    q = _identity(batch=2)
    R = matrix_from_quat(q)
    eye = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    assert torch.allclose(R, eye, atol=1e-6)


def test_matrix_from_quat_agrees_with_quat_apply():
    q = quat_from_euler_xyz(torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([1.1]))
    v = torch.tensor([[1.0, 0.5, -0.3]])
    R = matrix_from_quat(q)  # (1, 3, 3)
    v_by_matrix = torch.einsum("bij,bj->bi", R, v)
    v_by_quat = quat_apply(q, v)
    assert torch.allclose(v_by_matrix, v_by_quat, atol=1e-5)


# ---------------------------------------------------------------------------
# Euler <-> quat round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roll, pitch, yaw",
    [
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 1.1),
        (-1.2, 0.0, 0.4),
        (0.0, 0.7, -0.5),
    ],
)
def test_euler_quat_roundtrip(roll: float, pitch: float, yaw: float):
    r = torch.tensor([roll])
    p = torch.tensor([pitch])
    y = torch.tensor([yaw])
    q = quat_from_euler_xyz(r, p, y)
    rpy = quat_to_euler_xyz(q)
    assert torch.allclose(rpy[:, 0], r, atol=1e-5)
    assert torch.allclose(rpy[:, 1], p, atol=1e-5)
    assert torch.allclose(rpy[:, 2], y, atol=1e-5)


def test_quat_from_euler_xyz_is_unit_norm():
    r = torch.linspace(-1.0, 1.0, 8)
    p = torch.linspace(-1.0, 1.0, 8)
    y = torch.linspace(-1.0, 1.0, 8)
    q = quat_from_euler_xyz(r, p, y)
    assert torch.allclose(q.norm(dim=-1), torch.ones(8), atol=1e-5)


# ---------------------------------------------------------------------------
# quat_error_magnitude
# ---------------------------------------------------------------------------


def test_quat_error_magnitude_zero_for_equal_quats():
    q = quat_from_euler_xyz(torch.tensor([0.2]), torch.tensor([-0.4]), torch.tensor([0.8]))
    err = quat_error_magnitude(q, q)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_quat_error_magnitude_angle_matches_hand_calc():
    # Rotate 60 degrees about +X axis: expected error magnitude ≈ pi/3.
    theta = math.pi / 3
    r = torch.tensor([theta])
    zero = torch.tensor([0.0])
    q1 = _identity(batch=1)
    q2 = quat_from_euler_xyz(r, zero, zero)
    err = quat_error_magnitude(q1, q2)
    assert torch.allclose(err, torch.tensor([theta]), atol=1e-5)


# ---------------------------------------------------------------------------
# MuJoCo alignment sanity: input quats that LOOK like xyzw should NOT be treated
# as wxyz silently. We don't have a runtime check but we assert the round-trip
# helpers expose the mismatch clearly.
# ---------------------------------------------------------------------------


def test_feeding_xyzw_where_wxyz_expected_produces_wrong_rotation():
    """Guard-rail: demonstrate that convention confusion changes behavior.

    Reviewers modifying rotation helpers in the future should keep this failing
    if they accidentally switch conventions — it proves the helpers are
    convention-sensitive (not symmetric in the leading scalar).
    """
    # wxyz of 90° yaw
    q_wxyz = yaw_quat(torch.tensor([math.pi / 2]))
    # Same numbers reinterpreted as xyzw (scalar moves to end)
    q_fake = quat_wxyz_to_xyzw(q_wxyz)
    x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    good = quat_apply(q_wxyz, x_axis)
    bad = quat_apply(q_fake, x_axis)
    # The correct result sends +X to +Y; the mis-convention-applied one does not.
    assert not torch.allclose(good, bad, atol=1e-5)
