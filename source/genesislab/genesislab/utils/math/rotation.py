from __future__ import annotations

"""Quaternion and rotation helpers.

All quaternion tensors use the **wxyz** (scalar-first) component order, matching
Genesis / MuJoCo-style layout: ``[..., 0]`` = w, ``[..., 1:4]`` = (x, y, z).
"""

import torch


@torch.jit.script
def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion in [w, x, y, z] (wxyz) format."""
    return quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 * q2``; quaternions are [w, x, y, z] (wxyz)."""
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Unit-quaternion inverse; input/output [w, x, y, z] (wxyz)."""
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by unit quaternion ``quat`` (wxyz). ``vec`` is (..., 3)."""
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    v = vec
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(xyz, t, dim=-1)


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation (equivalent to ``quat_apply(quat_inv(quat), vec)``); quat is wxyz."""
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    v = vec
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w.unsqueeze(-1) * t + torch.cross(xyz, t, dim=-1)


def body_z_axis_world_wxyz(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Body +Z axis expressed in world frame.

    ``root_quat_w`` maps body → world and uses **[w, x, y, z] (wxyz)**, matching
    Genesis ``get_quat()`` / :meth:`~genesislab.engine.entity.lab_entity_data.LabEntityData.root_quat_w`.

    When the base is upright (body +Z aligned with world +Z), the result is
    approximately ``[0, 0, 1]``.
    """
    n = root_quat_w.shape[0]
    ez = torch.zeros(n, 3, device=root_quat_w.device, dtype=root_quat_w.dtype)
    ez[:, 2] = 1.0
    return quat_apply(root_quat_w, ez)


def tilt_angle_rad_from_up_wxyz(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Angle (radians) between body +Z and world +Z from root quaternion (wxyz).

    Uses :func:`body_z_axis_world_wxyz`; result is in ``[0, π]``.
    """
    z_w = body_z_axis_world_wxyz(root_quat_w)
    return torch.acos(z_w[:, 2].clamp(-1.0, 1.0))


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """XYZ Euler (radians) to quaternion [w, x, y, z] (wxyz)."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Convert [w, x, y, z] → [x, y, z, w] for interop with xyzw-only APIs."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack([x, y, z, w], dim=-1)


@torch.jit.script
def quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, z, w] → [w, x, y, z] (e.g. legacy Isaac-style → wxyz)."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion [w, x, y, z] (wxyz) to XYZ Euler (roll, pitch, yaw) in radians."""
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)

    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    sinr_cosr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosr, cosr)

    siny_cosy = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosy, cosy)

    return torch.stack([roll, pitch, yaw], dim=-1)


@torch.jit.script
def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Pure yaw about +Z as quaternion [w, x, y, z] (wxyz)."""
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zero, zero, yaw)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two wxyz quaternions (radians)."""
    dq = quat_mul(quat_inv(q1), q2)
    dq = _normalize_quat(dq)
    w = dq[..., 0].clamp(-1.0, 1.0)
    return 2.0 * torch.acos(torch.abs(w))


@torch.jit.script
def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) from unit quaternion [w, x, y, z] (wxyz)."""
    quat = _normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


@torch.jit.script
def subtract_frame_transforms(
    pos0: torch.Tensor,
    quat0: torch.Tensor,
    pos1: torch.Tensor,
    quat1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative transform from frame-0 to frame-1.

    ``quat0``, ``quat1`` are [w, x, y, z] (wxyz). Returns ``(pos_rel, quat_rel)`` with
    ``T_rel = inv(T0) * T1`` (frame-1 expressed in frame-0).
    """
    quat0_inv = quat_inv(quat0)
    pos_rel = quat_apply(quat0_inv, pos1 - pos0)
    quat_rel = quat_mul(quat0_inv, quat1)
    return pos_rel, quat_rel
