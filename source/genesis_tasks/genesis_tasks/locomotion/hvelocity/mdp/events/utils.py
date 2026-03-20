"""Utility helpers for event terms in velocity locomotion tasks."""

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def resolve_env_ids(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
    """Normalize env_ids to a 1D tensor on the correct device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(env.device)


def sample_range(
    low: float,
    high: float,
    shape: Tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    """Sample uniformly from [low, high] with the given shape."""
    if low == high:
        return torch.full(shape, float(low), device=device)
    return torch.rand(shape, device=device) * (high - low) + low


def sample_range_dict(
    ranges: Dict[str, Tuple[float, float]],
    keys: tuple[str, ...],
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample a (num_envs, len(keys)) tensor from a dict of per-key ranges."""
    lows = []
    highs = []
    for key in keys:
        low, high = ranges.get(key, (0.0, 0.0))
        lows.append(low)
        highs.append(high)
    low_t = torch.tensor(lows, device=device, dtype=torch.float32)
    high_t = torch.tensor(highs, device=device, dtype=torch.float32)
    if torch.allclose(low_t, high_t):
        return low_t.expand(num_envs, -1)
    rand = torch.rand((num_envs, len(keys)), device=device)
    return rand * (high_t - low_t) + low_t


def euler_xyz_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to quaternions in [x, y, z, w] format.

    Shapes:
        roll, pitch, yaw: (...,)
        return: (..., 4)
    """
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in [x, y, z, w] format.

    Both q1 and q2 have shape (..., 4).
    Returns q = q1 * q2 with the same shape.
    """
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)

