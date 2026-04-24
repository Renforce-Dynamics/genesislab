"""Utility helpers for event terms in velocity locomotion tasks.

Quaternion helpers live in :mod:`genesislab.utils.math.rotation` — do not redefine
``quat_mul`` / ``euler_xyz_to_quat`` here; importers should pull them from the math
package so the wxyz convention has a single source of truth.
"""

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
    """Sample a ``(num_envs, len(keys))`` tensor from a dict of per-key ranges.

    Strict about keys: any entry in ``ranges`` whose key is not in ``keys`` raises
    ``KeyError``. This used to be silently ignored, which hid typos like
    ``{"Z": (-0.1, 0.1)}`` (capital Z) or stray ``"heading"`` keys that never
    influenced the sampled pose.

    Also enforces ``high >= low`` per key so an inverted range fails fast instead
    of producing silently-biased samples.
    """
    allowed = set(keys)
    extras = [k for k in ranges.keys() if k not in allowed]
    if extras:
        raise KeyError(
            f"sample_range_dict: unsupported range key(s) {extras}; "
            f"allowed keys are {list(keys)}."
        )
    lows = []
    highs = []
    for key in keys:
        low, high = ranges.get(key, (0.0, 0.0))
        if high < low:
            raise ValueError(
                f"sample_range_dict: range for key '{key}' has high < low "
                f"({high} < {low})."
            )
        lows.append(low)
        highs.append(high)
    low_t = torch.tensor(lows, device=device, dtype=torch.float32)
    high_t = torch.tensor(highs, device=device, dtype=torch.float32)
    if torch.allclose(low_t, high_t):
        return low_t.unsqueeze(0).expand(num_envs, -1)
    rand = torch.rand((num_envs, len(keys)), device=device)
    return rand * (high_t - low_t) + low_t
