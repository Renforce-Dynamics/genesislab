"""Common types and configs for GenesisLab environments.

This module mirrors the minimal pieces of IsaacLab's ``envs.common`` needed
for manager-based RL environments, without introducing a hard dependency on
Gym or specific viewer implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, TypeVar, Union

import torch

from genesislab.utils.configclass import configclass


@configclass
class ViewerCfg:
    """Configuration of a simple scene viewport camera.

    This is a minimal subset of IsaacLab's :class:`ViewerCfg` used only for
    offline rendering and basic debugging. More advanced viewer options should
    be added as needed.
    """

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in meters)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in meters)."""

    resolution: tuple[int, int] = (1280, 720)
    """Camera resolution (width, height) in pixels."""

    origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """Frame in which ``eye`` and ``lookat`` are defined."""

    env_index: int = 0
    """Environment index used when ``origin_type`` is ``env`` or asset-relative."""

    asset_name: str = None
    """Optional asset name for asset-relative camera origins."""

    body_name: str = None
    """Optional body name within the asset for asset-relative camera origins."""


# Type aliases for vec env observations and step returns, aligned with IsaacLab.

SpaceType = TypeVar("SpaceType")
"""Placeholder for a valid space type (Gym spaces or simple container types)."""

VecEnvObs = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
"""Observation returned by a vectorized environment.

The top-level dict usually maps group names to either:

- A concatenated observation tensor.
- A nested dict of per-term tensors.
"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment:

1. Observations
2. Rewards
3. Terminated dones
4. Timeout dones
5. Extras dict
"""

