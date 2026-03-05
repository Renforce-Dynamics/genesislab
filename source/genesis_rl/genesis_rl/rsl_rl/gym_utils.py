"""Helpers for loading env / RL configs from Gymnasium specs.

These utilities centralize the logic for:

- Reading ``env_cfg_entry_point`` and ``rsl_rl_cfg_entry_point`` from
  Gymnasium registration kwargs.
- Importing the referenced objects without instantiating them, so that
  configclass-decorated classes or instances can be consumed directly by
  higher-level scripts (train / play / eval).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


def get_spec_kwargs(env_id: str) -> dict[str, Any]:
    """Return the registration kwargs dict for a given env id."""
    spec = gym.spec(env_id)
    return getattr(spec, "kwargs", {}) or {}


def resolve_env_cfg_entry_point(env_id: str) -> str:
    """Resolve the ``env_cfg_entry_point`` string from a Gymnasium env spec."""
    spec_kwargs = get_spec_kwargs(env_id)
    entry_point = spec_kwargs.get("env_cfg_entry_point", None)
    if entry_point is None:
        raise ValueError(
            f"Gym env '{env_id}' is missing 'env_cfg_entry_point' in its registration kwargs."
        )
    return entry_point


def resolve_rsl_rl_cfg_object(env_id: str) -> Any | None:
    """Resolve the RSL-RL configuration object from a Gymnasium env spec.

    This function:
    - Reads ``rsl_rl_cfg_entry_point`` from the env spec kwargs.
    - If present and a string ``\"module.path:AttrName\"``, imports and returns
      the referenced attribute *without* instantiation.
    - If the value is already a non-string object (e.g. a configclass or dict),
      returns it as is.

    Returns:
        The resolved configuration object, or ``None`` if the spec did not
        define ``rsl_rl_cfg_entry_point``.
    """
    spec_kwargs = get_spec_kwargs(env_id)
    entry = spec_kwargs.get("rsl_rl_cfg_entry_point", None)
    if entry is None:
        return None

    # If the entry is already an object (e.g. configclass-decorated), return as-is.
    if not isinstance(entry, str):
        return entry

    module_name, attr = entry.split(":")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


__all__ = ["get_spec_kwargs", "resolve_env_cfg_entry_point", "resolve_rsl_rl_cfg_object"]

