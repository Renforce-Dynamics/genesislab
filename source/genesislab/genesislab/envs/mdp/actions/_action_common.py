"""Shared helpers for MDP joint action terms."""

from __future__ import annotations

import warnings

import torch


def prepare_action_batch(actions: torch.Tensor, reference: torch.Tensor, *, term_name: str) -> torch.Tensor:
    """Return ``actions`` with the same shape as ``reference``, or raise.

    If the policy emits a single-env row ``(1, D)`` while the buffer is
    ``(num_envs, D)``, the batch is expanded to match.
    """
    if actions.shape == reference.shape:
        return actions
    if actions.shape[-1] == reference.shape[-1] and actions.shape[0] == 1:
        return actions.expand_as(reference)
    raise ValueError(
        f"Invalid action shape for {term_name}: expected {reference.shape}, got {actions.shape}."
    )


def warn_if_nonfinite_actions(actions: torch.Tensor, term_name: str) -> None:
    """Emit a warning if any action value is NaN or infinite."""
    if torch.isnan(actions).any():
        warnings.warn(f"{term_name}: NaN actions received!", stacklevel=2)
    if torch.isinf(actions).any():
        warnings.warn(f"{term_name}: infinite actions received!", stacklevel=2)
