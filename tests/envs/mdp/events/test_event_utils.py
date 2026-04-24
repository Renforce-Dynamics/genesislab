"""Tests for event sampling helpers (``sample_range_dict`` strictness).

Previously ``sample_range_dict`` silently dropped unknown keys. That hid typos
like ``{"Z": (-0.1, 0.1)}`` (capital Z) or legacy keys like ``"heading"`` that
never actually perturbed the state. The strict behavior is now enforced and
these tests lock it in.
"""

from __future__ import annotations

import pytest
import torch

from genesislab.envs.mdp.events.utils import sample_range, sample_range_dict


POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def test_sample_range_dict_basic_shape():
    out = sample_range_dict({"x": (-0.1, 0.1)}, keys=POSE_KEYS, num_envs=5, device="cpu")
    assert out.shape == (5, 6)


def test_sample_range_dict_zero_range_returns_constant():
    out = sample_range_dict({}, keys=POSE_KEYS, num_envs=4, device="cpu")
    assert torch.allclose(out, torch.zeros(4, 6))


def test_sample_range_dict_within_bounds():
    torch.manual_seed(0)
    ranges = {"x": (-0.3, 0.3), "yaw": (0.5, 0.6)}
    out = sample_range_dict(ranges, keys=POSE_KEYS, num_envs=64, device="cpu")
    # x samples within [-0.3, 0.3]
    assert (out[:, 0] >= -0.3).all() and (out[:, 0] <= 0.3).all()
    # yaw samples within [0.5, 0.6]
    assert (out[:, 5] >= 0.5).all() and (out[:, 5] <= 0.6).all()
    # unspecified keys stay at 0
    for col in (1, 2, 3, 4):
        assert torch.allclose(out[:, col], torch.zeros(64))


def test_sample_range_dict_rejects_unknown_keys():
    with pytest.raises(KeyError, match="unsupported range key"):
        sample_range_dict({"Z": (-0.1, 0.1)}, keys=POSE_KEYS, num_envs=2, device="cpu")


def test_sample_range_dict_rejects_typo_heading_key():
    with pytest.raises(KeyError):
        sample_range_dict({"heading": (0.0, 1.0)}, keys=POSE_KEYS, num_envs=2, device="cpu")


def test_sample_range_dict_rejects_inverted_range():
    with pytest.raises(ValueError, match="high < low"):
        sample_range_dict({"x": (0.5, -0.5)}, keys=POSE_KEYS, num_envs=1, device="cpu")


def test_sample_range_degenerate():
    """Zero-width range returns a constant tensor of the requested shape."""
    out = sample_range(0.7, 0.7, (3, 2), device="cpu")
    assert out.shape == (3, 2)
    assert torch.allclose(out, torch.full((3, 2), 0.7))


def test_sample_range_non_degenerate_within_bounds():
    torch.manual_seed(1)
    out = sample_range(-2.0, 1.0, (128,), device="cpu")
    assert out.shape == (128,)
    assert (out >= -2.0).all() and (out <= 1.0).all()
