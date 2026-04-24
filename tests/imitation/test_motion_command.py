"""Tests for ``MotionCommand`` orientation semantics.

Focus: :attr:`MotionCommand.robot_body_rot_w` used to do
``self.robot.data.body_rot_w[:, self.body_indexes]`` while ``body_rot_w`` is a
``(num_envs, 3)`` base Euler tensor (not per-link). That produced silent index
errors or wrong data every time the tracking MDP was stepped. The property now
mirrors the motion-side ``body_rot_w`` semantics: base Euler only, no body
indexing. This test pins that behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from genesis_tasks.imitation.tracking.mdp.commands import MotionCommand


def _fake_command(num_envs: int, num_bodies: int) -> SimpleNamespace:
    """Shape-compatible fake: just enough for the property accessors to work."""
    # body_rot_w on data is base Euler XYZ: (num_envs, 3)
    data = SimpleNamespace(
        body_rot_w=torch.tensor([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(num_envs)]),
        body_pos_w=torch.zeros(num_envs, num_bodies, 3),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_bodies).unsqueeze(0).expand(num_envs, -1, -1).clone(),
    )
    robot = SimpleNamespace(data=data)
    return SimpleNamespace(
        robot=robot,
        body_indexes=torch.tensor([0, 1, 2], dtype=torch.long),
    )


def test_robot_body_rot_w_returns_base_euler_shape():
    cmd = _fake_command(num_envs=4, num_bodies=5)
    rot = MotionCommand.robot_body_rot_w.fget(cmd)
    # Must be (num_envs, 3) — base Euler — not (num_envs, num_bodies, ...).
    assert rot.shape == (4, 3)
    # Values match the fake: env i has (0.1i, 0.2i, 0.3i).
    expected = torch.tensor([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(4)])
    assert torch.allclose(rot, expected)


def test_robot_body_rot_w_does_not_index_body_indexes():
    """Regression: with >3 body_indexes, the old ``[:, body_indexes]`` would fail."""
    cmd = _fake_command(num_envs=2, num_bodies=8)
    # Build body_indexes that would index OOB on the (2, 3) base Euler tensor.
    cmd.body_indexes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    # Property must still work — it does NOT index with body_indexes anymore.
    rot = MotionCommand.robot_body_rot_w.fget(cmd)
    assert rot.shape == (2, 3)
