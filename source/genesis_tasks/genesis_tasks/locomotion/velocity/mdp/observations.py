"""Observation terms for Go2 velocity tracking task."""

from __future__ import annotations

import torch


def joint_pos(env) -> torch.Tensor:
    """Joint position observations for Go2.

    Returns:
        Tensor of shape (num_envs, num_dofs).
    """
    dof_pos, _ = env._binding.get_joint_state("go2")
    return dof_pos


def joint_vel(env) -> torch.Tensor:
    """Joint velocity observations for Go2.

    Returns:
        Tensor of shape (num_envs, num_dofs).
    """
    _, dof_vel = env._binding.get_joint_state("go2")
    return dof_vel


def base_lin_vel(env) -> torch.Tensor:
    """Base linear velocity observations in world frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    _, _, lin_vel, _ = env._binding.get_root_state("go2")
    return lin_vel


def base_ang_vel(env) -> torch.Tensor:
    """Base angular velocity observations in world frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    _, _, _, ang_vel = env._binding.get_root_state("go2")
    return ang_vel


def command(env) -> torch.Tensor:
    """Current velocity command observations.

    Returns:
        Tensor of shape (num_envs, C), where C is command dimension.
    """
    # Expect a "lin_vel" command term if present.
    if hasattr(env, "command_manager") and hasattr(env.command_manager, "get_command"):
        try:
            cmd = env.command_manager.get_command("lin_vel")
            return cmd
        except (KeyError, AttributeError):
            pass

    # Fallback: zeros
    return torch.zeros((env.num_envs, 1), device=env.device)
