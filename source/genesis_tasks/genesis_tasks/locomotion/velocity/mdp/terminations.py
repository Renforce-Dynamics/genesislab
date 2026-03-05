"""Termination conditions for Go2 velocity tracking task."""

import torch


def base_height(env) -> torch.Tensor:
    """Base height termination condition.

    Terminates if base height falls below threshold.

    Returns:
        Boolean tensor of shape (num_envs,) indicating fallen robots.
    """
    binding = env._binding
    base_pos, _, _, _ = binding.get_root_state("go2")

    # Get threshold from config
    threshold = env.cfg.base_height_threshold

    # Check if base is too low (fallen)
    fallen = base_pos[:, 2] < threshold

    return fallen


def time_out(env) -> torch.Tensor:
    """Timeout termination condition.

    This is handled by the environment's finite horizon logic.
    Return all False here; timeouts are handled in base_env.

    Returns:
        Boolean tensor of shape (num_envs,) (all False).
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
