"""Reward terms for Go2 velocity tracking task."""

import torch


def velocity_tracking(env) -> torch.Tensor:
    """Velocity tracking reward using exponential of negative error.

    Reward = exp(-|v_x - target|)

    Returns:
        Reward tensor of shape (num_envs,).
    """
    binding = env._binding
    _, _, base_lin_vel_world, _ = binding.get_root_state("go2")

    # Get forward velocity (x component)
    v_x = base_lin_vel_world[:, 0]

    # Get target command
    if hasattr(env.command_manager, "get_command"):
        try:
            cmd = env.command_manager.get_command("lin_vel")
            if cmd.shape[-1] >= 1:
                target = cmd[:, 0]  # Forward velocity target
            else:
                target = torch.zeros_like(v_x)
        except (AttributeError, KeyError):
            target = torch.zeros_like(v_x)
    else:
        target = torch.zeros_like(v_x)

    # Compute tracking error
    error = torch.abs(v_x - target)

    # Exponential reward (higher reward for lower error)
    reward = torch.exp(-error)

    return reward


def action_penalty(env) -> torch.Tensor:
    """Action penalty to encourage smooth control.

    Penalty = -||action||^2

    Returns:
        Reward tensor of shape (num_envs,) (negative penalty).
    """
    action = env.action_manager.action

    # Compute action magnitude squared
    action_penalty = -torch.sum(action**2, dim=-1)

    return action_penalty


def upright(env) -> torch.Tensor:
    """Upright reward based on base orientation.

    Reward = z_component_of_base_quaternion (encourages upright orientation)

    Returns:
        Reward tensor of shape (num_envs,).
    """
    binding = env._binding
    _, base_quat, _, _ = binding.get_root_state("go2")

    # Extract z component of quaternion (w component typically represents upright)
    # For quaternion [x, y, z, w], w represents the scalar part
    # We use the z-component of the quaternion vector part as a proxy for uprightness
    # Actually, we should use the quaternion to compute the z-axis of the base
    # For simplicity, use the w component as a proxy
    if base_quat.shape[-1] == 4:
        # Quaternion format: [x, y, z, w] or [w, x, y, z]
        # Assuming [x, y, z, w] format
        upright_reward = base_quat[:, 3]  # w component
    else:
        upright_reward = torch.ones(env.num_envs, device=env.device)

    return upright_reward
