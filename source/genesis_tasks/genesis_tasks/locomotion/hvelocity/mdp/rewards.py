import math
from genesis_tasks.locomotion.velocity.mdp.rewards import *


def is_alive(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Alias for :func:`alive` (SoccerLab / IsaacLab naming)."""
    return alive(env, asset_cfg=asset_cfg)


def track_lin_vel_xy_yaw_frame_exp(
    env: "ManagerBasedRlEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Alias for :func:`track_lin_vel_xy_exp` (command and state are in the base frame)."""

    return track_lin_vel_xy_exp(env, std, command_name, asset_cfg)


def joint_vel_l2(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities with an L2-squared norm over selected (or all) joints."""

    entity = env.entities[asset_cfg.entity_name]
    jv = entity.data.joint_vel
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        jv = jv[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(jv), dim=1)


def mechanical_power_l1(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Approximate mechanical power ``sum(|tau * qdot|)`` for energy-style penalties."""

    entity = env.entities[asset_cfg.entity_name]
    tau = entity.data.applied_torque
    qd = entity.data.joint_vel
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        tau = tau[:, asset_cfg.joint_ids]
        qd = qd[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(tau * qd), dim=1)


def joint_deviation_l1(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L1 deviation of joint positions from the default pose for selected joints."""

    entity = env.entities[asset_cfg.entity_name]
    q = entity.data.joint_pos
    q0 = entity.data.default_joint_pos
    if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids is not None:
        q = q[:, asset_cfg.joint_ids]
        q0 = q0[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(q - q0), dim=1)


def base_height_l2(
    env: "ManagerBasedRlEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for base height error (L2 squared) relative to ``target_height`` (world z)."""

    entity = env.entities[asset_cfg.entity_name]
    z = entity.data.root_pos_w[:, 2]
    return torch.square(z - target_height)


def feet_gait(
    env: "ManagerBasedRlEnv",
    command_name: str,
    period: float,
    offset: list[float],
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Encourage a two-beat gait: feet alternate contact roughly out of phase."""

    contact_sensor = env.scene.sensors[sensor_cfg.entity_name]
    t = env.episode_length_buf.float() * env.step_dt
    phase = (t / period) % 1.0

    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise ValueError("feet_gait requires resolved body_ids on sensor_cfg")

    forces = contact_sensor.data.net_forces_w_history[-1]
    force_mag = torch.norm(forces[:, body_ids, :], dim=-1)
    contact = (force_mag > threshold).float()

    o0, o1 = float(offset[0]), float(offset[1])
    l_des = 0.5 * (1.0 + torch.cos(2.0 * math.pi * (phase + o0)))
    r_des = 0.5 * (1.0 + torch.cos(2.0 * math.pi * (phase + o1)))
    des = torch.stack([l_des, r_des], dim=1)

    err = torch.mean(torch.square(contact - des), dim=1)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :2], dim=1) > 0.05
    return torch.exp(-err / 0.25) * moving.to(err.dtype)


def feet_slide(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize lateral foot speed when feet are in contact (anti-skid)."""

    entity = env.entities[asset_cfg.entity_name]
    contact_sensor = env.scene.sensors[sensor_cfg.entity_name]
    forces = contact_sensor.data.net_forces_w_history[-1]
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        raise ValueError("feet_slide requires resolved body_ids on asset_cfg")

    force_mag = torch.norm(forces[:, body_ids, :], dim=-1)
    in_contact = (force_mag > contact_threshold).float()

    foot_v = entity.data.body_lin_vel_w[:, body_ids, :2]
    slide = torch.sum(torch.square(foot_v), dim=-1) * in_contact
    return torch.sum(slide, dim=1)


def foot_clearance_reward(
    env: "ManagerBasedRlEnv",
    std: float,
    tanh_mult: float,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward feet being lifted toward ``target_height`` clearance above the root height."""

    entity = env.entities[asset_cfg.entity_name]
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        raise ValueError("foot_clearance_reward requires resolved body_ids on asset_cfg")

    root_z = entity.data.root_pos_w[:, 2:3]
    foot_z = entity.data.body_pos_w[:, body_ids, 2]
    clearance = foot_z - root_z
    excess = torch.clamp(clearance - target_height, min=0.0)
    return torch.mean(torch.tanh(tanh_mult * excess / std), dim=1)
