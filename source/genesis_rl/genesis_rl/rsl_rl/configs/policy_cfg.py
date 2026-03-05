"""Policy network configuration classes for RSL-RL (GenesisLab)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from genesislab.utils.configclass import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """Initial exploration noise std for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """Type of noise std parameterization."""

    state_dependent_std: bool = False
    """Whether to use state-dependent std (default: False)."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize observations for the actor."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize observations for the critic."""

    actor_hidden_dims: list[int] = MISSING
    """Hidden layer dimensions for the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """Hidden layer dimensions for the critic network."""

    activation: str = MISSING
    """Activation function name (e.g. 'elu', 'tanh')."""


__all__ = ["RslRlPpoActorCriticCfg"]

