"""Algorithm configuration classes for RSL-RL (GenesisLab)."""

from __future__ import annotations

from dataclasses import MISSING

from genesislab.utils.configclass import configclass


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """Algorithm class name."""

    num_learning_epochs: int = MISSING
    """Number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """Number of mini-batches per update."""

    learning_rate: float = MISSING
    """Learning rate."""

    schedule: str = MISSING
    """Learning rate schedule."""

    gamma: float = MISSING
    """Discount factor."""

    lam: float = MISSING
    """GAE lambda."""

    entropy_coef: float = MISSING
    """Entropy regularization coefficient."""

    desired_kl: float = MISSING
    """Target KL for adaptive schedulers."""

    max_grad_norm: float = MISSING
    """Gradient clipping value."""

    value_loss_coef: float = MISSING
    """Coefficient for value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """Policy clip parameter."""


__all__ = ["RslRlPpoAlgorithmCfg"]

