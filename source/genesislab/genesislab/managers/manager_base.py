from __future__ import annotations

import abc
import inspect
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from genesislab.managers.scene_entity_config import SceneEntityCfg
from genesislab.utils.configclass import configclass
from genesislab.utils.imports import resolve_callable

if TYPE_CHECKING:
  from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


@configclass
class ManagerTermBaseCfg:
  """Base configuration for manager terms.

  This is the base config for terms in observation, reward, termination, curriculum,
  and event managers. It provides a common interface for specifying a callable
  and its parameters.

  The ``func`` field accepts either a function or a class:

  **Function-based terms** are simpler and suitable for stateless computations:

  .. code-block:: python

      RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.01)

  **Class-based terms** are instantiated with ``(cfg, env)`` and useful when you need
  to:

  - Cache computed values at initialization (e.g., resolve regex patterns to indices)
  - Maintain state across calls
  - Perform expensive setup once rather than every call

  .. code-block:: python

      class posture:
        def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
          # Resolve std dict to tensor once at init
          self.std = resolve_std_to_tensor(cfg.params["std"], env)

        def __call__(self, env, **kwargs) -> torch.Tensor:
          # Use cached self.std
          return compute_posture_reward(env, self.std)

      RewardTermCfg(func=posture, params={"std": {".*knee.*": 0.3}}, weight=1.0)

  Class-based terms can optionally implement ``reset(env_ids)`` for per-episode state.
  """

  # Required callable: annotated and given a member value using MISSING to
  # indicate "no default" for configclass' mutable-type processing.
  func: Any = MISSING
  """The callable that computes this term's value. Can be a function or a class.
  Classes are auto-instantiated with ``(cfg=term_cfg, env=env)``."""

  params: dict[str, Any] = {}
  """Additional keyword arguments passed to func when called."""


class ManagerTermBase:
  def __init__(self, env: "ManagerBasedRlEnv"):
    self._env = env

  # Properties.

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  @property
  def device(self) -> str:
    return self._env.device

  @property
  def name(self) -> str:
    return self.__class__.__name__

  # Methods.

  def reset(self, env_ids: torch.Tensor | slice) -> Any:
    """Resets the manager term."""
    del env_ids  # Unused.
    pass

  def __call__(self, *args, **kwargs) -> Any:
    """Returns the value of the term required by the manager."""
    raise NotImplementedError


class ManagerBase(abc.ABC):
  """Base class for all managers."""

  def __init__(self, env: "ManagerBasedRlEnv"):
    self._env = env

    self._prepare_terms()

  # Properties.

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  @property
  def device(self) -> str:
    return self._env.device

  @property
  @abc.abstractmethod
  def active_terms(self) -> list[str] | dict[Any, list[str]]:
    raise NotImplementedError

  # Methods.

  def reset(self, env_ids: torch.Tensor) -> dict[str, Any]:
    """Resets the manager and returns logging info for the current step."""
    del env_ids  # Unused.
    return {}

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def _prepare_terms(self):
    raise NotImplementedError

  def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg) -> None:
    """Resolve common term configuration fields.

    Currently this performs two tasks:

    1. Resolve any :class:`SceneEntityCfg` instances in ``term_cfg.params`` by
       calling their :meth:`resolve` method against the environment's binding
       (preferred) or scene.
    2. Resolve ``term_cfg.func`` into a callable or class if it is specified
       as a string import path.
    3. If ``term_cfg.func`` is a class, instantiate it with ``(cfg, env)`` so
       that managers uniformly call callables.
    """
    del term_name  # Unused naming hook for future logging.

    # Resolve entity config references.
    for value in term_cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        # Prefer resolving against the binding's entities when available.
        if hasattr(self._env, "_binding") and hasattr(self._env._binding, "entities"):
          value.resolve(self._env._binding.entities)
        else:
          # Fallback: resolve against the scene object.
          value.resolve(self._env.scene)

    # Resolve func when provided as a string path.
    term_cfg.func = resolve_callable(term_cfg.func)

    # Instantiate class-based term implementations once at manager construction.
    if inspect.isclass(term_cfg.func):
      term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

