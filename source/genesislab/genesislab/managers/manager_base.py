from __future__ import annotations

import abc
import copy
import inspect
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from genesislab.components.entities.scene_entity_cfg import SceneEntityCfg
from genesislab.managers.manager_term_cfg import ManagerTermBaseCfg
from genesislab.utils.imports import resolve_callable

if TYPE_CHECKING:
	from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ManagerTermBase:
	"""Base class for manager terms.

	Manager term implementations can be functions or classes. If the term is a class, it should
	inherit from this base class and implement the required methods.
	"""

	def __init__(self, cfg: ManagerTermBaseCfg | None = None, env: "ManagerBasedRlEnv" = None):
		"""Initialize the manager term.

		Args:
			cfg: The configuration object. Optional for backward compatibility.
			env: The environment instance.
		"""
		# Store the inputs
		self.cfg = cfg
		self._env = env

	# Properties.

	@property
	def num_envs(self) -> int:
		"""Number of environments."""
		return self._env.num_envs

	@property
	def device(self) -> str:
		"""Device on which to perform computations."""
		return self._env.device

	@property
	def __name__(self) -> str:
		"""Return the name of the class or subclass."""
		return self.__class__.__name__

	@property
	def name(self) -> str:
		"""Return the name of the class or subclass."""
		return self.__class__.__name__

	# Methods.

	def reset(self, env_ids: torch.Tensor | slice | Sequence[int] | None = None) -> None:
		"""Resets the manager term.

		Args:
			env_ids: The environment ids. Defaults to None, in which case
				all environments are considered.
		"""
		pass

	def serialize(self) -> dict:
		"""General serialization call. Includes the configuration dict."""
		if self.cfg is None:
			return {}
		# Simple serialization - can be extended
		return {"cfg": str(self.cfg)}

	def __call__(self, *args, **kwargs) -> Any:
		"""Returns the value of the term required by the manager.

		In case of a class implementation, this function is called by the manager
		to get the value of the term. The arguments passed to this function are
		the ones specified in the term configuration (see :attr:`ManagerTermBaseCfg.params`).

		Args:
			*args: Variable length argument list.
			**kwargs: Arbitrary keyword arguments.

		Returns:
			The value of the term.
		"""
		raise NotImplementedError("The method '__call__' should be implemented by the subclass.")


class ManagerBase(abc.ABC):
	"""Base class for all managers."""

	def __init__(self, cfg: object | None = None, env: "ManagerBasedRlEnv" = None):
		"""Initialize the manager.

		This function is responsible for parsing the configuration object and creating the terms.

		Args:
			cfg: The configuration object. If None, the manager is initialized without any terms.
			env: The environment instance.
		"""
		# Store the inputs
		self.cfg = copy.deepcopy(cfg) if cfg is not None else None
		self._env = env

		# Parse config to create terms information
		if self.cfg:
			self._prepare_terms()

	# Properties.

	@property
	def num_envs(self) -> int:
		"""Number of environments."""
		return self._env.num_envs

	@property
	def device(self) -> str:
		"""Device on which to perform computations."""
		return self._env.device

	@property
	@abc.abstractmethod
	def active_terms(self) -> list[str] | dict[str, list[str]]:
		"""Name of active terms."""
		raise NotImplementedError

	# Methods.

	def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> dict[str, float]:
		"""Resets the manager and returns logging information for the current time-step.

		Args:
			env_ids: The environment ids for which to log data.
				Defaults None, which logs data for all environments.

		Returns:
			Dictionary containing the logging information.
		"""
		return {}

	def find_terms(self, name_keys: str | Sequence[str]) -> list[str]:
		"""Find terms in the manager based on the names.

		This function searches the manager for terms based on the names. The names can be
		specified as regular expressions or a list of regular expressions. The search is
		performed on the active terms in the manager.

		Args:
			name_keys: A regular expression or a list of regular expressions to match the term names.

		Returns:
			A list of term names that match the input keys.
		"""
		# Resolve search keys
		if isinstance(self.active_terms, dict):
			list_of_strings = []
			for names in self.active_terms.values():
				list_of_strings.extend(names)
		else:
			list_of_strings = self.active_terms

		# Convert name_keys to list if it's a string
		if isinstance(name_keys, str):
			name_keys = [name_keys]

		# Simple regex matching implementation
		matched_names = []
		for pattern in name_keys:
			regex = re.compile(pattern)
			for name in list_of_strings:
				if regex.search(name) and name not in matched_names:
					matched_names.append(name)

		return matched_names

	def get_active_iterable_terms(
		self, env_idx: int
	) -> Sequence[tuple[str, Sequence[float]]]:
		"""Returns the active terms as iterable sequence of tuples.

		The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

		Returns:
			The active terms.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def _prepare_terms(self):
		"""Prepare terms information from the configuration object."""
		raise NotImplementedError

	def _resolve_common_term_cfg(
		self, term_name: str, term_cfg: ManagerTermBaseCfg, min_argc: int = 1
	) -> None:
		"""Resolve common attributes of the term configuration.

		Usually, called by the :meth:`_prepare_terms` method to resolve common attributes of the term
		configuration. These include:

		* Resolving the term function and checking if it is callable.
		* Resolving special attributes of the term configuration like ``entity_name``, etc.
		* Initializing the term if it is a class.

		By default, all term functions are expected to have at least one argument, which is the
		environment object. Some other managers may expect functions to take more arguments, for
		instance, the environment indices as the second argument. In such cases, the
		``min_argc`` argument can be used to specify the minimum number of arguments
		required by the term function to be called correctly by the manager.

		Args:
			term_name: The name of the term.
			term_cfg: The term configuration.
			min_argc: The minimum number of arguments required by the term function to be called correctly
				by the manager.
		"""
		del term_name  # Unused naming hook for future logging.

		# Check if the term is a valid term config
		if not isinstance(term_cfg, ManagerTermBaseCfg):
			raise TypeError(
				f"Configuration for the term '{term_name}' is not of type ManagerTermBaseCfg."
				f" Received: '{type(term_cfg)}'."
			)

		# Resolve entity config references.
		for key, value in term_cfg.params.items():
			if isinstance(value, SceneEntityCfg):
				# Resolve against the binding's entities - required
				if not hasattr(self._env, "_binding"):
					raise AttributeError(
						"Environment does not have '_binding' attribute. "
						"ManagerBase requires GenesisBinding to resolve SceneEntityCfg."
					)
				
				if not hasattr(self._env._binding, "entities"):
					raise AttributeError(
						"Binding does not have 'entities' attribute. "
						"Binding may not be properly initialized."
					)
				
				value.resolve(self._env._binding.entities)

		# Get the corresponding function or functional class
		if isinstance(term_cfg.func, str):
			term_cfg.func = resolve_callable(term_cfg.func)

		# Check if function is callable
		if not callable(term_cfg.func):
			raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")

		# Check if the term is a class of valid type
		if inspect.isclass(term_cfg.func):
			if not issubclass(term_cfg.func, ManagerTermBase):
				raise TypeError(
					f"Configuration for the term '{term_name}' is not of type ManagerTermBase."
					f" Received: '{type(term_cfg.func)}'."
				)
			# Initialize the term if it is a class
			term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

