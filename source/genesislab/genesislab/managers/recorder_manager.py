"""Recorder manager for recording data produced from the given world."""

from __future__ import annotations

import enum
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from genesislab.managers.manager_base import ManagerBase, ManagerTermBase
from genesislab.managers.manager_term_cfg import RecorderTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
	from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


class DatasetExportMode(enum.IntEnum):
	"""The mode to handle episode exports."""

	EXPORT_NONE = 0  # Export none of the episodes
	EXPORT_ALL = 1  # Export all episodes to a single dataset file
	EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES = 2  # Export succeeded and failed episodes in separate files
	EXPORT_SUCCEEDED_ONLY = 3  # Export only succeeded episodes to a single dataset file


@configclass
class RecorderManagerBaseCfg:
	"""Base class for configuring recorder manager terms."""

	dataset_file_handler_class_type: type = None
	"""The dataset file handler class type. Can be set to a custom handler class."""

	dataset_export_dir_path: str = "/tmp/genesislab/logs"
	"""The directory path where the recorded datasets are exported."""

	dataset_filename: str = "dataset"
	"""Dataset file name without file extension."""

	dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
	"""The mode to handle episode exports."""

	export_in_record_pre_reset: bool = True
	"""Whether to export episodes in the record_pre_reset call."""

	export_in_close: bool = False
	"""Whether to export episodes in the close call."""


class RecorderTerm(ManagerTermBase):
	"""Base class for recorder terms.

	The recorder term is responsible for recording data at various stages of the environment's lifecycle.
	A recorder term is comprised of four user-defined callbacks to record data in the corresponding stages:

	* Pre-reset recording: This callback is invoked at the beginning of `env.reset()` before the reset is effective.
	* Post-reset recording: This callback is invoked at the end of `env.reset()`.
	* Pre-step recording: This callback is invoked at the beginning of `env.step()`, after the step action is processed
		  and before the action is applied by the action manager.
	* Post-step recording: This callback is invoked at the end of `env.step()` when all the managers are processed.
	"""

	def __init__(self, cfg: RecorderTermCfg, env: "ManagerBasedRlEnv"):
		"""Initialize the recorder term.

		Args:
			cfg: The configuration object.
			env: The environment instance.
		"""
		# call the base class constructor
		super().__init__(cfg, env)

	"""
	User-defined callbacks.
	"""

	def record_pre_reset(
		self, env_ids: Sequence[int] | None
	) -> tuple[str | None, torch.Tensor | dict | None]:
		"""Record data at the beginning of env.reset() before reset is effective.

		Args:
			env_ids: The environment ids. All environments should be considered when set to None.

		Returns:
			A tuple of key and value to be recorded.
			The key can contain nested keys separated by '/'. For example, "obs/joint_pos" would add the given
			value under ['obs']['policy'] in the underlying dictionary in the recorded episode data.
			The value can be a tensor or a nested dictionary of tensors. The shape of a tensor in the value
			is (env_ids, ...).
		"""
		return None, None

	def record_post_reset(
		self, env_ids: Sequence[int] | None
	) -> tuple[str | None, torch.Tensor | dict | None]:
		"""Record data at the end of env.reset().

		Args:
			env_ids: The environment ids. All environments should be considered when set to None.

		Returns:
			A tuple of key and value to be recorded.
			Please refer to the `record_pre_reset` function for more details.
		"""
		return None, None

	def record_pre_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
		"""Record data in the beginning of env.step() after action is cached/processed in the ActionManager.

		Returns:
			A tuple of key and value to be recorded.
			Please refer to the `record_pre_reset` function for more details.
		"""
		return None, None

	def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
		"""Record data at the end of env.step() when all the managers are processed.

		Returns:
			A tuple of key and value to be recorded.
			Please refer to the `record_pre_reset` function for more details.
		"""
		return None, None

	def record_post_physics_decimation_step(
		self,
	) -> tuple[str | None, torch.Tensor | dict | None]:
		"""Record data after the physics step is executed in the decimation loop.

		Returns:
			A tuple of key and value to be recorded.
			Please refer to the `record_pre_reset` function for more details.
		"""
		return None, None

	def close(self, file_path: str):
		"""Finalize and "clean up" the recorder term.

		This can include tasks such as appending metadata (e.g. labels) to a file
		and properly closing any associated file handles or resources.

		Args:
			file_path: the absolute path to the file
		"""
		pass


class RecorderManager(ManagerBase):
	"""Manager for recording data from recorder terms."""

	def __init__(self, cfg: RecorderManagerBaseCfg | dict[str, RecorderTermCfg] | None, env: "ManagerBasedRlEnv"):
		"""Initialize the recorder manager.

		Args:
			cfg: The configuration object or dictionary (``dict[str, RecorderTermCfg]``).
			env: The environment instance.
		"""
		self._term_names: list[str] = list()
		self._terms: dict[str, RecorderTerm] = dict()

		# Do nothing if cfg is None or an empty dict
		if not cfg:
			super().__init__(cfg=None, env=env)
			return

		super().__init__(cfg=cfg, env=env)

		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		# Check if cfg is RecorderManagerBaseCfg (for base config) or dict (for terms)
		if isinstance(cfg, RecorderManagerBaseCfg):
			self._base_cfg = cfg
		else:
			# If it's a dict, create a default base config
			self._base_cfg = RecorderManagerBaseCfg()

		# Create episode data buffer indexed by environment id
		# Note: This is a simplified implementation. For full dataset export,
		# you would need to implement EpisodeData and dataset file handlers.
		self._episodes: dict[int, dict] = dict()
		for env_id in range(env.num_envs):
			self._episodes[env_id] = {}

		self._exported_successful_episode_count = {}
		self._exported_failed_episode_count = {}

	def __str__(self) -> str:
		"""Returns: A string representation for recorder manager."""
		msg = f"<RecorderManager> contains {len(self._term_names)} active terms.\n"
		# create table for term information
		table = PrettyTable()
		table.title = "Active Recorder Terms"
		table.field_names = ["Index", "Name"]
		# set alignment of table columns
		table.align["Name"] = "l"
		# add info on each term
		for index, name in enumerate(self._term_names):
			table.add_row([index, name])
		# convert table to string
		msg += table.get_string()
		msg += "\n"
		return msg

	def __del__(self):
		"""Destructor for recorder."""
		self.close()

	"""
	Properties.
	"""

	@property
	def active_terms(self) -> list[str]:
		"""Name of active recorder terms."""
		return self._term_names

	@property
	def exported_successful_episode_count(self, env_id=None) -> int:
		"""Number of successful episodes.

		Args:
			env_id: The environment id. Defaults to None, in which case all environments are considered.

		Returns:
			The number of successful episodes.
		"""
		if not hasattr(self, "_exported_successful_episode_count"):
			return 0
		if env_id is not None:
			return self._exported_successful_episode_count.get(env_id, 0)
		return sum(self._exported_successful_episode_count.values())

	@property
	def exported_failed_episode_count(self, env_id=None) -> int:
		"""Number of failed episodes.

		Args:
			env_id: The environment id. Defaults to None, in which case all environments are considered.

		Returns:
			The number of failed episodes.
		"""
		if not hasattr(self, "_exported_failed_episode_count"):
			return 0
		if env_id is not None:
			return self._exported_failed_episode_count.get(env_id, 0)
		return sum(self._exported_failed_episode_count.values())

	"""
	Operations.
	"""

	def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
		"""Resets the recorder data.

		Args:
			env_ids: The environment ids. Defaults to None, in which case
				all environments are considered.

		Returns:
			An empty dictionary.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return {}

		# resolve environment ids
		if env_ids is None:
			env_ids = list(range(self._env.num_envs))
		if isinstance(env_ids, torch.Tensor):
			env_ids = env_ids.tolist()

		for term in self._terms.values():
			term.reset(env_ids=env_ids)

		for env_id in env_ids:
			self._episodes[env_id] = {}

		# nothing to log here
		return {}

	def get_episode(self, env_id: int) -> dict:
		"""Returns the episode data for the given environment id.

		Args:
			env_id: The environment id.

		Returns:
			The episode data for the given environment id.
		"""
		return self._episodes.get(env_id, {})

	def add_to_episodes(
		self, key: str, value: torch.Tensor | dict, env_ids: Sequence[int] | None = None
	):
		"""Adds the given key-value pair to the episodes for the given environment ids.

		Args:
			key: The key of the given value to be added to the episodes. The key can contain nested keys
				separated by '/'. For example, "obs/joint_pos" would add the given value under ['obs']['policy']
				in the underlying dictionary in the episode data.
			value: The value to be added to the episodes. The value can be a tensor or a nested dictionary of tensors.
				The shape of a tensor in the value is (env_ids, ...).
			env_ids: The environment ids. Defaults to None, in which case all environments are considered.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		# resolve environment ids
		if key is None:
			return
		if env_ids is None:
			env_ids = list(range(self._env.num_envs))
		if isinstance(env_ids, torch.Tensor):
			env_ids = env_ids.tolist()

		if isinstance(value, dict):
			for sub_key, sub_value in value.items():
				self.add_to_episodes(f"{key}/{sub_key}", sub_value, env_ids)
			return

		for value_index, env_id in enumerate(env_ids):
			if env_id not in self._episodes:
				self._episodes[env_id] = {}
			# Simple dict storage - can be extended with proper EpisodeData structure
			if key not in self._episodes[env_id]:
				self._episodes[env_id][key] = []
			self._episodes[env_id][key].append(value[value_index].cpu() if isinstance(value, torch.Tensor) else value)

	def record_pre_step(self) -> None:
		"""Trigger recorder terms for pre-step functions."""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		for term in self._terms.values():
			key, value = term.record_pre_step()
			self.add_to_episodes(key, value)

	def record_post_step(self) -> None:
		"""Trigger recorder terms for post-step functions."""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		for term in self._terms.values():
			key, value = term.record_post_step()
			self.add_to_episodes(key, value)

	def record_post_physics_decimation_step(self) -> None:
		"""Trigger recorder terms for post-physics step functions in the decimation loop."""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		for term in self._terms.values():
			key, value = term.record_post_physics_decimation_step()
			self.add_to_episodes(key, value)

	def record_pre_reset(
		self, env_ids: Sequence[int] | None, force_export_or_skip=None
	) -> None:
		"""Trigger recorder terms for pre-reset functions.

		Args:
			env_ids: The environment ids in which a reset is triggered.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		if env_ids is None:
			env_ids = list(range(self._env.num_envs))
		if isinstance(env_ids, torch.Tensor):
			env_ids = env_ids.tolist()

		for term in self._terms.values():
			key, value = term.record_pre_reset(env_ids)
			self.add_to_episodes(key, value, env_ids)

		# Set task success values for the relevant episodes
		success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
		# Check success indicator from termination terms
		if hasattr(self._env, "termination_manager"):
			if "success" in self._env.termination_manager.active_terms:
				success_results |= self._env.termination_manager.get_term("success")[env_ids]
		# Store success in episodes (simplified)
		for i, env_id in enumerate(env_ids):
			if env_id not in self._episodes:
				self._episodes[env_id] = {}
			self._episodes[env_id]["success"] = success_results[i].item()

		if force_export_or_skip or (
			force_export_or_skip is None and self._base_cfg.export_in_record_pre_reset
		):
			self.export_episodes(env_ids)

	def record_post_reset(self, env_ids: Sequence[int] | None) -> None:
		"""Trigger recorder terms for post-reset functions.

		Args:
			env_ids: The environment ids in which a reset is triggered.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		for term in self._terms.values():
			key, value = term.record_post_reset(env_ids)
			self.add_to_episodes(key, value, env_ids)

	def export_episodes(
		self, env_ids: Sequence[int] | None = None, demo_ids: Sequence[int] | None = None
	) -> None:
		"""Concludes and exports the episodes for the given environment ids.

		Note: This is a simplified implementation. For full dataset export functionality,
		you would need to implement proper EpisodeData and dataset file handlers.

		Args:
			env_ids: The environment ids. Defaults to None, in which case
				all environments are considered.
			demo_ids: Custom identifiers for the exported episodes.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return

		if env_ids is None:
			env_ids = list(range(self._env.num_envs))
		if isinstance(env_ids, torch.Tensor):
			env_ids = env_ids.tolist()

		# Simplified export - just update counters
		# Full implementation would write to dataset files
		for env_id in env_ids:
			if env_id in self._episodes and self._episodes[env_id]:
				episode_succeeded = self._episodes[env_id].get("success", False)
				if episode_succeeded:
					self._exported_successful_episode_count[env_id] = (
						self._exported_successful_episode_count.get(env_id, 0) + 1
					)
				else:
					self._exported_failed_episode_count[env_id] = (
						self._exported_failed_episode_count.get(env_id, 0) + 1
					)
			# Reset the episode buffer for the given environment after export
			self._episodes[env_id] = {}

	def close(self):
		"""Closes the recorder manager by exporting any remaining data to file as well as properly
		closes the recorder terms.
		"""
		# Do nothing if no active recorder terms are provided
		if len(self.active_terms) == 0:
			return
		if hasattr(self, "_base_cfg") and self._base_cfg.export_in_close:
			self.export_episodes()
		for term in self._terms.values():
			file_path = os.path.join(
				self._base_cfg.dataset_export_dir_path, self._base_cfg.dataset_filename
			)
			term.close(file_path)

	"""
	Helper functions.
	"""

	def _prepare_terms(self):
		"""Prepares a list of recorder terms."""
		# check if config is dict already
		if isinstance(self.cfg, dict):
			cfg_items = self.cfg.items()
		else:
			cfg_items = self.cfg.__dict__.items()
		for term_name, term_cfg in cfg_items:
			# skip non-term settings (if cfg is RecorderManagerBaseCfg)
			if isinstance(self.cfg, RecorderManagerBaseCfg):
				if term_name in [
					"dataset_file_handler_class_type",
					"dataset_filename",
					"dataset_export_dir_path",
					"dataset_export_mode",
					"export_in_record_pre_reset",
					"export_in_close",
				]:
					continue
			# check if term config is None
			if term_cfg is None:
				continue
			# check valid type
			if not isinstance(term_cfg, RecorderTermCfg):
				raise TypeError(
					f"Configuration for the term '{term_name}' is not of type RecorderTermCfg."
					f" Received: '{type(term_cfg)}'."
				)
			# create the recorder term
			term = term_cfg.class_type(term_cfg, self._env)
			# sanity check if term is valid type
			if not isinstance(term, RecorderTerm):
				raise TypeError(
					f"Returned object for the term '{term_name}' is not of type RecorderTerm."
				)
			# add term name and parameters
			self._term_names.append(term_name)
			self._terms[term_name] = term
