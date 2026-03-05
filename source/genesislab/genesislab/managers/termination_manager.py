"""Termination manager for computing done signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from genesislab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
  from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whether the term contributes towards episodic timeouts."""


class TerminationManager(ManagerBase):
  """Manages termination conditions for the environment.

  The termination manager aggregates multiple termination terms to compute
  episode done signals. Terms can be either truncations (time-based) or
  terminations (failure conditions).
  """

  _env: "ManagerBasedRlEnv"

  def __init__(self, cfg: dict[str, TerminationTermCfg], env: "ManagerBasedRlEnv"):
    self._term_names: list[str] = list()
    self._term_cfgs: list[TerminationTermCfg] = list()
    self._class_term_cfgs: list[TerminationTermCfg] = list()

    self.cfg = deepcopy(cfg)
    super().__init__(env)

    self._term_dones = dict()
    for term_name in self._term_names:
      self._term_dones[term_name] = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.bool
      )
    self._truncated_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._terminated_buf = torch.zeros_like(self._truncated_buf)

  def __str__(self) -> str:
    msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Termination Terms"
    table.field_names = ["Index", "Name", "Time Out"]
    table.align["Name"] = "l"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.time_out])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  @property
  def dones(self) -> torch.Tensor:
    return self._truncated_buf | self._terminated_buf

  @property
  def time_outs(self) -> torch.Tensor:
    return self._truncated_buf

  @property
  def terminated(self) -> torch.Tensor:
    return self._terminated_buf

  # Methods.

  def reset(
    self, env_ids: torch.Tensor | slice = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    extras = {}
    for key in self._term_dones.keys():
      extras["Episode_Termination/" + key] = torch.count_nonzero(
        self._term_dones[key][env_ids]
      ).item()
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self) -> torch.Tensor:
    self._truncated_buf[:] = False
    self._terminated_buf[:] = False
    for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
      value = term_cfg.func(self._env, **term_cfg.params)
      if term_cfg.time_out:
        self._truncated_buf |= value
      else:
        self._terminated_buf |= value
      self._term_dones[name][:] = value
    return self._truncated_buf | self._terminated_buf

  def get_term(self, name: str) -> torch.Tensor:
    return self._term_dones[name]

  def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for key in self._term_dones.keys():
      terms.append((key, [self._term_dones[key][env_idx].float().cpu().item()]))
    return terms

  def _prepare_terms(self):
    # Extract terms from configclass __dict__ or dict items
    if isinstance(self.cfg, dict):
      term_cfg_items = self.cfg.items()
    else:
      term_cfg_items = self.cfg.__dict__.items()

    for term_name, term_cfg in term_cfg_items:
      # Skip private fields
      if term_name.startswith("_"):
        continue
      term_cfg: TerminationTermCfg
      if term_cfg is None or term_cfg is MISSING:
        print(f"term: {term_name} set to None/MISSING, skipping...")
        continue
      if not isinstance(term_cfg, TerminationTermCfg):
        raise TypeError(
          f"Configuration for the term '{term_name}' is not of type 'TerminationTermCfg'."
          f" Received: '{type(term_cfg)}'."
        )
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
