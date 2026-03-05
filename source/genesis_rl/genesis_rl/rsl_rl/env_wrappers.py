"""Environment wrappers to connect GenesisLab envs to ``rsl_rl`` runners.

This module mirrors the role of IsaacLab's ``isaaclab_rl.rsl_rl`` env wrappers,
but is implemented for GenesisLab's manager-based RL environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv
from genesislab.envs.common import VecEnvObs


@dataclass
class ObsGroupMapping:
    """Mapping from rsl_rl obs sets to Genesis observation groups.

    In rsl_rl, the runner configuration uses an ``obs_groups`` dictionary to
    decide which observation groups feed which models (actor, critic, etc.).

    This small dataclass mirrors that idea on the environment side so that
    ``GenesisRslRlVecEnv`` can assemble a TensorDict in the expected format.
    """

    actor: Sequence[str] = ("policy",)
    critic: Sequence[str] = ("policy",)


class GenesisRslRlVecEnv(VecEnv):
    """Adapter from :class:`ManagerBasedRlEnv` to :class:`rsl_rl.env.VecEnv`.

    This wrapper:
    - Calls into a :class:`ManagerBasedRlEnv` instance for reset/step.
    - Converts Genesis-style observations into a :class:`TensorDict`.
    - Exposes ``num_envs``, ``num_actions``, ``max_episode_length``,
      ``episode_length_buf``, ``device`` and ``cfg`` as expected by ``rsl_rl``.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        obs_mapping: ObsGroupMapping | None = None,
    ) -> None:
        self._env: ManagerBasedRlEnv = env
        self.obs_mapping: ObsGroupMapping = obs_mapping or ObsGroupMapping()

        # Required VecEnv attributes
        self.num_envs: int = env.num_envs
        self.device = torch.device(env.device)

        # Action / episode meta
        self.num_actions: int = env.action_manager.total_action_dim
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf

        # Keep original cfg object around so rsl_rl's logger can access it.
        self.cfg = env.cfg

        # Cache for latest observations (Genesis-style dict)
        self._last_obs: VecEnvObs | None = None

        # Run an initial reset so that buffers are populated.
        self.reset()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _build_tensordict(self, obs_dict: Mapping[str, torch.Tensor]) -> TensorDict:
        """Convert a flat observation dict into a TensorDict for rsl_rl.

        For now we keep the structure intentionally simple:

        - We assume the environment exposes a *concatenated* observation tensor
          for the policy group under the key ``"policy"``.
        - We then create two views:
          - ``"actor"``: concatenated tensor used by the policy network.
          - ``"critic"``: same tensor for the value network.

        This matches the common configuration where both actor and critic share
        the same input.

        If you wish to expose multiple groups, you can extend this function and
        the ``ObsGroupMapping`` dataclass accordingly.
        """
        if "policy" not in obs_dict:
            raise KeyError(
                "Expected observation group 'policy' in ManagerBasedRlEnv observations. "
                "Please ensure your ObservationManager produces a concatenated 'policy' tensor."
            )

        policy_obs = obs_dict["policy"]
        if policy_obs.ndim != 2 or policy_obs.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected 'policy' obs of shape (num_envs, dim), got {tuple(policy_obs.shape)}"
            )

        # rsl_rl expects a TensorDict with batch_size = (num_envs,).
        # We keep a minimal structure: a single tensor for each obs set.
        data = {
            "actor": policy_obs,
            "critic": policy_obs,
        }
        return TensorDict(data, batch_size=[self.num_envs])

    # --------------------------------------------------------------------- #
    # VecEnv API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: Dict[str, Any] | None = None,
    ):
        """Reset the underlying Genesis environment and return initial observations.

        This mirrors the signature of Gymnasium's vector-env reset, but forwards
        the call directly to :class:`ManagerBasedRlEnv`.
        """
        obs_dict, info = self._env.reset(seed=seed, env_ids=env_ids, options=options)
        self._last_obs = obs_dict
        td_obs = self.get_observations()
        return td_obs, info

    def get_observations(self) -> TensorDict:
        """Return the current observations as a ``TensorDict``."""
        if self._last_obs is None:
            # Should not normally happen because we reset in __init__,
            # but keep a safe fallback.
            obs_dict, _ = self._env.reset()
            self._last_obs = obs_dict
        # ``_last_obs`` is a Genesis-style VecEnvObs (dict of tensors).
        flat_obs: Dict[str, torch.Tensor] = {}
        for k, v in self._last_obs.items():
            if isinstance(v, torch.Tensor):
                flat_obs[k] = v
        return self._build_tensordict(flat_obs)

    def step(self, actions: torch.Tensor):
        """Step the underlying Genesis environment.

        Returns:
            (observations, rewards, dones, extras) in the format expected
            by :class:`rsl_rl.runners.OnPolicyRunner`.
        """
        obs, rewards, terminated, time_outs, info = self._env.step(actions)

        # Cache the latest observations (Genesis-style) for get_observations().
        self._last_obs = obs

        # Combine terminated + timeouts into a single done flag.
        dones = terminated | time_outs

        # rsl_rl uses "time_outs" inside extras for logging / masking.
        extras = dict(info)
        extras.setdefault("time_outs", time_outs)

        td_obs = self.get_observations()
        return td_obs, rewards, dones, extras

    # The rsl_rl runners access ``env.device`` directly.
    @property
    def device_type(self) -> torch.device:
        return self.device

