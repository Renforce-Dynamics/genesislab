"""Go2 velocity tracking environment implementation."""

from __future__ import annotations

import torch

from genesislab.envs import ManagerBasedGenesisEnv
from .go2_flat_env_cfg import Go2VelocityEnvCfg


class Go2VelocityEnv(ManagerBasedGenesisEnv):
    """Go2 velocity tracking environment.

    This environment implements a velocity tracking task where the Go2 quadruped
    robot must track a forward velocity command while maintaining stability.

    The task uses:
    - PD control for joint position targets
    - Velocity command tracking rewards
    - Base height termination
    - Batched observations and rewards
    """

    def __init__(self, cfg: Go2VelocityEnvCfg, device: str = "cuda"):
        """Initialize the Go2 velocity tracking environment.

        Args:
            cfg: Go2 environment configuration.
            device: Device to use for tensors ('cuda' or 'cpu').
        """
        super().__init__(cfg=cfg, device=device)

        # Set PD gains for the robot
        self._set_pd_gains(cfg.pd_kp, cfg.pd_kd)

    def _set_pd_gains(self, kp: float, kd: float) -> None:
        """Set PD gains for all controlled joints.

        Args:
            kp: Position gain.
            kd: Velocity gain.
        """
        dof_indices = self._binding._dof_indices.get("go2")

        if dof_indices is not None:
            num_dofs = len(dof_indices)
        else:
            # Fallback: infer number of DOFs from the entity if indices are not specified.
            entity = self._binding.entities["go2"]
            dof_pos, _ = self._binding.get_joint_state("go2")
            num_dofs = dof_pos.shape[-1]

        # Set PD gains
        kp_tensor = torch.full((num_dofs,), kp, device=self.device)
        kd_tensor = torch.full((num_dofs,), kd, device=self.device)

        self._binding.set_pd_gains("go2", kp_tensor, kd_tensor)
