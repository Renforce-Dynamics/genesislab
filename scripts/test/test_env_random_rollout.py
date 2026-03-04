"""Test script for random policy rollout.

This script validates that a GenesisLab environment can:
- Be created from config
- Step with random actions
- Reset correctly
- Compute observations and rewards
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genesislab.configs.env_cfg import ManagerBasedGenesisEnvCfg
from genesislab.configs.robot_cfg import RobotCfg
from genesislab.configs.scene_cfg import SceneCfg
from genesislab.envs import ManagerBasedGenesisEnv
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
    from genesislab.envs.base_env import ManagerBasedGenesisEnv as _MBEnv


@dataclass(kw_only=True)
class DemoActionTermCfg(ActionTermCfg):
    """Simple demo action term config used by the random rollout test."""

    def build(self, env: "_MBEnv") -> "DemoActionTerm":
        return DemoActionTerm(cfg=self, env=env)


class DemoActionTerm(ActionTerm):
    """Minimal action term that directly maps actions to joint position targets."""

    def __init__(self, cfg: DemoActionTermCfg, env: "_MBEnv"):
        super().__init__(cfg, env)
        dof_pos, _ = env._binding.get_joint_state(cfg.entity_name)
        self._action_dim = dof_pos.shape[-1]
        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_action.shape:
            if actions.shape[-1] == self._action_dim and actions.shape[0] == 1:
                actions = actions.expand_as(self._raw_action)
            else:
                raise ValueError(
                    f"Invalid action shape for DemoActionTerm: expected "
                    f"{self._raw_action.shape}, got {actions.shape}."
                )
        self._raw_action[:] = actions
        self._targets[:] = actions

    def apply_actions(self) -> None:
        self._env._binding.set_joint_targets(
            self.cfg.entity_name,
            self._targets,
            control_type="position",
        )


def test_random_rollout():
    """Test random policy rollout."""
    print("Testing random policy rollout...")

    # Create a minimal environment config using the Go2 asset
    scene_cfg = SceneCfg(
        num_envs=4,
        dt=0.002,
        substeps=1,
        backend="cuda" if torch.cuda.is_available() else "cpu",
        robots={
            "go2": RobotCfg(
                morph_type="MJCF",
                morph_path="assetslib/unitree/unitree_go2/mjcf/go2.xml",
                initial_pose={"pos": [0.0, 0.0, 0.5], "quat": [0.0, 0.0, 0.0, 1.0]},
                fixed_base=False,
                control_dofs=None,  # Control all DOFs
            )
        },
        terrain={"type": "plane"},
    )

    env_cfg = ManagerBasedGenesisEnvCfg(
        decimation=10,
        scene=scene_cfg,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "dof_pos": ObservationTermCfg(
                        func=lambda env: env._binding.get_joint_state("go2")[0],
                    ),
                },
                concatenate=True,
            )
        },
        actions={
            "go2": DemoActionTermCfg(
                entity_name="go2",
            )
        },
        rewards={
            "survival": RewardTermCfg(
                func=lambda env: torch.ones(env.num_envs, device=env.device),
                weight=1.0,
            )
        },
        terminations={
            "fall": TerminationTermCfg(
                func=lambda env: env._binding.get_root_state("robot")[0][:, 2] < 0.1,
                time_out=False,
            )
        },
        episode_length_s=5.0,
        is_finite_horizon=False,
    )

    # Create environment
    print("Creating environment...")
    env = ManagerBasedGenesisEnv(cfg=env_cfg, device="cuda")
    print(f"✓ Environment created with {env.num_envs} environments")

    # Reset
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset completed: obs keys {list(obs.keys())}")

    # Random rollout
    print("\nRunning random policy rollout...")
    num_steps = 100
    total_reward = torch.zeros(env.num_envs, device=env.device)

    for step in range(num_steps):
        # Random actions
        action = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        action = torch.clamp(action, -1.0, 1.0)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        total_reward += reward

        # Print progress
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}: avg reward = {total_reward.mean().item():.4f}")

        # Reset if needed
        if terminated.any() or truncated.any():
            reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            print(f"  Resetting {len(reset_envs)} environments at step {step + 1}")
            obs, info = env.reset(env_ids=reset_envs)

    print(f"\n✓ Rollout completed: final avg reward = {total_reward.mean().item():.4f}")
    print("✓ All random rollout tests passed!")


if __name__ == "__main__":
    test_random_rollout()
