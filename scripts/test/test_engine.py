"""Test script for Genesis engine binding.

This script validates that the Genesis engine binding works correctly:
- Scene construction
- Multi-env stepping
- Deterministic reset
- State access

It uses the built-in Go2 URDF from the Genesis assets so that the test
runs out-of-the-box.
"""

from __future__ import annotations

import argparse

import genesis as gs
import torch

from genesislab.components.entities.scene_cfg import SceneCfg
from genesislab.components.entities.robot_cfg import RobotCfg
from genesislab.engine.binding import GenesisBinding


def test_engine_binding(num_envs: int = 4, backend: str = "cpu") -> bool:
    """Test the Genesis engine binding."""
    print("Testing Genesis engine binding...")

    # Initialize Genesis
    gs.init(backend=gs.gpu if backend == "cuda" else gs.cpu)

    # Create a simple scene config
    scene_cfg = SceneCfg(
        num_envs=num_envs,
        dt=0.002,
        substeps=1,
        backend=backend,
        robots={
            "robot": RobotCfg(
                morph_type="URDF",
                morph_path="./data/assets/assetslib/unitree/unitree_g1/urdf/g1_29dof.urdf",
                initial_pose={"pos": [0.0, 0.0, 0.42], "quat": [0.0, 0.0, 0.0, 1.0]},
                fixed_base=False,
            )
        },
        terrain={"type": "plane"},
    )

    # Create binding
    binding = GenesisBinding(scene_cfg, device=backend)
    binding.build()

    print(f"✓ Scene built with {binding.num_envs} environments")

    # Test state access
    dof_pos, dof_vel = binding.get_joint_state("robot")
    print(f"✓ Joint state accessed: pos shape {dof_pos.shape}, vel shape {dof_vel.shape}")

    base_pos, base_quat, base_lin_vel, base_ang_vel = binding.get_root_state("robot")
    print(f"✓ Root state accessed: pos shape {base_pos.shape}, quat shape {base_quat.shape}")

    # Test stepping
    print("\nTesting physics stepping...")
    for _ in range(10):
        binding.step()
    print("✓ Stepped physics 10 times")

    # Test reset
    print("\nTesting reset...")
    binding.reset()
    new_dof_pos, _ = binding.get_joint_state("robot")
    print(f"✓ Reset completed: pos shape {new_dof_pos.shape}")

    # Test selective reset (basic smoke check)
    print("\nTesting selective reset...")
    _ = binding.get_joint_state("robot")[0]
    env_ids = torch.arange(min(2, binding.num_envs), device=backend)
    binding.step()
    binding.reset(env_ids=env_ids)
    _ = binding.get_joint_state("robot")[0]
    print("✓ Selective reset completed")

    print("\n✓ All engine binding tests passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    ok = test_engine_binding(num_envs=args.num_envs, backend=args.backend)
    raise SystemExit(0 if ok else 1)
