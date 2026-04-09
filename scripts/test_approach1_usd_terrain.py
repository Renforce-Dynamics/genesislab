#!/usr/bin/env python3
"""Test Approach 1: USD as Terrain (Static Geometry)

This demonstrates using a static USD file as terrain through the terrain system.
The robot walks on the USD terrain (Floor, Wall, Ceiling from Scene.usd).

Key features:
- Uses TerrainCfg with terrain_type="usd"
- Multi-environment grid layout support
- Static USD only (no articulated objects)
- Integrated with terrain system

Usage:
    # Headless mode (default)
    python scripts/test_approach1_usd_terrain.py

    # With viewer
    python scripts/test_approach1_usd_terrain.py --viewer
"""

import argparse
import genesis as gs
from genesislab import LabScene
from genesislab.engine.scene import SceneCfg
from genesislab.components.terrains import TerrainCfg
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Approach 1: USD Terrain")
    parser.add_argument("--viewer", action="store_true", help="Enable viewer")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("测试方案1: USD作为Terrain（静态地形）")
    print("=" * 80)

    # Initialize Genesis
    gs.init(backend=gs.cuda, precision="32", logging_level="warning")

    # Create configuration with USD terrain
    cfg = SceneCfg(
        num_envs=1,  # Single environment for clear visualization
        viewer=args.viewer,  # Viewer mode from args
        backend="cuda",

        # ⭐ 方案1: USD作为Terrain
        terrain=TerrainCfg(
            terrain_type="usd",  # Use USD terrain type
            usd_path="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Terrain.usd",
            env_spacing=10.0,
        ),

        # Add G1 humanoid robot
        robots={
            "humanoid": G1_FULL_ACT_CFG,
        },
    )

    print(f"\n配置信息:")
    print(f"  - 方案: USD Terrain (静态地形)")
    print(f"  - USD文件: Terrain.usd")
    print(f"  - 环境数量: {cfg.num_envs}")
    print(f"  - 机器人: G1 Humanoid")
    print(f"  - 包含: Floor, Wall, Ceiling (静态)")
    print(f"  - Viewer: {'✅ 启用' if args.viewer else '❌ 无头模式'}")

    # Build scene
    print(f"\n[1/3] 构建场景...")
    scene = LabScene(cfg)
    scene.build()
    print("✅ 场景构建完成")

    # Reset
    print(f"\n[2/3] 重置环境...")
    scene.reset()
    print("✅ 环境已重置")

    # Run simulation
    print(f"\n[3/3] 运行仿真...")
    print("提示: 机器人将站在USD地形上")
    print("     运行200步进行测试\n")

    max_steps = 20000
    for step_count in range(max_steps):
        # Step simulation
        scene.step()

        # Print status every 50 steps
        if (step_count + 1) % 50 == 0:
            print(f"Step {step_count + 1:06d}/{max_steps} - 仿真运行中...")

    print(f"\n仿真完成（总步数: {max_steps}）")

    print("\n" + "=" * 80)
    print("✅ 方案1测试完成！")
    print("   - USD地形成功加载")
    print("   - 机器人成功生成在地形上")
    print("=" * 80)


if __name__ == "__main__":
    main()
