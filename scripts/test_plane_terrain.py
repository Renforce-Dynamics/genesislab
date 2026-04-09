#!/usr/bin/env python3
"""Test Baseline: Plane Terrain (Traditional Flat Ground)

This demonstrates the traditional plane terrain approach for comparison.

Key features:
- Uses TerrainCfg with terrain_type="plane" (default)
- Infinite flat ground
- Fastest initialization
- Best for basic locomotion training
"""

import genesis as gs
from genesislab import LabScene
from genesislab.engine.scene import SceneCfg
from genesislab.components.terrains import TerrainCfg
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG


def main():
    print("\n" + "=" * 80)
    print("测试基线: Plane Terrain（传统平地）")
    print("=" * 80)

    # Initialize Genesis
    gs.init(backend=gs.cuda, precision="32", logging_level="warning")

    # Create configuration with plane terrain
    cfg = SceneCfg(
        num_envs=1,
        viewer=False,  # Headless mode
        backend="cuda",

        # 基线方案: Plane Terrain（无限平面）
        terrain=TerrainCfg(terrain_type="plane"),

        # Add G1 humanoid robot
        robots={
            "humanoid": G1_FULL_ACT_CFG,
        },
    )

    print(f"\n配置信息:")
    print(f"  - 方案: Plane Terrain (传统平地)")
    print(f"  - 地形: 无限平面")
    print(f"  - 环境数量: {cfg.num_envs}")
    print(f"  - 机器人: G1 Humanoid")

    # Build scene
    print(f"\n[1/3] 构建场景...")
    scene = LabScene(cfg)
    scene.build()
    print("✅ 场景构建完成（极快！）")

    # Reset
    print(f"\n[2/3] 重置环境...")
    scene.reset()
    print("✅ 环境已重置")

    # Run simulation
    print(f"\n[3/3] 运行仿真...")
    print("提示: 机器人站在无限平面上")
    print("     运行200步进行测试\n")

    max_steps = 200
    for step_count in range(max_steps):
        # Step simulation
        scene.step()

        # Print status every 50 steps
        if (step_count + 1) % 50 == 0:
            print(f"Step {step_count + 1:06d}/{max_steps} - 仿真运行中...")

    print(f"\n仿真完成（总步数: {max_steps}）")

    print("\n" + "=" * 80)
    print("✅ Plane Terrain测试完成！")
    print("   - 平面地形（baseline）")
    print("   - 初始化速度最快")
    print("   - 适合基础运动训练")
    print("=" * 80)


if __name__ == "__main__":
    main()
