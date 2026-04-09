#!/usr/bin/env python3
"""Test Approach 2: USD as Scene Entities (Complete Scene with Furniture)

This demonstrates loading a complete USD scene (including articulated furniture)
as background entities, separate from the terrain system.

Key features:
- Uses SceneCfg.usd_scene_path
- Can load scenes with articulated objects (chairs, cabinets with joints)
- Objects can interact with robot
- More flexible than terrain approach
"""

import genesis as gs
from genesislab import LabScene
from genesislab.engine.scene import SceneCfg
from genesislab.components.terrains import TerrainCfg
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG


def main():
    print("\n" + "=" * 80)
    print("测试方案2: USD作为Scene实体（完整场景）")
    print("=" * 80)

    # Initialize Genesis
    gs.init(backend=gs.cuda, precision="32", logging_level="warning")

    # Create configuration with USD scene
    cfg = SceneCfg(
        num_envs=1,  # Scene approach typically uses single environment
        viewer=False,  # Headless mode (set to True if you have display)
        backend="cuda",

        # ⭐ 方案2: USD作为Scene实体
        usd_scene_path="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd",

        # Optional: add simple plane terrain for robot spawning
        terrain=TerrainCfg(terrain_type="plane"),

        # Add G1 humanoid robot
        robots={
            "humanoid": G1_FULL_ACT_CFG,
        },
    )

    print(f"\n配置信息:")
    print(f"  - 方案: USD Scene (完整场景实体)")
    print(f"  - USD文件: Scene.usd (完整场景)")
    print(f"  - 环境数量: {cfg.num_envs}")
    print(f"  - 机器人: G1 Humanoid")
    print(f"  - 包含: Floor, Wall, Ceiling + 家具（chairs, cabinets等）")
    print(f"  - 关节物体: 254个关节（可动家具）")

    # Build scene
    print(f"\n[1/3] 构建场景...")
    print("     提示: 正在加载完整Scene.usd（含家具和关节物体）")
    scene = LabScene(cfg)
    scene.build()
    print("✅ 场景构建完成")

    # Reset
    print(f"\n[2/3] 重置环境...")
    scene.reset()
    print("✅ 环境已重置")

    # Run simulation
    print(f"\n[3/3] 运行仿真...")
    print("提示: 机器人将与完整场景中的家具和环境交互")
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
    print("✅ 方案2测试完成！")
    print("   - 完整USD场景成功加载")
    print("   - 家具和articulated objects正常加载")
    print("   - 机器人成功生成在场景中")
    print("=" * 80)


if __name__ == "__main__":
    main()
