#!/usr/bin/env python3
"""Diagnose USD scene scale and position issues."""

import genesis as gs
from genesislab import LabScene
from genesislab.engine.scene import SceneCfg
from genesislab.components.terrains import TerrainCfg
from genesislab.components.environment_objects import (
    EnvironmentObjectsConfig,
    USDObjectCfg,
)
from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG

print("=" * 80)
print("USD 场景缩放和位置诊断")
print("=" * 80)

# Initialize Genesis
gs.init(backend=gs.cuda, precision="32", logging_level="warning")

# Test different configurations
configs = [
    {
        "name": "默认配置 (pos=0,0,0, scale=1.0)",
        "pos": (0.0, 0.0, 0.0),
        "scale": 1.0,
    },
    {
        "name": "场景偏移 (pos=5,0,0, scale=1.0)",
        "pos": (5.0, 0.0, 0.0),
        "scale": 1.0,
    },
    {
        "name": "场景放大2倍 (pos=0,0,0, scale=2.0)",
        "pos": (0.0, 0.0, 0.0),
        "scale": 2.0,
    },
]

print("\n建议测试配置：\n")
for i, config in enumerate(configs, 1):
    print(f"{i}. {config['name']}")
    print(f"   USDObjectCfg(")
    print(f"       pos={config['pos']},")
    print(f"       scale={config['scale']},")
    print(f"   )")
    print()

print("=" * 80)
print("\n要测试配置，使用 --viewer 模式运行：")
print()
print("python third_party/genPiHub/scripts/amo/play_amo_with_terrain_usd.py --viewer")
print()
print("如果场景'小小的'，可能的原因：")
print("  1. USD 场景和机器人都在原点 (0,0,0)，重叠在一起")
print("  2. 摄像机视角太远")
print("  3. 需要调整 scale 参数")
print()
print("解决方案：")
print("  - 方案 A: 将 USD 场景偏移到机器人旁边")
print("    在 amo_env_builder.py 中修改：")
print("    pos=(5.0, 0.0, 0.0)  # 向 X 方向偏移 5 米")
print()
print("  - 方案 B: 放大 USD 场景")
print("    在 amo_env_builder.py 中修改：")
print("    scale=2.0  # 放大 2 倍")
print("=" * 80)
