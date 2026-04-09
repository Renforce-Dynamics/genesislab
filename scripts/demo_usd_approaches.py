#!/usr/bin/env python3
"""Demonstrate two approaches for using USD in GenesisLab.

Approach 1: USD as Terrain (for static geometry only)
    - Use TerrainCfg with terrain_type="usd"
    - Best for purely static environments (floors, walls)
    - Integrated with terrain system for multi-env layouts

Approach 2: USD as Scene Entities (more flexible)
    - Use SceneCfg.usd_scene_path
    - Can load complete scenes with articulated objects
    - Objects can interact with robots

Usage:
    # Approach 1: Static terrain
    python scripts/demo_usd_approaches.py --approach terrain

    # Approach 2: Scene entities
    python scripts/demo_usd_approaches.py --approach scene
"""

import argparse
import genesis as gs
from genesislab import LabScene, SceneCfg
from genesislab.components.terrains import TerrainCfg
from genesislab.components.robots import G1Cfg


def demo_approach1_terrain():
    """Approach 1: USD as Terrain (static only)."""
    print("\n" + "=" * 80)
    print("APPROACH 1: USD as Terrain (Static Geometry)")
    print("=" * 80)

    cfg = SceneCfg(
        num_envs=4,
        env_spacing=(12.0, 12.0),
        viewer=True,
        backend="cuda",
        # USD as terrain - requires purely static USD
        terrain=TerrainCfg(
            terrain_type="usd",
            usd_path="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Terrain.usd",
            env_spacing=12.0,
        ),
        robots={"humanoid": G1Cfg()},
    )

    scene = LabScene(cfg)
    scene.build()

    print("\n✅ Approach 1: Scene built with USD terrain")
    print(f"   - Terrain: USD (static Floor, Wall, Ceiling)")
    print(f"   - Environments: {cfg.num_envs}")
    print(f"   - Robot: G1 Humanoid")

    # Reset and step
    scene.reset()
    for _ in range(100):
        scene.step()

    print("\n✅ Simulation complete!")


def demo_approach2_scene():
    """Approach 2: USD as Scene Entities (with articulated objects)."""
    print("\n" + "=" * 80)
    print("APPROACH 2: USD as Scene Entities (Full Scene)")
    print("=" * 80)

    cfg = SceneCfg(
        num_envs=1,  # Scene approach typically uses single env
        viewer=True,
        backend="cuda",
        # USD as background scene (can include furniture, articulated objects)
        usd_scene_path="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd",
        # Optional: add simple plane terrain for robot spawning
        terrain=TerrainCfg(terrain_type="plane"),
        robots={"humanoid": G1Cfg()},
    )

    scene = LabScene(cfg)
    scene.build()

    print("\n✅ Approach 2: Scene built with USD entities")
    print(f"   - USD Scene: Complete environment (furniture + static geometry)")
    print(f"   - Terrain: Plane (for robot placement)")
    print(f"   - Robot: G1 Humanoid")

    # Reset and step
    scene.reset()
    for _ in range(100):
        scene.step()

    print("\n✅ Simulation complete!")


def main():
    parser = argparse.ArgumentParser(description="Demo USD approaches in GenesisLab")
    parser.add_argument(
        "--approach",
        choices=["terrain", "scene"],
        default="terrain",
        help="Which approach to demonstrate",
    )
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.cuda if gs.cuda_available() else gs.cpu, precision="32", logging_level="warning")

    if args.approach == "terrain":
        demo_approach1_terrain()
    else:
        demo_approach2_scene()


if __name__ == "__main__":
    main()
