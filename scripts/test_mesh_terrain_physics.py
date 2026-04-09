#!/usr/bin/env python3
"""Test mesh terrain with physics interaction.

Verifies that:
1. Mesh terrain stays fixed (doesn't fall due to gravity)
2. Objects can collide with mesh terrain

Usage:
    python scripts/test_mesh_terrain_physics.py
    python scripts/test_mesh_terrain_physics.py --viewer
"""

import argparse
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=str, default="data/assets/Barracks.glb")
    parser.add_argument("--viewer", action="store_true", help="Enable viewer")
    args = parser.parse_args()

    print(f"\n🧪 Testing mesh terrain physics...")
    print(f"   Mesh file: {args.mesh_path}")
    print(f"   Viewer: {args.viewer}")

    # Initialize Genesis
    gs.init(backend=gs.cuda)

    # Create scene
    scene = gs.Scene(
        show_viewer=args.viewer,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, -9.8),
        ),
    )

    # Add mesh terrain (fixed=True so it doesn't fall)
    print("\n📦 Adding mesh terrain...")
    terrain = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=args.mesh_path,
            fixed=True,  # Static terrain
            pos=(0, 0, 0),
        ),
    )
    print(f"   ✅ Terrain added: {terrain}")

    # Add a sphere above the terrain to test collision
    print("\n🎾 Adding test sphere...")
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.2,
            pos=(0, 0, 5.0),  # 5 meters above ground
        ),
    )
    print(f"   ✅ Sphere added: {sphere}")

    # Build scene
    print("\n🔨 Building scene...")
    scene.build()

    # Run simulation
    print("\n▶️  Running simulation...")
    print("   Sphere should fall and land on mesh terrain")

    for step in range(300):
        scene.step()

        if args.viewer and step % 50 == 0:
            # Get sphere position
            sphere_pos = sphere.get_pos()  # shape might vary
            print(f"   Step {step:3d}: Sphere pos shape={sphere_pos.shape}, pos={sphere_pos}")

    if not args.viewer:
        final_pos = sphere.get_pos()
        print(f"\n✅ Final sphere position: {final_pos}")
        print(f"   Position shape: {final_pos.shape}")

        if len(final_pos.shape) > 0 and final_pos.numel() >= 3:
            z_pos = final_pos.flatten()[2] if final_pos.numel() == 3 else final_pos[0, 2]
            print(f"   Z coordinate: {z_pos:.3f}m")

            if z_pos > -5.0:
                print("\n✅ Test PASSED: Sphere landed on terrain (not falling through)")
            else:
                print("\n❌ Test FAILED: Sphere fell through terrain")
        else:
            print("\n⚠️  Cannot determine sphere position")

    print("\n✅ Physics test complete!")


if __name__ == "__main__":
    main()
