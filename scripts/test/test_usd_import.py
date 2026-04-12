"""Test script for importing USD scenes into Genesis.

This script demonstrates how to load and visualize USD files in Genesis,
with configurable viewer mode.

Usage:
    # With viewer (default)
    python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd

    # Without viewer
    python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd --no-viewer

    # Fix material issues by clearing cache
    python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd --clear-cache

    # Custom simulation steps
    python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd --num-steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import genesis as gs
import torch


def test_usd_import(
    usd_path: str,
    viewer: bool = True,
    num_steps: int = 500,
    backend: str = "cuda",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: float = 1.0,
    load_articulation: bool = False,
    clear_cache: bool = False,
) -> bool:
    """Test USD scene import.

    Args:
        usd_path: Path to USD file.
        viewer: Whether to enable visualization.
        num_steps: Number of simulation steps to run.
        backend: Backend device ('cuda' or 'cpu').
        pos: Position (x, y, z) for the USD object.
        rot: Rotation quaternion (w, x, y, z) for the USD object.
        scale: Scale factor for the USD object.
        load_articulation: Whether to load as articulated object (with joints).
        clear_cache: Whether to clear Genesis USD cache before importing.

    Returns:
        True if test succeeded.
    """
    # Clear cache if requested
    if clear_cache:
        import shutil
        cache_dir = os.path.expanduser("~/.cache/genesis/usd")
        if os.path.exists(cache_dir):
            print(f"🗑️  Clearing Genesis USD cache: {cache_dir}")
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared")
        else:
            print("ℹ️  No cache to clear")
    # Validate USD file exists
    if not os.path.exists(usd_path):
        print(f"❌ Error: USD file not found: {usd_path}")
        return False

    print(f"🔧 Testing USD import: {usd_path}")
    print(f"   Viewer: {viewer}")
    print(f"   Backend: {backend}")
    print(f"   Steps: {num_steps}")
    print(f"   Position: {pos}")
    print(f"   Rotation (WXYZ): {rot}")
    print(f"   Scale: {scale}")
    print(f"   Load articulation: {load_articulation}")

    try:
        # Create scene
        scene = gs.Scene(
            show_viewer=viewer,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.81),
            ),
        )

        # Create USD morph
        morph = gs.morphs.USD(
            file=usd_path,
            pos=pos,
            quat=rot,
            scale=scale,
            decompose_object_error_threshold=float("inf")
        )

        # Add to scene
        if load_articulation:
            # Load as articulated object (may have joints)
            entity = scene.add_stage(morph=morph)
            print(f"✅ Loaded USD as articulated object (stage)")
        else:
            # Load as static entity
            entity = scene.add_entity(morph=morph)
            print(f"✅ Loaded USD as static entity")

        # Build scene
        print("🔨 Building scene...")
        scene.build()
        print("✅ Scene built successfully")

        # Run simulation
        print(f"▶️  Running {num_steps} simulation steps...")
        for step in range(num_steps):
            scene.step()

            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                print(f"   Step {step + 1}/{num_steps}")

        print(f"✅ Simulation completed ({num_steps} steps)")
        return True

    except Exception as e:
        print(f"❌ Error during USD import test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Genesis USD scene import with optional viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load USD with viewer
  python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd

  # Load USD without viewer
  python scripts/test/test_usd_import.py --usd-path /path/to/scene.usd --no-viewer

  # Custom position and rotation
  python scripts/test/test_usd_import.py --usd-path scene.usd --pos 1.0 2.0 0.5 --rot 0.707 0.707 0.0 0.0

  # Load as articulated object (with joints)
  python scripts/test/test_usd_import.py --usd-path robot.usd --articulation
        """,
    )

    # Required arguments
    parser.add_argument(
        "--usd-path",
        type=str,
        required=True,
        help="Path to USD file to import",
    )

    # Viewer control
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=True,
        help="Enable viewer (default: True)",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Disable viewer",
    )

    # Simulation parameters
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Backend device (default: auto-detect)",
    )

    # USD object parameters
    parser.add_argument(
        "--pos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Position of USD object (default: 0 0 0)",
    )
    parser.add_argument(
        "--rot",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Rotation quaternion in WXYZ format (default: 1 0 0 0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for USD object (default: 1.0)",
    )
    parser.add_argument(
        "--articulation",
        action="store_true",
        default=False,
        help="Load as articulated object (with joints) instead of static entity",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear Genesis USD cache before importing (fixes material issues)",
    )

    args = parser.parse_args()

    # Initialize Genesis
    backend = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Genesis with backend: {backend}")

    gs.init(
        backend=gs.gpu if backend == "cuda" else gs.cpu,
        logging_level="INFO",
    )

    # Configure CUDA if available
    if torch.cuda.is_available() and backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    # Run test
    success = test_usd_import(
        usd_path=args.usd_path,
        viewer=args.viewer,
        num_steps=args.num_steps,
        backend=backend,
        pos=tuple(args.pos),
        rot=tuple(args.rot),
        scale=args.scale,
        load_articulation=args.articulation,
        clear_cache=args.clear_cache,
    )

    if success:
        print("\n✅ USD import test completed successfully!")
        return 0
    else:
        print("\n❌ USD import test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
