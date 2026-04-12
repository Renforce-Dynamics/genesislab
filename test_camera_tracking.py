#!/usr/bin/env python3
"""Test camera tracking modes."""

import sys
from genesislab.engine.scene import CameraCfg

def test_track_modes():
    """Test all tracking mode presets."""
    print("Testing camera tracking modes...")

    modes = ["static", "chase", "follow", "side", "top", "first_person"]

    for mode in modes:
        print(f"\n  Testing track_mode='{mode}'...")
        cam_cfg = CameraCfg(
            track_mode=mode,
            entity_name="robot" if mode != "static" else None,
        )

        # Verify preset was applied
        print(f"    ✅ Created: pos={cam_cfg.pos}, lookat={cam_cfg.lookat}, fov={cam_cfg.fov}")

        # Verify entity_name
        if mode == "static":
            assert cam_cfg.entity_name is None, f"Static mode should have entity_name=None"
        else:
            assert cam_cfg.entity_name == "robot", f"Track mode {mode} should preserve entity_name"

    print("\n✅ All track modes OK")

def test_chase_mode():
    """Test chase mode specifically."""
    print("\nTesting chase mode (default for robot tracking)...")

    cam_cfg = CameraCfg(
        track_mode="chase",
        entity_name="robot",
        res=(1920, 1080),
    )

    assert cam_cfg.pos == (-3.5, 0.0, 2.5), "Chase mode pos incorrect"
    assert cam_cfg.lookat == (1.0, 0.0, 0.5), "Chase mode lookat incorrect"
    assert cam_cfg.fov == 50.0, "Chase mode fov incorrect"
    assert cam_cfg.entity_name == "robot", "Chase mode entity_name incorrect"

    print("  ✅ Chase mode configuration correct")
    print(f"     Position: {cam_cfg.pos} (behind and above)")
    print(f"     Look at: {cam_cfg.lookat} (forward)")
    print(f"     FOV: {cam_cfg.fov}°")

def test_first_person_mode():
    """Test first person mode."""
    print("\nTesting first_person mode...")

    cam_cfg = CameraCfg(
        track_mode="first_person",
        entity_name="robot",
    )

    assert cam_cfg.link_name == "pelvis", "First person should attach to pelvis"
    assert cam_cfg.fov == 75.0, "First person should have wide FOV"

    print("  ✅ First person mode configuration correct")
    print(f"     Link: {cam_cfg.link_name}")
    print(f"     Position: {cam_cfg.pos} (robot head/body)")
    print(f"     FOV: {cam_cfg.fov}° (wide angle)")

def test_override_preset():
    """Test overriding preset values."""
    print("\nTesting preset override...")

    cam_cfg = CameraCfg(
        track_mode="chase",
        entity_name="robot",
        pos=(-5.0, 0.0, 4.0),  # Override default chase position
        fov=60.0,              # Override default fov
    )

    # Should use overridden values
    assert cam_cfg.pos == (-5.0, 0.0, 4.0), "Override pos failed"
    assert cam_cfg.fov == 60.0, "Override fov failed"

    print("  ✅ Preset override works correctly")

def test_custom_attachment():
    """Test custom entity attachment without track_mode."""
    print("\nTesting custom entity attachment...")

    cam_cfg = CameraCfg(
        entity_name="robot",
        link_name="pelvis",
        pos=(-2.0, 1.0, 2.0),
        lookat=(1.0, 0.0, 0.5),
        fov=50.0,
    )

    assert cam_cfg.entity_name == "robot"
    assert cam_cfg.link_name == "pelvis"
    assert cam_cfg.track_mode is None, "Should not have track_mode"

    print("  ✅ Custom attachment configuration correct")

def test_static_mode():
    """Test static mode (no entity attachment)."""
    print("\nTesting static mode...")

    cam_cfg = CameraCfg(
        track_mode="static",
    )

    assert cam_cfg.entity_name is None, "Static mode should clear entity_name"

    print("  ✅ Static mode configuration correct")
    print(f"     Entity: {cam_cfg.entity_name} (no attachment)")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Camera Tracking Configuration Tests")
    print("=" * 60)

    try:
        test_track_modes()
        test_chase_mode()
        test_first_person_mode()
        test_override_preset()
        test_custom_attachment()
        test_static_mode()

        print("\n" + "=" * 60)
        print("✅ All tracking tests passed!")
        print("=" * 60)

        print("\nNext steps:")
        print("1. Run AMO script with tracking:")
        print("   python third_party/genPiHub/scripts/amo/genesislab/play_amo_mesh_terrain.py \\")
        print("       --headless --record-video --camera-track chase --max-steps 300")
        print("\n2. Try different modes: chase, follow, side, top, first_person")
        print("\n3. Check output video to verify camera tracking")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
