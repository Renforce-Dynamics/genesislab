#!/usr/bin/env python3
"""Quick test script for camera and recording functionality."""

import sys
import genesis as gs
from genesislab.engine.scene import SceneCfg, CameraCfg, RecordingCfg

def test_camera_config():
    """Test that camera config can be created."""
    print("Testing CameraCfg...")
    cam_cfg = CameraCfg(
        res=(1280, 720),
        pos=(5.0, 0.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
        fov=45.0,
    )
    assert cam_cfg.res == (1280, 720)
    assert cam_cfg.pos == (5.0, 0.0, 3.0)
    print("✅ CameraCfg OK")

def test_recording_config():
    """Test that recording config can be created."""
    print("Testing RecordingCfg...")
    rec_cfg = RecordingCfg(
        enabled=True,
        save_path="output/test.mp4",
        fps=60,
    )
    assert rec_cfg.enabled == True
    assert rec_cfg.fps == 60
    print("✅ RecordingCfg OK")

def test_scene_config():
    """Test that scene config accepts camera and recording."""
    print("Testing SceneCfg with camera and recording...")
    scene_cfg = SceneCfg(
        num_envs=1,
        viewer=False,
        camera=CameraCfg(
            res=(1280, 720),
            pos=(5.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.5),
        ),
        recording=RecordingCfg(
            enabled=False,  # Don't actually record in test
            save_path="output/test.mp4",
        ),
    )
    assert scene_cfg.camera is not None
    assert scene_cfg.recording is not None
    print("✅ SceneCfg with camera/recording OK")

def test_scene_config_no_camera():
    """Test that scene config works without camera."""
    print("Testing SceneCfg without camera...")
    scene_cfg = SceneCfg(
        num_envs=1,
        viewer=False,
        camera=None,
        recording=None,
    )
    assert scene_cfg.camera is None
    assert scene_cfg.recording is None
    print("✅ SceneCfg without camera OK")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Camera and Recording Configuration Tests")
    print("=" * 60)

    try:
        test_camera_config()
        test_recording_config()
        test_scene_config()
        test_scene_config_no_camera()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test with actual scene build (requires full env setup)")
        print("2. Run AMO script with --record-video flag")
        print("3. Verify video output is created")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
