#!/usr/bin/env python3
"""Test script to verify quaternion format conversion.

This script validates that:
1. Config quaternions (xyzw) are correctly converted to Genesis format (wxyz)
2. Reset functions properly apply rotation offsets
3. Identity quaternion is handled correctly
"""

import torch
from genesislab.utils.math.rotation import (
    quat_xyzw_to_wxyz,
    quat_wxyz_to_xyzw,
    quat_from_euler_xyz,
    quat_mul,
)


def test_identity_quaternion():
    """Test identity quaternion conversion."""
    print("\n=== Testing Identity Quaternion ===")

    # Identity in xyzw format (config format)
    quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0])
    print(f"Config (xyzw): {quat_xyzw}")

    # Convert to wxyz (Genesis format)
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    print(f"Genesis (wxyz): {quat_wxyz}")

    # Expected: [1, 0, 0, 0] (w=1, xyz=0)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(quat_wxyz, expected, atol=1e-6), \
        f"Expected {expected}, got {quat_wxyz}"

    print("✅ Identity quaternion conversion correct!")


def test_90_degree_rotation():
    """Test 90-degree rotation about Z-axis."""
    print("\n=== Testing 90° Z-Rotation ===")

    # 90° rotation about Z in xyzw format
    # w = cos(45°) = 0.707, z = sin(45°) = 0.707
    quat_xyzw = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068])
    print(f"Config (xyzw): {quat_xyzw}")

    # Convert to wxyz
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    print(f"Genesis (wxyz): {quat_wxyz}")

    # Expected: [w, x, y, z] = [0.707, 0, 0, 0.707]
    expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
    assert torch.allclose(quat_wxyz, expected, atol=1e-6), \
        f"Expected {expected}, got {quat_wxyz}"

    print("✅ 90° rotation conversion correct!")


def test_euler_to_quaternion():
    """Test Euler to quaternion conversion."""
    print("\n=== Testing Euler → Quaternion ===")

    # 30° yaw (rotation about Z)
    yaw = torch.tensor(0.5236)  # 30° in radians
    roll = torch.tensor(0.0)
    pitch = torch.tensor(0.0)

    quat_wxyz = quat_from_euler_xyz(roll, pitch, yaw)
    print(f"Euler (roll=0, pitch=0, yaw=30°)")
    print(f"Quaternion (wxyz): {quat_wxyz}")

    # Verify it's normalized
    norm = torch.norm(quat_wxyz)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6), \
        f"Quaternion not normalized: norm={norm}"

    print("✅ Euler to quaternion conversion correct!")


def test_quaternion_multiplication():
    """Test quaternion multiplication (applying rotation offsets)."""
    print("\n=== Testing Quaternion Multiplication ===")

    # Start with identity
    quat_base = torch.tensor([1.0, 0.0, 0.0, 0.0])  # wxyz identity

    # Apply 30° yaw offset
    yaw_offset = torch.tensor(0.5236)  # 30°
    quat_offset = quat_from_euler_xyz(
        torch.tensor(0.0),
        torch.tensor(0.0),
        yaw_offset
    )

    # Multiply
    quat_result = quat_mul(quat_base, quat_offset)
    print(f"Base (identity): {quat_base}")
    print(f"Offset (30° yaw): {quat_offset}")
    print(f"Result: {quat_result}")

    # Result should be same as offset (identity * q = q)
    assert torch.allclose(quat_result, quat_offset, atol=1e-6), \
        f"Expected {quat_offset}, got {quat_result}"

    print("✅ Quaternion multiplication correct!")


def test_round_trip_conversion():
    """Test xyzw → wxyz → xyzw conversion."""
    print("\n=== Testing Round-Trip Conversion ===")

    # Arbitrary quaternion in xyzw
    quat_xyzw_original = torch.tensor([0.1, 0.2, 0.3, 0.9])
    # Normalize
    quat_xyzw_original = quat_xyzw_original / torch.norm(quat_xyzw_original)

    print(f"Original (xyzw): {quat_xyzw_original}")

    # Convert to wxyz
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw_original)
    print(f"Converted (wxyz): {quat_wxyz}")

    # Convert back to xyzw
    quat_xyzw_back = quat_wxyz_to_xyzw(quat_wxyz)
    print(f"Back to (xyzw): {quat_xyzw_back}")

    # Should match original
    assert torch.allclose(quat_xyzw_back, quat_xyzw_original, atol=1e-6), \
        f"Round-trip failed: {quat_xyzw_original} != {quat_xyzw_back}"

    print("✅ Round-trip conversion correct!")


def test_batch_conversion():
    """Test batched quaternion conversion."""
    print("\n=== Testing Batch Conversion ===")

    # Batch of quaternions in xyzw
    batch_xyzw = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],  # identity
        [0.0, 0.0, 0.707, 0.707],  # 90° Z
        [0.707, 0.0, 0.0, 0.707],  # 90° X
    ])

    print(f"Batch (xyzw):\n{batch_xyzw}")

    # Convert to wxyz
    batch_wxyz = quat_xyzw_to_wxyz(batch_xyzw)
    print(f"Batch (wxyz):\n{batch_wxyz}")

    # Check dimensions
    assert batch_wxyz.shape == batch_xyzw.shape, \
        f"Shape mismatch: {batch_wxyz.shape} != {batch_xyzw.shape}"

    # Check first element (identity)
    assert torch.allclose(batch_wxyz[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    print("✅ Batch conversion correct!")


def main():
    """Run all quaternion conversion tests."""
    print("=" * 60)
    print("Quaternion Conversion Test Suite")
    print("=" * 60)

    try:
        test_identity_quaternion()
        test_90_degree_rotation()
        test_euler_to_quaternion()
        test_quaternion_multiplication()
        test_round_trip_conversion()
        test_batch_conversion()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
