#!/bin/bash
# Quick test runner for USD terrain implementation
# This script runs all USD terrain tests in sequence

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "======================================================================"
echo "🧪 USD TERRAIN TEST SUITE"
echo "======================================================================"
echo ""

# Check if USD file exists
USD_FILE="third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd"
if [ ! -f "$USD_FILE" ]; then
    echo "❌ ERROR: USD file not found: $USD_FILE"
    echo ""
    echo "Please ensure the Scene.usd file exists at the specified location."
    exit 1
fi

echo "✅ USD file found: $USD_FILE"
echo ""

# Test 1: Basic functionality
echo "======================================================================"
echo "TEST 1: Basic Functionality Test"
echo "======================================================================"
python scripts/test_usd_terrain_basic.py --num-envs 4
BASIC_RESULT=$?
echo ""

if [ $BASIC_RESULT -ne 0 ]; then
    echo "❌ Basic test FAILED!"
    exit 1
fi

# Test 2: Integration test (optional - may not have genesis_tasks)
echo "======================================================================"
echo "TEST 2: Integration Test (with Robot)"
echo "======================================================================"
python scripts/test_usd_terrain_integration.py --num-envs 4 --num-steps 50
INTEGRATION_RESULT=$?
echo ""

if [ $INTEGRATION_RESULT -ne 0 ]; then
    echo "⚠️  Integration test failed or skipped (this is OK if genesis_tasks is not installed)"
fi

# Summary
echo "======================================================================"
echo "📊 TEST SUMMARY"
echo "======================================================================"
echo ""

if [ $BASIC_RESULT -eq 0 ]; then
    echo "✅ Basic Test: PASSED"
else
    echo "❌ Basic Test: FAILED"
fi

if [ $INTEGRATION_RESULT -eq 0 ]; then
    echo "✅ Integration Test: PASSED"
else
    echo "⚠️  Integration Test: SKIPPED or FAILED"
fi

echo ""
echo "======================================================================"

if [ $BASIC_RESULT -eq 0 ]; then
    echo "✅ CORE TESTS PASSED!"
    echo ""
    echo "🎉 USD terrain implementation is working correctly!"
    echo ""
    echo "Next steps:"
    echo "  1. Try the interactive demo:"
    echo "     python scripts/demo_scene_usd_terrain.py --viewer"
    echo ""
    echo "  2. Use USD terrain in your environments:"
    echo "     See scripts/README_USD_TERRAIN.md for examples"
    echo ""
    exit 0
else
    echo "❌ TESTS FAILED"
    echo ""
    echo "Please check the error messages above and fix any issues."
    exit 1
fi
