#!/bin/bash
# GenesisLab — Download external assets
# Usage: bash scripts/setup/download_assets.sh
set -e

ASSETS_DIR="./data/assets/assetslib"

echo "📦 Downloading GenesisLab assets..."

if [ ! -d "$ASSETS_DIR/.git" ]; then
    echo "→ Cloning assetslib..."
    mkdir -p "$(dirname "$ASSETS_DIR")"
    git clone git@github.com:Renforce-Dynamics/assetslib.git "$ASSETS_DIR"
else
    echo "✔ assetslib already exists at $ASSETS_DIR"
    echo "  To update: cd $ASSETS_DIR && git pull"
fi

echo "✅ Assets ready."
