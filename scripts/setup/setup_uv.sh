#!/bin/bash
# GenesisLab — One-command setup with uv
# Usage: bash scripts/setup/setup_uv.sh
set -e

echo "🚀 GenesisLab setup with uv"

# Check uv is installed
if ! command -v uv &>/dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 1. Create venv and install all dependencies + editable source packages
echo "📦 Creating virtual environment and installing dependencies..."
uv venv .venv --python 3.10
UV_LINK_MODE=copy uv sync

# 2. Clone and install third-party dependencies
echo "🧩 Setting up third-party dependencies..."

if [ ! -d "third_party/rsl_rl/.git" ]; then
    echo "→ Cloning rsl_rl v3.1.2..."
    git clone --branch v3.1.2 --depth 1 https://github.com/leggedrobotics/rsl_rl.git third_party/rsl_rl
else
    echo "✔ rsl_rl already exists"
fi

if [ ! -d "third_party/genPiHub/.git" ]; then
    echo "→ Cloning genPiHub..."
    git clone git@github.com:Renforce-Dynamics/genPiHub.git third_party/genPiHub
else
    echo "✔ genPiHub already exists"
fi

echo "📦 Installing third-party editable packages..."
uv pip install -e third_party/rsl_rl -e third_party/genPiHub

echo ""
echo "✅ GenesisLab setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To download assets (optional, required for running tasks):"
echo "  bash scripts/setup/download_assets.sh"
echo ""
echo "Quick test:"
echo "  python -c 'import genesis; import genesislab; print(\"All good!\")'"
