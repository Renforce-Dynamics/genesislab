"""Genesis Assets package for robot and asset configurations.

This package provides asset paths and configurations for GenesisLab.
All asset paths are resolved relative to the package directory structure.
"""

import os

# Package directory paths (similar to robotlib's structure)
GENESIS_ASSETS_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
"""Path to the genesis_assets package directory."""

GENESIS_ASSETS_DATA_DIR = os.path.join(GENESIS_ASSETS_REPO_DIR, "..", "..", "..", "data")
"""Path to the data directory (relative to package, similar to robotlib structure)."""

GENESIS_ASSETS_ASSETS_DIR = os.path.join(GENESIS_ASSETS_DATA_DIR, "assets", "assetslib")
"""Path to the assets directory."""

GENESIS_ASSETS_USD_DIR = os.path.join(GENESIS_ASSETS_ASSETS_DIR, "usd")
"""Path to the USD assets directory."""

GENESIS_ASSETS_ASSETLIB_DIR = os.path.join(GENESIS_ASSETS_ASSETS_DIR, "assetslib")
"""Path to the assetslib directory (for third-party assets)."""

GENESIS_ASSETS_UNITREE_MODEL_DIR = os.path.join(GENESIS_ASSETS_ASSETS_DIR, "unitree")
"""Path to the Unitree model directory."""
