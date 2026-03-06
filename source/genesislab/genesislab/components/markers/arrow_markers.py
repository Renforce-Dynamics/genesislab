"""Simple arrow marker utilities for Genesis viewer.

This is a lightweight analogue of Isaac Lab's ``VisualizationMarkers`` for the
Genesis rasterizer. It allows drawing batched arrows given translations and
direction vectors, and is intended primarily for debug visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ArrowMarkersCfg:
    """Configuration for a group of arrow markers."""

    radius: float = 0.015
    """Arrow body radius."""

    color: tuple[float, float, float, float] = (0.2, 0.9, 0.2, 0.9)
    """Arrow color in RGBA."""


class ArrowMarkers:
    """Batched arrow visualizer for Genesis scenes.

    This class wraps Genesis's ``draw_debug_arrow`` API and provides a simple
    ``visualize`` method that accepts batched translations and directions.
    """

    def __init__(self, scene, cfg: ArrowMarkersCfg):
        self._scene = scene
        # Access low-level rasterizer context for more control.
        self._ctx = scene._visualizer.context  # type: ignore[attr-defined]
        self.cfg = cfg

    def visualize(
        self,
        translations: np.ndarray,
        directions: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """Draw arrows for a batch of positions and directions.

        Args:
            translations: Array of shape (N, 3) with world-frame positions.
            directions: Array of shape (N, 3) with world-frame vectors.
            mask: Optional boolean mask of shape (N,) to select a subset.
        """
        if translations.size == 0 or directions.size == 0:
            return

        assert translations.shape == directions.shape, (
            f"translations and directions must have same shape, got "
            f"{translations.shape} and {directions.shape}."
        )

        if mask is None:
            idx_range: Sequence[int] = range(translations.shape[0])
        else:
            idx_range = np.nonzero(mask)[0]

        for i in idx_range:
            pos = translations[i]
            vec = directions[i]
            # Skip near-zero arrows to avoid visual clutter.
            if np.linalg.norm(vec) < 1e-5:
                continue
            self._ctx.draw_debug_arrow(
                pos=pos,
                vec=vec,
                radius=self.cfg.radius,
                color=self.cfg.color,
                persistent=False,
            )

