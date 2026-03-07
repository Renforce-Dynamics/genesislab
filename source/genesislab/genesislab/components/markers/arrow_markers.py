"""Simple arrow marker utilities for Genesis viewer.

This is a lightweight analogue of Isaac Lab's ``VisualizationMarkers`` for the
Genesis rasterizer. It allows drawing batched arrows given translations and
direction vectors, and is intended primarily for debug visualization.
"""

from __future__ import annotations

from genesislab.utils.configclass import configclass
from typing import Sequence

import numpy as np


@configclass
class ArrowMarkersCfg:
    """Configuration for a group of arrow markers."""

    radius: float = 0.015
    """Arrow body radius."""

    color: tuple[float, float, float, float] = (0.2, 0.9, 0.2, 0.9)
    """Arrow color in RGBA."""
    
    max_arrows: int = 4


class ArrowMarkers:
    """Batched arrow visualizer for Genesis scenes.

    This class wraps Genesis's ``draw_debug_arrow`` API and provides a simple
    ``visualize`` method that accepts batched translations and directions.
    """

    def __init__(self, scene, cfg: ArrowMarkersCfg):
        self._scene = scene
        self.cfg = cfg
        # Keep track of debug arrow nodes so we can clear them like Forge does.
        self._nodes: list[object] = []

    def clear(self) -> None:
        """Clear all currently drawn arrows from the scene."""
        if not self._nodes:
            return
        if hasattr(self._scene, "clear_debug_object"):
            for node in self._nodes:
                try:
                    self._scene.clear_debug_object(node)  # type: ignore[call-arg]
                except Exception:
                    # Debug visualization should never crash the sim
                    continue
        # Reset local cache regardless of whether clearing succeeded.
        self._nodes = []

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

        # Clear previous arrows so debug visuals don't flicker, mirroring Forge:
        # - keep nodes around
        # - clear them before drawing new ones
        self.clear()

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

            node = None
            # Preferred path: use the public Scene debug API, same as Forge.
            if hasattr(self._scene, "draw_debug_arrow"):
                node = self._scene.draw_debug_arrow(  # type: ignore[call-arg]
                    pos=pos,
                    vec=vec,
                    radius=self.cfg.radius,
                    color=self.cfg.color,
                )
            else:
                # Try low-level visualizer context - required for arrow drawing
                if not hasattr(self._scene, "_visualizer"):
                    raise AttributeError(
                        "Scene does not have '_visualizer' attribute. "
                        "Arrow markers require a visualizer to draw arrows."
                    )
                
                visualizer = self._scene._visualizer
                if visualizer is None:
                    raise ValueError(
                        "Scene._visualizer is None. "
                        "Arrow markers require a visualizer to be initialized."
                    )
                
                if not hasattr(visualizer, "context"):
                    raise AttributeError(
                        "Visualizer does not have 'context' attribute. "
                        "Arrow markers require visualizer context to draw arrows."
                    )
                
                ctx = visualizer.context
                if ctx is None:
                    raise ValueError(
                        "Visualizer.context is None. "
                        "Arrow markers require visualizer context to be initialized."
                    )
                
                if not hasattr(ctx, "draw_debug_arrow"):
                    raise AttributeError(
                        "Visualizer context does not have 'draw_debug_arrow' method. "
                        "Arrow markers require this method to draw arrows."
                    )
                
                node = ctx.draw_debug_arrow(
                    pos=pos,
                    vec=vec,
                    radius=self.cfg.radius,
                    color=self.cfg.color,
                    persistent=False,
                )

            if node is not None:
                self._nodes.append(node)

            if self.cfg.max_arrows > -1 and i > self.cfg.max_arrows:
                break

