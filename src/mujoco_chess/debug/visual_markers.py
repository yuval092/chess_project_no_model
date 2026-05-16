from __future__ import annotations

import logging

import chess

LOGGER = logging.getLogger(__name__)


class VisualMarkers:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def highlight_square(self, env, square: chess.Square, rgba=(1, 1, 0, 0.5)) -> None:
        """Change the rgba of the debug_center geom for the given square.

        No-op if not enabled or if the debug center geom is not present in the model
        (i.e. debug.show_square_centers was False when the XML was generated).
        """
        if not self.enabled:
            return
        file_name = chess.FILE_NAMES[chess.square_file(square)]
        rank_name = str(chess.square_rank(square) + 1)
        geom_name = f"debug_center_{file_name}{rank_name}"
        try:
            geom_id = env.model.geom(geom_name).id
            env.model.geom_rgba[geom_id] = list(rgba)
        except (KeyError, AttributeError, Exception) as exc:
            LOGGER.debug("highlight_square: geom '%s' not found or model unavailable: %s", geom_name, exc)
