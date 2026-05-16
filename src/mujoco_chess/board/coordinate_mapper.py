from __future__ import annotations

"""Board orientation convention.

White plays from rank 1 at minimum Y. File a is at minimum X and file h is at
maximum X. `square_to_world` returns world coordinates for square centers on the
top board surface; those positions are for motion targets and must not be used
as local MuJoCo geom positions inside the `board_frame` body.
"""

import chess
import numpy as np

from mujoco_chess.utils.config_loader import BoardConfig


class CoordinateMapper:
    def __init__(self, board_config: BoardConfig) -> None:
        self.config = board_config

    def square_to_world(self, square: chess.Square) -> np.ndarray:
        file_index, rank_index = self.square_to_file_rank(square)
        size = self.config.square_size
        return np.array(
            [
                self.config.board_origin_x + file_index * size + size / 2,
                self.config.board_origin_y + rank_index * size + size / 2,
                self.config.board_origin_z + self.config.board_thickness,
            ],
            dtype=float,
        )

    def world_to_square(self, pos: np.ndarray) -> chess.Square | None:
        x = float(pos[0])
        y = float(pos[1])
        if not (self.config.board_origin_x <= x < self.config.board_max_x):
            return None
        if not (self.config.board_origin_y <= y < self.config.board_max_y):
            return None
        file_index = int((x - self.config.board_origin_x) // self.config.square_size)
        rank_index = int((y - self.config.board_origin_y) // self.config.square_size)
        if 0 <= file_index < 8 and 0 <= rank_index < 8:
            return chess.square(file_index, rank_index)
        return None

    def square_to_file_rank(self, square: chess.Square) -> tuple[int, int]:
        return chess.square_file(square), chess.square_rank(square)

    def hover_position(self, square: chess.Square, hover_z: float) -> np.ndarray:
        pos = self.square_to_world(square)
        pos[2] = hover_z
        return pos

    def grasp_position(self, square: chess.Square, grasp_z: float) -> np.ndarray:
        pos = self.square_to_world(square)
        pos[2] = grasp_z
        return pos
