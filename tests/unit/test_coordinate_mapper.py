from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.utils.config_loader import load_all_configs


def test_square_to_world_corners_and_round_trip() -> None:
    mapper = CoordinateMapper(load_all_configs().board)
    assert np.allclose(mapper.square_to_world(chess.A1), [-0.245, -0.245, 0.52])
    assert np.allclose(mapper.square_to_world(chess.H8), [0.245, 0.245, 0.52])
    for square in chess.SQUARES:
        assert mapper.world_to_square(mapper.square_to_world(square)) == square
    assert mapper.world_to_square(np.array([2.0, 2.0, 0.52])) is None


def test_hover_and_grasp_z() -> None:
    mapper = CoordinateMapper(load_all_configs().board)
    assert mapper.hover_position(chess.E4, 1.2)[2] == 1.2
    assert mapper.grasp_position(chess.E4, 0.8)[2] == 0.8
