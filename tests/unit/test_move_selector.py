from __future__ import annotations

import chess

from mujoco_chess.chess_logic.move_selector import HeuristicMoveSelector, RandomMoveSelector, create_move_selector
from mujoco_chess.utils.config_loader import load_all_configs


def test_selectors_return_legal_moves() -> None:
    board = chess.Board()
    for selector in (RandomMoveSelector(), HeuristicMoveSelector()):
        assert selector.choose_move(board) in board.legal_moves


def test_selector_factory() -> None:
    config = load_all_configs().game
    assert isinstance(create_move_selector(config), HeuristicMoveSelector)
    assert isinstance(create_move_selector(config.model_copy(update={"selector_type": "random"})), RandomMoveSelector)
