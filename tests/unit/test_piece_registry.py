from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry, PieceStatus


def test_registry_initialize_and_update() -> None:
    reg = PhysicalPieceRegistry(board=chess.Board())
    reg.initialize({})
    assert len(reg.get_all_active()) == 32
    assert len(reg.get_all_reserve()) == 16
    pawn = reg.get_piece_at(chess.E2)
    assert pawn is not None
    reg.move_piece(pawn.body_name, chess.E4, np.array([1, 2, 3]))
    assert reg.get_piece_at(chess.E4).body_name == pawn.body_name
    reg.capture_piece(pawn.body_name, chess.WHITE, 0, np.array([4, 5, 6]))
    assert reg.get_piece_by_body(pawn.body_name).status == PieceStatus.CAPTURED


def test_registry_sync() -> None:
    reg = PhysicalPieceRegistry(board=chess.Board())
    reg.initialize({})
    assert reg.validate_sync(chess.Board()) == []
    board = chess.Board()
    board.remove_piece_at(chess.E2)
    assert reg.validate_sync(board)
