from __future__ import annotations

import chess

from mujoco_chess.chess_logic.engine import ChessEngine


def test_initial_board_and_legal_moves() -> None:
    engine = ChessEngine()
    assert engine.get_board_state().board_fen() == chess.Board().board_fen()
    assert len(engine.get_legal_moves()) == 20


def test_apply_does_not_mutate_and_commit_does() -> None:
    engine = ChessEngine()
    move = chess.Move.from_uci("e2e4")
    analysis = engine.apply_move(move)
    assert analysis.from_square == chess.E2
    assert analysis.to_square == chess.E4
    assert engine.get_board_state().piece_at(chess.E2) is not None
    engine.commit_move(move)
    assert engine.get_board_state().piece_at(chess.E4) is not None


def test_capture_detection() -> None:
    engine = ChessEngine()
    for uci in ("e2e4", "d7d5"):
        engine.commit_move(chess.Move.from_uci(uci))
    analysis = engine.apply_move(chess.Move.from_uci("e4d5"))
    assert analysis.is_capture
    assert analysis.captured_piece_type == chess.PAWN


def test_castling_detection() -> None:
    engine = ChessEngine()
    engine._board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    analysis = engine.apply_move(chess.Move.from_uci("e1g1"))
    assert analysis.is_castling
    assert analysis.castling_rook_from == chess.H1
    assert analysis.castling_rook_to == chess.F1


def test_en_passant_and_promotion_detection() -> None:
    engine = ChessEngine()
    engine._board = chess.Board("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    assert engine.apply_move(chess.Move.from_uci("e5d6")).is_en_passant
    engine._board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert engine.apply_move(chess.Move.from_uci("a7a8q")).is_promotion
