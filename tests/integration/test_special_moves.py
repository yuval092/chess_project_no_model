from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.move_translation.translator import ActionType, MoveTranslator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import load_all_configs


class RecordingExecutor:
    def __init__(self) -> None:
        self.actions_executed: list = []

    def execute_action(self, action, transaction=None) -> bool:
        self.actions_executed.append(action)
        return True


def _pipeline(fen: str | None = None, executor=None):
    config = load_all_configs()
    mapper = CoordinateMapper(config.board)
    board = chess.Board(fen) if fen else chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({body: mapper.square_to_world(rec.square) for body, rec in registry.records.items() if rec.square is not None})
    translator = MoveTranslator(
        registry,
        mapper,
        SlotManager(config.board, config.pieces),
        config.waypoint,
        config.pieces,
    )
    engine = ChessEngine()
    engine._board = board
    loop = GameLoop(engine, registry, translator, executor or RecordingExecutor())
    return loop, engine, registry, mapper, config


# ---------------------------------------------------------------------------
# M24 — Castling
# ---------------------------------------------------------------------------

_CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def test_castling_kingside_white_action_count() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1g1")

    assert loop.execute_chess_move(move)

    assert len(executor.actions_executed) == 2
    assert all(a.action_type == ActionType.MOVE_PIECE for a in executor.actions_executed)


def test_castling_kingside_white_squares() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1g1")

    assert loop.execute_chess_move(move)

    squares = [a.destination_square for a in executor.actions_executed]
    assert chess.G1 in squares
    assert chess.F1 in squares


def test_castling_kingside_white_king_first() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1g1")

    assert loop.execute_chess_move(move)

    assert executor.actions_executed[0].destination_square == chess.G1
    assert executor.actions_executed[1].destination_square == chess.F1


def test_castling_kingside_white_registry() -> None:
    loop, engine, registry, mapper, _ = _pipeline(_CASTLING_FEN)
    move = chess.Move.from_uci("e1g1")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    king_record = registry.get_piece_at(chess.G1)
    rook_record = registry.get_piece_at(chess.F1)
    assert king_record is not None
    assert king_record.piece_type == chess.KING
    assert rook_record is not None
    assert rook_record.piece_type == chess.ROOK
    assert registry.get_piece_at(chess.E1) is None
    assert registry.get_piece_at(chess.H1) is None
    assert registry.validate_sync(board) == []


def test_castling_queenside_white_action_count() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1c1")

    assert loop.execute_chess_move(move)

    assert len(executor.actions_executed) == 2
    assert all(a.action_type == ActionType.MOVE_PIECE for a in executor.actions_executed)


def test_castling_queenside_white_squares() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1c1")

    assert loop.execute_chess_move(move)

    squares = [a.destination_square for a in executor.actions_executed]
    assert chess.C1 in squares
    assert chess.D1 in squares


def test_castling_queenside_white_king_first() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_CASTLING_FEN, executor)
    move = chess.Move.from_uci("e1c1")

    assert loop.execute_chess_move(move)

    assert executor.actions_executed[0].destination_square == chess.C1
    assert executor.actions_executed[1].destination_square == chess.D1


def test_castling_queenside_white_registry() -> None:
    loop, engine, registry, mapper, _ = _pipeline(_CASTLING_FEN)
    move = chess.Move.from_uci("e1c1")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    king_record = registry.get_piece_at(chess.C1)
    rook_record = registry.get_piece_at(chess.D1)
    assert king_record is not None
    assert king_record.piece_type == chess.KING
    assert rook_record is not None
    assert rook_record.piece_type == chess.ROOK
    assert registry.get_piece_at(chess.E1) is None
    assert registry.get_piece_at(chess.A1) is None
    assert registry.validate_sync(board) == []


_BLACK_CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"


def test_castling_kingside_black_registry() -> None:
    loop, engine, registry, _, _ = _pipeline(_BLACK_CASTLING_FEN)
    move = chess.Move.from_uci("e8g8")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    king_record = registry.get_piece_at(chess.G8)
    rook_record = registry.get_piece_at(chess.F8)
    assert king_record is not None
    assert king_record.piece_type == chess.KING
    assert king_record.color == chess.BLACK
    assert rook_record is not None
    assert rook_record.piece_type == chess.ROOK
    assert rook_record.color == chess.BLACK
    assert registry.validate_sync(board) == []


def test_castling_queenside_black_registry() -> None:
    loop, engine, registry, _, _ = _pipeline(_BLACK_CASTLING_FEN)
    move = chess.Move.from_uci("e8c8")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    king_record = registry.get_piece_at(chess.C8)
    rook_record = registry.get_piece_at(chess.D8)
    assert king_record is not None
    assert king_record.piece_type == chess.KING
    assert king_record.color == chess.BLACK
    assert rook_record is not None
    assert rook_record.piece_type == chess.ROOK
    assert rook_record.color == chess.BLACK
    assert registry.validate_sync(board) == []


# ---------------------------------------------------------------------------
# M25 — En Passant
# ---------------------------------------------------------------------------

# White pawn on e5, black pawn just double-pushed to d5 (en passant target d6).
_EN_PASSANT_FEN = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"


def test_en_passant_action_count() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_EN_PASSANT_FEN, executor)
    move = chess.Move.from_uci("e5d6")

    assert loop.execute_chess_move(move)

    assert len(executor.actions_executed) == 2


def test_en_passant_captured_pawn_teleported_to_graveyard() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_EN_PASSANT_FEN, executor)
    move = chess.Move.from_uci("e5d6")

    assert loop.execute_chess_move(move)

    capture_action = executor.actions_executed[0]
    assert capture_action.action_type == ActionType.TELEPORT
    assert capture_action.graveyard_slot is not None
    assert capture_action.graveyard_color == chess.BLACK


def test_en_passant_captured_pawn_was_on_d5_not_d6() -> None:
    executor = RecordingExecutor()
    loop, _, registry_before, mapper, config = _pipeline(_EN_PASSANT_FEN, executor)
    move = chess.Move.from_uci("e5d6")

    assert loop.execute_chess_move(move)

    capture_action = executor.actions_executed[0]
    d5_world = mapper.square_to_world(chess.D5)
    assert np.allclose(capture_action.source_pos, d5_world, atol=1e-6), (
        f"captured pawn source should be d5={d5_world}, got {capture_action.source_pos}"
    )


def test_en_passant_attacker_moves_to_d6() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_EN_PASSANT_FEN, executor)
    move = chess.Move.from_uci("e5d6")

    assert loop.execute_chess_move(move)

    attacker_action = executor.actions_executed[1]
    assert attacker_action.action_type == ActionType.MOVE_PIECE
    assert attacker_action.destination_square == chess.D6


def test_en_passant_registry_sync() -> None:
    loop, engine, registry, _, _ = _pipeline(_EN_PASSANT_FEN)
    move = chess.Move.from_uci("e5d6")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    attacker = registry.get_piece_at(chess.D6)
    assert attacker is not None
    assert attacker.piece_type == chess.PAWN
    assert attacker.color == chess.WHITE
    captured_list = registry.get_all_captured()
    assert len(captured_list) == 1
    assert captured_list[0].color == chess.BLACK
    assert registry.validate_sync(board) == []


# ---------------------------------------------------------------------------
# M26 — Promotion
# ---------------------------------------------------------------------------

# White pawn on e7, black king on d8 (out of the way), white king on e1.
# e7e8q is a clean push-promotion with no capture.
_PROMOTION_FEN = "3k4/4P3/8/8/8/8/8/4K3 w - - 0 1"

# White pawn on e7, black rook on d8, black king on e8.
# e7d8q is a capture-promotion.
_PROMOTION_CAPTURE_FEN = "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1"


def test_promotion_normal_action_count() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_FEN, executor)
    move = chess.Move.from_uci("e7e8q")

    assert loop.execute_chess_move(move)

    assert len(executor.actions_executed) == 2


def test_promotion_normal_action_types() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_FEN, executor)
    move = chess.Move.from_uci("e7e8q")

    assert loop.execute_chess_move(move)

    assert all(a.action_type == ActionType.TELEPORT for a in executor.actions_executed)


def test_promotion_normal_pawn_goes_to_storage() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_FEN, executor)
    move = chess.Move.from_uci("e7e8q")

    assert loop.execute_chess_move(move)

    pawn_action = executor.actions_executed[0]
    assert pawn_action.pawn_storage_slot is not None
    assert pawn_action.reserve_piece is False
    assert pawn_action.graveyard_slot is None


def test_promotion_normal_reserve_goes_to_e8() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_FEN, executor)
    move = chess.Move.from_uci("e7e8q")

    assert loop.execute_chess_move(move)

    reserve_action = executor.actions_executed[1]
    assert reserve_action.reserve_piece is True
    assert reserve_action.destination_square == chess.E8


def test_promotion_normal_registry_sync() -> None:
    loop, engine, registry, _, _ = _pipeline(_PROMOTION_FEN)
    move = chess.Move.from_uci("e7e8q")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    promoted = registry.get_piece_at(chess.E8)
    assert promoted is not None
    assert promoted.piece_type == chess.QUEEN
    assert promoted.color == chess.WHITE
    stored_pawns = [r for r in registry.records.values() if r.pawn_storage_slot is not None]
    assert len(stored_pawns) == 1
    assert stored_pawns[0].color == chess.WHITE
    assert registry.validate_sync(board) == []


def test_promotion_capture_action_count() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_CAPTURE_FEN, executor)
    move = chess.Move.from_uci("e7d8q")

    assert loop.execute_chess_move(move)

    assert len(executor.actions_executed) == 3


def test_promotion_capture_captured_piece_teleported() -> None:
    executor = RecordingExecutor()
    loop, _, _, _, _ = _pipeline(_PROMOTION_CAPTURE_FEN, executor)
    move = chess.Move.from_uci("e7d8q")

    assert loop.execute_chess_move(move)

    graveyard_action = executor.actions_executed[0]
    assert graveyard_action.action_type == ActionType.TELEPORT
    assert graveyard_action.graveyard_slot is not None
    assert graveyard_action.graveyard_color == chess.BLACK


def test_promotion_capture_registry_sync() -> None:
    loop, engine, registry, _, _ = _pipeline(_PROMOTION_CAPTURE_FEN)
    move = chess.Move.from_uci("e7d8q")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    promoted = registry.get_piece_at(chess.D8)
    assert promoted is not None
    assert promoted.piece_type == chess.QUEEN
    assert promoted.color == chess.WHITE
    captured_list = registry.get_all_captured()
    assert len(captured_list) == 1
    assert captured_list[0].piece_type == chess.ROOK
    assert captured_list[0].color == chess.BLACK
    stored_pawns = [r for r in registry.records.values() if r.pawn_storage_slot is not None]
    assert len(stored_pawns) == 1
    assert registry.validate_sync(board) == []
