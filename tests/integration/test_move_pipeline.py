from __future__ import annotations

import chess
import numpy as np
import pytest

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.move_translation.translator import MoveTranslator, PartialMoveFailureError
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import load_all_configs


class RecordingExecutor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.transactions = []

    def execute_action(self, action, transaction=None) -> bool:
        del action
        self.transactions.append(transaction)
        return not self.fail


def _pipeline(executor: RecordingExecutor | None = None) -> tuple[GameLoop, ChessEngine, PhysicalPieceRegistry, CoordinateMapper]:
    config = load_all_configs()
    mapper = CoordinateMapper(config.board)
    board = chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({body: mapper.square_to_world(record.square) for body, record in _starting_records(board).items()})
    translator = MoveTranslator(
        registry,
        mapper,
        SlotManager(config.board, config.pieces),
        config.waypoint,
        config.pieces,
    )
    engine = ChessEngine()
    return GameLoop(engine, registry, translator, executor), engine, registry, mapper


def _starting_records(board: chess.Board):
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({})
    return {body: record for body, record in registry.records.items() if record.square is not None}


def test_execute_chess_move_e2e4_commits_board_and_registry() -> None:
    executor = RecordingExecutor()
    loop, engine, registry, mapper = _pipeline(executor)
    move = chess.Move.from_uci("e2e4")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    assert board.piece_at(chess.E2) is None
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    record = registry.get_piece_at(chess.E4)
    assert record is not None
    assert registry.get_piece_at(chess.E2) is None
    np.testing.assert_allclose(record.expected_pos, mapper.square_to_world(chess.E4))
    assert registry.validate_sync(board) == []
    assert executor.transactions[0] is not None
    assert executor.transactions[0].completed_action_indices == [0]
    assert executor.transactions[0].committed


def test_execute_chess_move_failure_does_not_commit() -> None:
    executor = RecordingExecutor(fail=True)
    loop, engine, registry, _ = _pipeline(executor)
    move = chess.Move.from_uci("e2e4")

    with pytest.raises(PartialMoveFailureError) as exc:
        loop.execute_chess_move(move)

    assert exc.value.failed_action_index == 0
    assert not exc.value.transaction.committed
    assert exc.value.transaction.completed_action_indices == []
    assert engine.get_board_state().piece_at(chess.E2) == chess.Piece(chess.PAWN, chess.WHITE)
    assert registry.get_piece_at(chess.E2) is not None
    assert registry.get_piece_at(chess.E4) is None


def test_execute_capture_commits_attacker_and_captured_piece() -> None:
    executor = RecordingExecutor()
    loop, engine, registry, _ = _pipeline(executor)

    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("d7d5"))
    assert loop.execute_chess_move(chess.Move.from_uci("e4d5"))

    board = engine.get_board_state()
    assert board.piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.WHITE)
    attacker = registry.get_piece_at(chess.D5)
    assert attacker is not None
    captured = registry.get_all_captured()
    assert len(captured) == 1
    assert captured[0].color == chess.BLACK
    assert captured[0].graveyard_slot == 0
    assert registry.validate_sync(board) == []
    capture_transaction = executor.transactions[-2]
    assert capture_transaction.completed_action_indices == [0, 1]
    assert set(capture_transaction.pending_expected_positions) == {captured[0].body_name, attacker.body_name}
