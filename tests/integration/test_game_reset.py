from __future__ import annotations

import chess
import numpy as np
import pytest

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.move_translation.translator import MoveTranslator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry, PieceStatus
from mujoco_chess.utils.config_loader import load_all_configs


class RecordingExecutor:
    def __init__(self) -> None:
        self.transactions = []

    def execute_action(self, action, transaction=None) -> bool:
        self.transactions.append(transaction)
        return True


def _pipeline() -> tuple[GameLoop, ChessEngine, PhysicalPieceRegistry, CoordinateMapper, SlotManager]:
    config = load_all_configs()
    mapper = CoordinateMapper(config.board)
    board = chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    settled = {
        body: mapper.square_to_world(record.square)
        for body, record in _starting_records(board).items()
    }
    registry.initialize(settled)
    slot_manager = SlotManager(config.board, config.pieces)
    translator = MoveTranslator(
        registry,
        mapper,
        slot_manager,
        config.waypoint,
        config.pieces,
    )
    engine = ChessEngine()
    loop = GameLoop(
        engine,
        registry,
        translator,
        RecordingExecutor(),
        slot_manager=slot_manager,
        mapper=mapper,
    )
    return loop, engine, registry, mapper, slot_manager


def _starting_records(board: chess.Board):
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({})
    return {body: record for body, record in registry.records.items() if record.square is not None}


def test_reset_restores_starting_board() -> None:
    """After executing moves and calling reset(), board is starting position."""
    loop, engine, registry, mapper, slot_manager = _pipeline()

    # Make some moves
    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("e7e5"))
    assert loop.execute_chess_move(chess.Move.from_uci("d2d4"))

    # Verify board changed
    board_before_reset = engine.get_board_state()
    assert board_before_reset.piece_at(chess.E4) is not None
    assert board_before_reset.piece_at(chess.E2) is None

    # Reset
    loop.reset()

    # Verify starting position
    board_after = engine.get_board_state()
    starting = chess.Board()
    assert board_after.fen() == starting.fen()


def test_reset_clears_captured_pieces() -> None:
    """After capture + reset, get_all_captured() is empty."""
    loop, engine, registry, mapper, slot_manager = _pipeline()

    # Sequence that results in a capture: e2e4, d7d5, e4d5
    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("d7d5"))
    assert loop.execute_chess_move(chess.Move.from_uci("e4d5"))

    # Verify a capture happened
    captured = registry.get_all_captured()
    assert len(captured) > 0

    # Reset
    loop.reset()

    # Verify no captured pieces
    captured_after = registry.get_all_captured()
    assert len(captured_after) == 0


def test_reset_restores_reserve_piece_status() -> None:
    """After reset, all reserve pieces are RESERVE status."""
    loop, engine, registry, mapper, slot_manager = _pipeline()

    # Make some moves to change state
    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("d7d5"))

    # Reset
    loop.reset()

    # All reserve pieces should be RESERVE status
    reserve_pieces = [
        rec for rec in registry.records.values()
        if rec.status == PieceStatus.RESERVE
    ]
    # Starting position has 16 reserve pieces (2 per type per color: Q,R,B,N × 2 colors × 2 slots)
    assert len(reserve_pieces) == 16
    for rec in reserve_pieces:
        assert rec.status == PieceStatus.RESERVE
        assert rec.square is None


def test_reset_registry_sync_passes() -> None:
    """After reset, validate_sync returns no errors."""
    loop, engine, registry, mapper, slot_manager = _pipeline()

    # Make some moves
    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("e7e5"))

    # Reset
    loop.reset()

    # Validate registry is in sync with the board
    board = engine.get_board_state()
    errors = registry.validate_sync(board)
    assert errors == [], f"Sync errors after reset: {[e.message for e in errors]}"
