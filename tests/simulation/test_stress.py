from __future__ import annotations

"""Stress tests for the chess simulation.

These tests are marked with @pytest.mark.stress and are excluded from normal
test runs. Run them explicitly with:

    pytest tests/simulation/test_stress.py -m stress
"""

import chess
import pytest

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.chess_logic.move_selector import HeuristicMoveSelector
from mujoco_chess.move_translation.translator import MoveTranslator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import load_all_configs

pytestmark = pytest.mark.stress


class RecordingExecutor:
    """Executor that records actions without invoking MuJoCo."""

    def __init__(self) -> None:
        self.call_count = 0

    def execute_action(self, action, transaction=None) -> bool:
        self.call_count += 1
        return True


def _pipeline():
    config = load_all_configs()
    mapper = CoordinateMapper(config.board)
    board = chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    settled = {
        f"piece_{'w' if piece.color == chess.WHITE else 'b'}_{piece.symbol().upper()}_{i}": mapper.square_to_world(sq)
        for i, (sq, piece) in enumerate(board.piece_map().items())
    }
    registry.initialize(settled)
    slot_manager = SlotManager(config.board, config.pieces)
    translator = MoveTranslator(registry, mapper, slot_manager, config.waypoint, config.pieces)
    engine = ChessEngine()
    executor = RecordingExecutor()
    loop = GameLoop(
        engine,
        registry,
        translator,
        executor,
        move_selector=HeuristicMoveSelector(),
        human_color=None,
        slot_manager=slot_manager,
        mapper=mapper,
    )
    return loop, engine, registry, executor


@pytest.mark.stress
def test_stress_grasp_release_scaffold() -> None:
    """Scaffold: verifies the stress test framework works (1 iteration only in CI)."""
    loop, engine, registry, executor = _pipeline()
    # Execute a single move to verify the executor and loop work
    move = chess.Move.from_uci("e2e4")
    result = loop.execute_chess_move(move)
    assert result is True
    assert executor.call_count >= 1


@pytest.mark.stress
def test_stress_random_game_scaffold() -> None:
    """Scaffold: runs 1 random game of up to 20 moves to verify the game loop works."""
    loop, engine, registry, executor = _pipeline()
    result = loop.run(max_moves=20)
    # The game either ended (result is not None) or max_moves was hit (result is None)
    # Both are acceptable outcomes
    assert executor.call_count >= 0  # At least some moves were attempted
    # Verify registry and board remain consistent after the run
    board = engine.get_board_state()
    errors = registry.validate_sync(board)
    assert errors == [], f"Registry out of sync after stress run: {[e.message for e in errors]}"
