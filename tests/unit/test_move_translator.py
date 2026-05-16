from __future__ import annotations

import chess

from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.move_translation.translator import ActionType, MoveTranslator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import load_all_configs


def _translator(board: chess.Board | None = None):
    config = load_all_configs()
    board = board or chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({})
    return MoveTranslator(registry, CoordinateMapper(config.board), SlotManager(config.board, config.pieces), config.waypoint, config.pieces)


def test_normal_move_translation() -> None:
    actions = _translator().translate(chess.Move.from_uci("e2e4"), chess.Board())
    assert len(actions) == 1
    assert actions[0].piece_body.startswith("piece_w_P")
    assert actions[0].destination_square == chess.E4
    assert actions[0].action_type == ActionType.MOVE_PIECE
    assert actions[0].grasp_z > actions[0].source_pos[2]
    assert actions[0].release_z > actions[0].destination_pos[2]


def test_capture_translation() -> None:
    board = chess.Board("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1")
    actions = _translator(board).translate(chess.Move.from_uci("e4d5"), board)
    assert len(actions) == 2
    # Captured piece is teleported to graveyard; attacker uses arm.
    assert actions[0].action_type == ActionType.TELEPORT
    assert actions[0].graveyard_slot == 0
    assert actions[1].action_type == ActionType.MOVE_PIECE


def test_castling_translation() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    actions = _translator(board).translate(chess.Move.from_uci("e1g1"), board)
    assert [a.destination_square for a in actions] == [chess.G1, chess.F1]
    assert all(a.action_type == ActionType.MOVE_PIECE for a in actions)
