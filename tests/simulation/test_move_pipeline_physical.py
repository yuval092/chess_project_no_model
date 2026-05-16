from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.app.startup import bootstrap
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.move_translation.translator import MoveTranslator
from mujoco_chess.mujoco_env.env import MuJoCoEnv
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry


def test_physical_game_loop_executes_e2e4() -> None:
    loop, engine, registry, env, mapper = _physical_loop()
    move = chess.Move.from_uci("e2e4")

    assert loop.execute_chess_move(move)

    board = engine.get_board_state()
    assert board.piece_at(chess.E2) is None
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert registry.get_piece_at(chess.E2) is None
    record = registry.get_piece_at(chess.E4)
    assert record is not None
    assert np.linalg.norm(env.get_body_pos(record.body_name)[:2] - mapper.square_to_world(chess.E4)[:2]) < loop.translator.waypoint.position_tolerance * 2
    assert registry.validate_sync(board) == []


def test_physical_game_loop_executes_capture_to_graveyard() -> None:
    loop, engine, registry, env, mapper = _physical_loop()

    assert loop.execute_chess_move(chess.Move.from_uci("e2e4"))
    assert loop.execute_chess_move(chess.Move.from_uci("d7d5"))
    assert loop.execute_chess_move(chess.Move.from_uci("e4d5"))

    board = engine.get_board_state()
    assert board.piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.WHITE)
    attacker = registry.get_piece_at(chess.D5)
    assert attacker is not None
    assert np.linalg.norm(env.get_body_pos(attacker.body_name)[:2] - mapper.square_to_world(chess.D5)[:2]) < loop.translator.waypoint.position_tolerance * 2
    captured = registry.get_all_captured()
    assert len(captured) == 1
    assert captured[0].color == chess.BLACK
    assert captured[0].graveyard_slot == 0
    assert np.linalg.norm(env.get_body_pos(captured[0].body_name) - captured[0].expected_pos) < loop.translator.waypoint.position_tolerance
    assert registry.validate_sync(board) == []


def _physical_loop() -> tuple[GameLoop, ChessEngine, PhysicalPieceRegistry, MuJoCoEnv, CoordinateMapper]:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    engine = ChessEngine()
    registry = PhysicalPieceRegistry(board=engine.get_board_state())
    registry.initialize({})
    mapper = CoordinateMapper(config.board)
    translator = MoveTranslator(
        registry,
        mapper,
        SlotManager(config.board, config.pieces),
        config.waypoint,
        config.pieces,
    )
    executor = MotionExecutor(env, config.waypoint, config.arm)
    return GameLoop(engine, registry, translator, executor), engine, registry, env, mapper
