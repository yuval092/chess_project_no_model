from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.app.startup import bootstrap
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.debug.scenarios import ReachabilityChecker, test_grasp, test_release
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.mujoco_env.env import MuJoCoEnv, MuJoCoUnavailableError
from mujoco_chess.move_translation.translator import MoveTranslator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry


_SCENARIO_FENS: dict[str, str] = {
    "castling": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "en-passant": "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "promotion": "3k4/4P3/8/8/8/8/8/4K3 w - - 0 1",
}

_SCENARIO_MOVES: dict[str, list[str]] = {
    "castling": ["e1g1", "e8g8", "e1c1"],
    "en-passant": ["e5d6"],
    "promotion": ["e7e8q"],
}


def _run_scenario(scenario: str) -> None:
    if scenario not in _SCENARIO_FENS:
        known = ", ".join(sorted(_SCENARIO_FENS))
        print(f"Unknown scenario '{scenario}'. Known: {known}")
        return
    fen = _SCENARIO_FENS[scenario]
    moves_uci = _SCENARIO_MOVES[scenario]
    config, xml_path = bootstrap(generate_xml=True)
    try:
        env = MuJoCoEnv(xml_path, config)
        env.load()
        env.step(config.waypoint.initial_settle_steps)
        print(f"Loaded environment for scenario '{scenario}'")
        board = chess.Board(fen)
        engine = ChessEngine()
        engine._board = board
        registry = PhysicalPieceRegistry(board=engine.get_board_state())
        registry.initialize({})
        mapper = CoordinateMapper(config.board)
        slot_manager = SlotManager(config.board, config.pieces)
        translator = MoveTranslator(registry, mapper, slot_manager, config.waypoint, config.pieces)
        executor = MotionExecutor(env, config.waypoint, config.arm)
        loop = GameLoop(engine, registry, translator, executor)
        for uci in moves_uci:
            move = chess.Move.from_uci(uci)
            if not engine.is_move_legal(move):
                print(f"  skip {uci}: not legal in current position")
                continue
            result = loop.execute_chess_move(move)
            status = "PASS" if result else "FAIL"
            print(f"  {uci}: {status}")
        env.close()
    except MuJoCoUnavailableError as exc:
        print(f"MuJoCo unavailable: {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mujoco-chess-debug")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-env")
    move_ee = sub.add_parser("move-ee-to")
    move_ee.add_argument("--x", type=float, required=True)
    move_ee.add_argument("--y", type=float, required=True)
    move_ee.add_argument("--z", type=float, required=True)
    sub.add_parser("inspect-piece-stability")
    reach = sub.add_parser("check-reachability")
    reach.add_argument("--level", type=int, choices=(1, 2), default=1)
    grasp = sub.add_parser("test-grasp")
    grasp.add_argument("--square", required=True)
    grasp.add_argument("--repeat", type=int, default=1)
    release = sub.add_parser("test-release")
    release.add_argument("--square", required=True)
    release.add_argument("--repeat", type=int, default=1)
    stage = sub.add_parser("run-waypoint-stage")
    stage.add_argument("--stage", required=True)
    run_move = sub.add_parser("run-move")
    run_move.add_argument("--move", required=True)
    scenario = sub.add_parser("run-scenario")
    scenario.add_argument("scenario")
    random_game = sub.add_parser("run-random-game")
    random_game.add_argument("--max-moves", type=int, default=20)
    args = parser.parse_args(argv)

    if args.command in {"run-env", "move-ee-to", "inspect-piece-stability", "check-reachability", "test-grasp", "test-release", "run-move"}:
        config, xml_path = bootstrap(generate_xml=True)
        try:
            env = MuJoCoEnv(xml_path, config)
            env.load()
            env.step(config.waypoint.initial_settle_steps)
            print(f"Loaded environment from {xml_path}")
            print(f"board_frame position: {env.get_body_pos('board_frame')}")
            if args.command == "move-ee-to":
                result = MotionExecutor(env, config.waypoint, config.arm).move_to(np.array([args.x, args.y, args.z]))
                status = "PASS" if result.success else "FAIL"
                print(f"move-ee-to: {status} steps={result.steps_taken} final_pos={result.final_pos} final_vel={result.final_vel:.4f} {result.error_message or ''}")
            if args.command == "check-reachability":
                checker = ReachabilityChecker(env, config)
                results = checker.check_level1() if args.level == 1 else checker.check_level2()
                passed = sum(1 for result in results if result.passed)
                print(f"Reachability level {args.level}: {passed}/{len(results)} passed")
                failures = [result for result in results if not result.passed]
                if failures:
                    print("First failures:")
                    for result in failures[:10]:
                        print(f"  {result.target}: {result.message}")
                    print("Suggestions: tune configs/arm.yaml base_x/base_y/base_z/base_yaw, or revise board/storage layout before waypoint milestones.")
                report_path = Path(config.logging.log_dir) / "reachability_report.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    "\n".join(
                        [
                            f"level={args.level}",
                            f"passed={passed}",
                            f"total={len(results)}",
                            *[f"{result.target}: {'PASS' if result.passed else 'FAIL'} {result.message}" for result in results],
                        ]
                    ),
                    encoding="utf-8",
                )
                print(f"Saved report: {report_path}")
            if args.command == "test-grasp":
                square = chess.parse_square(args.square)
                for iteration in range(args.repeat):
                    result = test_grasp(env, config, square)
                    print(f"{iteration + 1}/{args.repeat} {args.square}: {'PASS' if result.success else 'FAIL'} {result.message}")
            if args.command == "test-release":
                square = chess.parse_square(args.square)
                for iteration in range(args.repeat):
                    result = test_release(env, config, square)
                    print(f"{iteration + 1}/{args.repeat} {args.square}: {'PASS' if result.success else 'FAIL'} {result.message}")
            if args.command == "run-move":
                move = chess.Move.from_uci(args.move)
                engine = ChessEngine()
                registry = PhysicalPieceRegistry(board=engine.get_board_state())
                registry.initialize({})
                mapper = CoordinateMapper(config.board)
                slot_manager = SlotManager(config.board, config.pieces)
                translator = MoveTranslator(registry, mapper, slot_manager, config.waypoint, config.pieces)
                executor = MotionExecutor(env, config.waypoint, config.arm)
                result = GameLoop(engine, registry, translator, executor).execute_chess_move(move)
                print(f"run-move {args.move}: {'PASS' if result else 'FAIL'}")
            env.close()
        except MuJoCoUnavailableError as exc:
            print(f"MuJoCo unavailable: {exc}")
    elif args.command == "run-scenario":
        _run_scenario(args.scenario)
    else:
        print(f"{args.command} is registered but full execution is pending the waypoint and transaction milestones.")
