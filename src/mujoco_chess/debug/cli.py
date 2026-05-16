from __future__ import annotations

import argparse
import math
from pathlib import Path

import chess
import numpy as np

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.app.startup import bootstrap
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.chess_logic.move_selector import HeuristicMoveSelector
from mujoco_chess.debug.recovery import clear_velocities, reset_arm, reset_piece
from mujoco_chess.debug.scenarios import ReachabilityChecker, test_grasp, test_release
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.motion.waypoint import WaypointStage
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

_VALID_STAGES = [s.value for s in WaypointStage]


def _open_viewer_if_requested(env: MuJoCoEnv, use_viewer: bool) -> None:
    if use_viewer:
        try:
            env.open_viewer()
            print("MuJoCo viewer opened.")
        except Exception as exc:
            print(f"Warning: could not open viewer: {exc}")


def _run_scenario(scenario: str, use_viewer: bool = False) -> None:
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
        env.initialize_arm()
        env.step(config.waypoint.initial_settle_steps)
        _open_viewer_if_requested(env, use_viewer)
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
        env.sync_viewer()
        env.close()
    except MuJoCoUnavailableError as exc:
        print(f"MuJoCo unavailable: {exc}")


def _inspect_piece_stability(env: MuJoCoEnv, config) -> None:
    """Run per-piece stability checks and print a pass/fail report."""
    import mujoco

    board = chess.Board()
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({})

    board_surface_z = config.board.board_origin_z + config.board.board_thickness
    sink_threshold = config.health.piece_sink_threshold_m
    float_threshold = config.health.piece_float_threshold_m
    tilt_threshold = config.health.piece_tilt_threshold_deg
    vel_threshold = config.health.unexpected_velocity_threshold

    passed = 0
    failed = 0
    lines: list[str] = []

    all_body_names: list[str] = []
    for name in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, name) or ""
        if body_name.startswith("piece_"):
            all_body_names.append(body_name)

    if not all_body_names:
        print("No piece bodies found in model.")
        return

    for body_name in sorted(all_body_names):
        pos = env.get_body_pos(body_name)
        lin_vel, _ = env.get_body_vel(body_name)
        quat = env.get_body_quat(body_name)
        w, x, y, z = quat
        local_z = np.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])
        local_z /= np.linalg.norm(local_z)
        tilt_deg = math.degrees(math.acos(float(np.clip(np.dot(local_z, [0.0, 0.0, 1.0]), -1.0, 1.0))))
        speed = float(np.linalg.norm(lin_vel))

        issues: list[str] = []
        expected_z_low = board_surface_z - sink_threshold
        expected_z_high = board_surface_z + config.pieces.base_height + config.pieces.shaft_height + float_threshold
        if pos[2] < expected_z_low:
            issues.append(f"SINK z={pos[2]:.4f}<{expected_z_low:.4f}")
        if pos[2] > expected_z_high:
            issues.append(f"FLOAT z={pos[2]:.4f}>{expected_z_high:.4f}")
        if tilt_deg > tilt_threshold:
            issues.append(f"TILT {tilt_deg:.1f}deg>{tilt_threshold}deg")
        if speed > vel_threshold:
            issues.append(f"VEL {speed:.4f}>{vel_threshold}")

        status = "PASS" if not issues else "FAIL"
        detail = "  ".join(issues) if issues else ""
        line = f"  {status} {body_name:<35} z={pos[2]:.4f}  tilt={tilt_deg:.1f}deg  vel={speed:.4f}  {detail}"
        lines.append(line)
        if issues:
            failed += 1
        else:
            passed += 1

    for line in lines:
        print(line)
    print(f"\nResult: {passed} passed, {failed} failed (of {len(all_body_names)} pieces)")

    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "piece_stability_report.txt"
    report_path.write_text("\n".join(lines) + f"\n\nResult: {passed} passed, {failed} failed\n", encoding="utf-8")
    print(f"Saved report: {report_path}")


def _run_waypoint_stage(env: MuJoCoEnv, config, stage_name: str, square: str | None, dest_square: str | None) -> None:
    """Execute a single named waypoint stage against the live simulation."""
    executor = MotionExecutor(env, config.waypoint, config.arm)
    mapper = CoordinateMapper(config.board)

    try:
        stage = WaypointStage(stage_name)
    except ValueError:
        print(f"Unknown stage '{stage_name}'. Valid stages: {', '.join(_VALID_STAGES)}")
        return

    home = np.array(config.arm.home_position, dtype=float)

    if stage in (WaypointStage.MOVE_TO_HOME, WaypointStage.RETURN_HOME):
        print(f"Moving arm to home position {home}...")
        result = executor.move_to(home)
        print(f"MOVE_TO_HOME: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}  final_vel={result.final_vel:.4f}")
        return

    if square is None:
        print(f"Stage '{stage_name}' requires --square (e.g. --square e2)")
        return

    try:
        sq = chess.parse_square(square)
    except ValueError:
        print(f"Invalid square '{square}'")
        return

    center = mapper.square_to_world(sq)
    hover_pos = center.copy()
    hover_pos[2] += config.waypoint.safe_hover_clearance
    grasp_pos = center.copy()
    grasp_pos[2] += config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.grasp_z_offset

    from mujoco_chess.motion.gripper import GripperController
    gripper = GripperController(env, config.arm)

    if stage == WaypointStage.MOVE_TO_SOURCE_HOVER:
        print(f"Moving to hover above {square} {hover_pos}...")
        result = executor.move_to(hover_pos)
        print(f"MOVE_TO_SOURCE_HOVER: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    elif stage == WaypointStage.DESCEND_TO_GRASP:
        print(f"Descending to grasp position above {square} {grasp_pos}...")
        result = executor.move_to(grasp_pos)
        print(f"DESCEND_TO_GRASP: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    elif stage == WaypointStage.CLOSE_GRIPPER:
        print("Closing gripper (grasp)...")
        gripper.command_grasp()
        env.step(config.waypoint.gripper_settle_steps)
        print("CLOSE_GRIPPER: done")

    elif stage == WaypointStage.ASCEND_WITH_PIECE:
        print(f"Ascending with piece to hover above {square}...")
        result = executor.move_to(hover_pos)
        print(f"ASCEND_WITH_PIECE: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    elif stage == WaypointStage.OPEN_GRIPPER:
        print("Opening gripper...")
        gripper.command_open()
        env.step(config.waypoint.gripper_settle_steps)
        print("OPEN_GRIPPER: done")

    elif stage == WaypointStage.MOVE_TO_DESTINATION_HOVER:
        if dest_square is None:
            print("MOVE_TO_DESTINATION_HOVER requires --dest (destination square, e.g. --dest e4)")
            return
        try:
            dest_sq = chess.parse_square(dest_square)
        except ValueError:
            print(f"Invalid destination square '{dest_square}'")
            return
        dest_center = mapper.square_to_world(dest_sq)
        dest_hover = dest_center.copy()
        dest_hover[2] += config.waypoint.safe_hover_clearance
        print(f"Moving to hover above {dest_square} {dest_hover}...")
        result = executor.move_to(dest_hover)
        print(f"MOVE_TO_DESTINATION_HOVER: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    elif stage == WaypointStage.DESCEND_TO_RELEASE:
        release_pos = center.copy()
        release_pos[2] += config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.release_z_offset
        print(f"Descending to release position above {square} {release_pos}...")
        result = executor.move_to(release_pos)
        print(f"DESCEND_TO_RELEASE: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    elif stage == WaypointStage.ASCEND_AFTER_RELEASE:
        print(f"Ascending after release to hover above {square}...")
        result = executor.move_to(hover_pos)
        print(f"ASCEND_AFTER_RELEASE: {'PASS' if result.success else 'FAIL'}  steps={result.steps_taken}")

    env.sync_viewer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mujoco-chess-debug",
        description=(
            "Debug CLI for the MuJoCo robotic chess simulation.\n\n"
            "Most commands load the full MuJoCo environment, run the settle phase,\n"
            "and then execute the requested operation. Use --viewer to open the\n"
            "passive MuJoCo 3D viewer during any command.\n\n"
            "Examples:\n"
            "  mujoco-chess-debug run-env --viewer\n"
            "  mujoco-chess-debug run-move --move e2e4 --viewer\n"
            "  mujoco-chess-debug run-random-game --max-moves 40 --viewer\n"
            "  mujoco-chess-debug run-scenario castling --viewer\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, help="Available debug commands")

    # run-env
    run_env_p = sub.add_parser(
        "run-env",
        help="Load the environment, settle pieces, and (optionally) keep the viewer open.",
        description=(
            "Loads the generated MuJoCo XML, initialises the Fetch arm, runs the settle phase,\n"
            "prints a brief status report, and keeps the viewer open until you close it.\n"
            "Use this to visually verify the board layout, piece positions, and arm position."
        ),
    )
    run_env_p.add_argument("--viewer", action="store_true", default=True, help="Open the MuJoCo 3D viewer (default: True)")
    run_env_p.add_argument("--no-viewer", action="store_false", dest="viewer", help="Skip opening the viewer")

    # move-ee-to
    move_ee_p = sub.add_parser(
        "move-ee-to",
        help="Move the end-effector to a specific world coordinate and report success.",
        description=(
            "Commands the arm to move its end-effector to (x, y, z) in world coordinates.\n"
            "Prints success/failure, step count, final position, and final linear velocity."
        ),
    )
    move_ee_p.add_argument("--x", type=float, required=True, help="Target X (metres)")
    move_ee_p.add_argument("--y", type=float, required=True, help="Target Y (metres)")
    move_ee_p.add_argument("--z", type=float, required=True, help="Target Z (metres)")
    move_ee_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer while moving")

    # inspect-piece-stability
    ips_p = sub.add_parser(
        "inspect-piece-stability",
        help="Check every piece for sinking, floating, tilting, or excess velocity after settle.",
        description=(
            "After the settle phase, checks all piece bodies in the model for:\n"
            "  - Z position within [board_z - sink_threshold, board_z + piece_height + float_threshold]\n"
            "  - Tilt angle (from vertical) below piece_tilt_threshold_deg\n"
            "  - Linear velocity below unexpected_velocity_threshold\n"
            "Prints a per-piece PASS/FAIL table and saves a report to logs/."
        ),
    )
    ips_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during check")

    # check-reachability
    reach_p = sub.add_parser(
        "check-reachability",
        help="Verify the Fetch arm can reach all 64 board squares.",
        description=(
            "Level 1: moves the end-effector to hover height above every board square\n"
            "         in a snake scan (a1→h1, h2→a2, ...) and checks position error.\n"
            "Level 2: additionally checks grasp height for a sample of corner/centre squares.\n"
            "Saves a full report to logs/reachability_report.txt."
        ),
    )
    reach_p.add_argument("--level", type=int, choices=(1, 2), default=1, help="Reachability level (1=hover, 2=hover+grasp)")
    reach_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during check")

    # test-grasp
    grasp_p = sub.add_parser(
        "test-grasp",
        help="Attempt to grasp the piece on a given square and verify contact.",
        description=(
            "Moves the arm to hover, descends to grasp height, closes the gripper,\n"
            "checks grasp contact health, then lifts. Reports success/failure and lift delta."
        ),
    )
    grasp_p.add_argument("--square", required=True, help="Board square with a starting piece (e.g. e2)")
    grasp_p.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the test")
    grasp_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during test")

    # test-release
    release_p = sub.add_parser(
        "test-release",
        help="Grasp the piece on --source, carry it to --square, release, and verify placement.",
        description=(
            "Full grasp-carry-release cycle: grasps the piece at --source (default e2),\n"
            "carries it to --square, releases, ascends, and checks drift/tilt/velocity."
        ),
    )
    release_p.add_argument("--square", required=True, help="Destination square (e.g. e4)")
    release_p.add_argument("--source", default="e2", help="Source square to pick from (default: e2)")
    release_p.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the test")
    release_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during test")

    # run-waypoint-stage
    stage_p = sub.add_parser(
        "run-waypoint-stage",
        help="Execute a single named waypoint stage (arm motion primitive).",
        description=(
            "Runs one stage of the waypoint algorithm in isolation. Useful for\n"
            "diagnosing arm motion without running a full move.\n\n"
            f"Valid stages: {', '.join(_VALID_STAGES)}\n\n"
            "Examples:\n"
            "  run-waypoint-stage --stage MOVE_TO_HOME\n"
            "  run-waypoint-stage --stage MOVE_TO_SOURCE_HOVER --square e2\n"
            "  run-waypoint-stage --stage DESCEND_TO_GRASP --square e2\n"
            "  run-waypoint-stage --stage MOVE_TO_DESTINATION_HOVER --square e2 --dest e4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stage_p.add_argument("--stage", required=True, choices=_VALID_STAGES, help="Waypoint stage name")
    stage_p.add_argument("--square", default=None, help="Source square (required for most stages)")
    stage_p.add_argument("--dest", default=None, help="Destination square (required for MOVE_TO_DESTINATION_HOVER)")
    stage_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during stage")

    # run-move
    run_move_p = sub.add_parser(
        "run-move",
        help="Execute a normal board-to-board move through the full game pipeline.",
        description=(
            "Executes a UCI move (e.g. e2e4) through the move translator, physical executor,\n"
            "and registry/board commit path — the same as a real game move.\n"
            "The starting position is the standard chess starting position."
        ),
    )
    run_move_p.add_argument("--move", required=True, help="UCI move string (e.g. e2e4, e1g1 for castling)")
    run_move_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during move")

    # run-scenario
    scenario_p = sub.add_parser(
        "run-scenario",
        help="Run a named multi-move scenario (castling, en-passant, promotion).",
        description=(
            "Runs a predefined scenario from a specific starting FEN:\n"
            "  castling    — kingside (white), kingside (black), queenside (white)\n"
            "  en-passant  — e5xd6 en passant capture\n"
            "  promotion   — e7e8q queen promotion\n"
        ),
    )
    scenario_p.add_argument("scenario", choices=list(_SCENARIO_FENS.keys()), help="Scenario name")
    scenario_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during scenario")

    # run-random-game
    rg_p = sub.add_parser(
        "run-random-game",
        help="Run a full AI-vs-AI game using HeuristicMoveSelector.",
        description=(
            "Runs a complete chess game with both sides controlled by HeuristicMoveSelector\n"
            "against the real MuJoCo simulation. Prints the result (1-0, 0-1, 1/2-1/2)\n"
            "or None if max_moves was reached."
        ),
    )
    rg_p.add_argument("--max-moves", type=int, default=20, help="Maximum moves before stopping (default: 20)")
    rg_p.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer during game")

    # reset-piece
    rp_p = sub.add_parser(
        "reset-piece",
        help="Teleport a piece body back to its last recorded expected position.",
        description=(
            "Uses set_body_freejoint_pos() (debug/recovery only) to teleport the named\n"
            "piece body back to its last known expected position in the registry.\n"
            "Steps the simulation 100 frames after teleport."
        ),
    )
    rp_p.add_argument("--body", required=True, dest="body_name", help="MuJoCo body name (e.g. piece_w_P_0)")

    # reset-arm
    sub.add_parser(
        "reset-arm",
        help="Move the end-effector to the home position.",
        description="Commands the arm to return to its home position. Steps 100 frames.",
    )

    # clear-velocities
    sub.add_parser(
        "clear-velocities",
        help="Zero all joint velocities in the simulation.",
        description=(
            "Sets all qvel entries to zero and calls mj_forward. Use after a crash\n"
            "or stuck state. Steps 100 frames after clearing."
        ),
    )

    args = parser.parse_args(argv)
    use_viewer = getattr(args, "viewer", False)

    # ------------------------------------------------------------------ #
    # Commands that load the environment first                             #
    # ------------------------------------------------------------------ #
    env_commands = {
        "run-env", "move-ee-to", "inspect-piece-stability",
        "check-reachability", "test-grasp", "test-release",
        "run-waypoint-stage", "run-move",
        "reset-piece", "reset-arm", "clear-velocities",
    }

    if args.command in env_commands:
        config, xml_path = bootstrap(generate_xml=True)
        try:
            env = MuJoCoEnv(xml_path, config)
            env.load()
            env.initialize_arm()
            env.step(config.waypoint.initial_settle_steps)
            print(f"Loaded environment from {xml_path}")
            print(f"board_frame position: {env.get_body_pos('board_frame')}")

            _open_viewer_if_requested(env, use_viewer)

            if args.command == "run-env":
                _inspect_piece_stability(env, config)
                if use_viewer:
                    print("Viewer open. Press Ctrl+C or close the window to exit.")
                    try:
                        while env.viewer is not None and env.viewer.is_running():
                            env.step(10)
                    except KeyboardInterrupt:
                        pass

            elif args.command == "move-ee-to":
                result = MotionExecutor(env, config.waypoint, config.arm).move_to(np.array([args.x, args.y, args.z]))
                status = "PASS" if result.success else "FAIL"
                print(f"move-ee-to: {status}  steps={result.steps_taken}  final_pos={result.final_pos}  final_vel={result.final_vel:.4f}  {result.error_message or ''}")
                env.sync_viewer()

            elif args.command == "inspect-piece-stability":
                _inspect_piece_stability(env, config)
                env.sync_viewer()

            elif args.command == "check-reachability":
                checker = ReachabilityChecker(env, config)
                results = checker.check_level1() if args.level == 1 else checker.check_level2()
                passed = sum(1 for r in results if r.passed)
                print(f"Reachability level {args.level}: {passed}/{len(results)} passed")
                failures = [r for r in results if not r.passed]
                if failures:
                    print("Failures:")
                    for r in failures[:10]:
                        print(f"  {r.target}: {r.message}")
                    if len(failures) > 10:
                        print(f"  ... and {len(failures) - 10} more")
                    print("Tip: tune configs/arm.yaml (base_x/base_y/base_z) if squares are unreachable.")
                report_path = Path(config.logging.log_dir) / "reachability_report.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    "\n".join(
                        [f"level={args.level}", f"passed={passed}", f"total={len(results)}"]
                        + [f"{r.target}: {'PASS' if r.passed else 'FAIL'} {r.message}" for r in results]
                    ),
                    encoding="utf-8",
                )
                print(f"Saved report: {report_path}")
                env.sync_viewer()

            elif args.command == "test-grasp":
                square = chess.parse_square(args.square)
                for iteration in range(args.repeat):
                    result = test_grasp(env, config, square)
                    print(f"{iteration + 1}/{args.repeat} {args.square}: {'PASS' if result.success else 'FAIL'}  {result.message}")
                env.sync_viewer()

            elif args.command == "test-release":
                square = chess.parse_square(args.square)
                source = chess.parse_square(getattr(args, "source", "e2"))
                for iteration in range(args.repeat):
                    result = test_release(env, config, square, source_square=source)
                    print(
                        f"{iteration + 1}/{args.repeat} {args.square}: {'PASS' if result.success else 'FAIL'}"
                        f"  drift={result.drift:.4f}  tilt={result.tilt_deg:.2f}deg  {result.message}"
                    )
                env.sync_viewer()

            elif args.command == "run-waypoint-stage":
                _run_waypoint_stage(env, config, args.stage, args.square, args.dest)

            elif args.command == "run-move":
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
                env.sync_viewer()

            elif args.command == "reset-piece":
                engine = ChessEngine()
                registry = PhysicalPieceRegistry(board=engine.get_board_state())
                registry.initialize({})
                try:
                    reset_piece(env, registry, args.body_name)
                    env.step(100)
                    print(f"reset-piece {args.body_name}: PASS")
                except KeyError:
                    print(f"reset-piece: body '{args.body_name}' not found in registry")
                env.sync_viewer()

            elif args.command == "reset-arm":
                reset_arm(env, config.arm.home_position)
                env.step(100)
                print("reset-arm: PASS")
                env.sync_viewer()

            elif args.command == "clear-velocities":
                clear_velocities(env)
                env.step(100)
                print("clear-velocities: PASS")
                env.sync_viewer()

            env.close()
        except MuJoCoUnavailableError as exc:
            print(f"MuJoCo unavailable: {exc}")

    elif args.command == "run-scenario":
        _run_scenario(args.scenario, use_viewer=use_viewer)

    elif args.command == "run-random-game":
        config, xml_path = bootstrap(generate_xml=True)
        try:
            env = MuJoCoEnv(xml_path, config)
            env.load()
            env.initialize_arm()
            env.step(config.waypoint.initial_settle_steps)
            _open_viewer_if_requested(env, use_viewer)
            engine = ChessEngine()
            mapper = CoordinateMapper(config.board)
            board_state = engine.get_board_state()
            registry = PhysicalPieceRegistry(board=board_state)
            registry.initialize({})
            slot_manager = SlotManager(config.board, config.pieces)
            translator = MoveTranslator(registry, mapper, slot_manager, config.waypoint, config.pieces)
            executor = MotionExecutor(env, config.waypoint, config.arm)
            move_selector = HeuristicMoveSelector()
            loop = GameLoop(
                engine,
                registry,
                translator,
                executor,
                move_selector=move_selector,
                human_color=None,
                slot_manager=slot_manager,
                mapper=mapper,
            )
            result = loop.run(max_moves=args.max_moves)
            print(f"run-random-game: result={result!r} (max_moves={args.max_moves})")
            env.sync_viewer()
            env.close()
        except MuJoCoUnavailableError as exc:
            print(f"MuJoCo unavailable: {exc}")
