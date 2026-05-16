from __future__ import annotations

import logging
import queue

import chess

from mujoco_chess.app.game_loop import GameLoop
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.chess_logic.move_selector import HeuristicMoveSelector, RandomMoveSelector
from mujoco_chess.health.checks import (
    check_piece_not_drifted,
    check_piece_not_fallen,
    check_piece_not_floating,
    check_piece_not_sunk,
    check_piece_not_tilted,
)
from mujoco_chess.health.runner import CheckContext, HealthCheckRunner
from mujoco_chess.logging_setup.setup import setup_logging
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.move_translation.translator import MoveTranslator
from mujoco_chess.mujoco_env.env import MuJoCoEnv, MuJoCoUnavailableError
from mujoco_chess.mujoco_env.placement import assert_valid_layout
from mujoco_chess.mujoco_env.xml_generator import XMLGenerator
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.ui.board_ui import BoardUI
from mujoco_chess.utils.config_loader import load_all_configs

LOGGER = logging.getLogger(__name__)


def build_health_runner(env, registry, config) -> HealthCheckRunner:
    """Create a HealthCheckRunner with STARTUP checks registered."""
    runner = HealthCheckRunner(env=env, registry=registry, config=config)
    for check_fn in (
        check_piece_not_drifted,
        check_piece_not_tilted,
        check_piece_not_sunk,
        check_piece_not_floating,
        check_piece_not_fallen,
    ):
        runner.register(CheckContext.STARTUP, check_fn)
    return runner


def bootstrap(generate_xml: bool = True):
    config = load_all_configs()
    setup_logging(config.logging)
    assert_valid_layout(config)
    xml_path = XMLGenerator(config).generate() if generate_xml else config.root_dir / "generated" / "chess_env.xml"
    LOGGER.info("Bootstrap complete: %s", xml_path)
    return config, xml_path


def main() -> None:
    """Launch the full chess simulation: MuJoCo viewer + Pygame board UI."""
    config, xml_path = bootstrap(generate_xml=True)
    print(f"Generated MuJoCo XML: {xml_path}")

    try:
        env = MuJoCoEnv(xml_path, config)
        env.load()

        print("Initializing Fetch arm (freezing in crane pose)...")
        env.initialize_arm()

        print(f"Settling pieces ({config.waypoint.piece_settle_steps} steps)...")
        env.step(config.waypoint.piece_settle_steps)

        print(f"Moving arm to home position ({config.waypoint.arm_settle_steps} steps)...")
        env.move_arm_to_home()

        print("Opening MuJoCo passive viewer...")
        env.open_viewer()

        # Build the full game pipeline
        engine = ChessEngine()
        board_state = engine.get_board_state()
        mapper = CoordinateMapper(config.board)
        registry = PhysicalPieceRegistry(board=board_state)
        registry.initialize({})
        slot_manager = SlotManager(config.board, config.pieces)
        translator = MoveTranslator(registry, mapper, slot_manager, config.waypoint, config.pieces)
        executor = MotionExecutor(env, config.waypoint, config.arm)

        # Determine human color and AI selector from game config
        human_color = chess.WHITE if config.game.human_color == "white" else chess.BLACK
        selector: HeuristicMoveSelector | RandomMoveSelector
        if config.game.selector_type == "heuristic":
            selector = HeuristicMoveSelector()
        else:
            selector = RandomMoveSelector()

        # Launch Pygame board UI in a secondary thread
        event_queue: queue.Queue = queue.Queue()
        state_queue: queue.Queue = queue.Queue()
        board_ui = BoardUI(event_queue=event_queue, state_queue=state_queue)
        ui_thread = board_ui.start_thread()
        print("Pygame board UI started.")

        health_runner = build_health_runner(env, registry, config)

        loop = GameLoop(
            engine,
            registry,
            translator,
            executor,
            move_selector=selector,
            human_color=human_color,
            event_queue=event_queue,
            state_queue=state_queue,
            slot_manager=slot_manager,
            mapper=mapper,
            health_runner=health_runner,
        )

        color_name = "White" if human_color == chess.WHITE else "Black"
        print(f"Game started. You play {color_name}. Use the Pygame window to make moves.")
        print("Close the Pygame window or press Q to quit.")

        result = loop.run()
        print(f"Game over: {result}")

        env.close()
        ui_thread.join(timeout=2.0)

    except MuJoCoUnavailableError as exc:
        print(f"MuJoCo unavailable: {exc}")
        raise SystemExit(1) from exc
