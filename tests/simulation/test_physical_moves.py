from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.debug.scenarios import test_grasp as run_grasp_scenario
from mujoco_chess.debug.scenarios import test_release as run_release_scenario
from mujoco_chess.health.checks import check_grasp_contact, check_piece_not_slipping, check_piece_released
from mujoco_chess.health.runner import CheckContext, HealthCheckRunner
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.motion.waypoint import WaypointPlan
from mujoco_chess.mujoco_env.env import MuJoCoEnv


def test_move_to_reaches_board_target() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    mapper = CoordinateMapper(config.board)
    target = mapper.square_to_world(chess.D4)
    target[2] += config.waypoint.safe_hover_clearance

    result = MotionExecutor(env, config.waypoint, config.arm).move_to(target, timeout_steps=4000)

    assert result.success, result.error_message
    assert result.steps_taken < 4000
    assert result.final_vel < config.waypoint.velocity_threshold_linear
    assert np.linalg.norm(env.get_body_pos(config.arm.ee_weld_body_name) - target) < config.waypoint.position_tolerance


def test_move_to_returns_false_for_unreachable_target() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    target = np.array([3.0, 3.0, 3.0])

    result = MotionExecutor(env, config.waypoint, config.arm).move_to(target, timeout_steps=50)

    assert not result.success
    assert result.error_message is not None


def test_grasp_lifts_starting_pawn() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()

    result = run_grasp_scenario(env, config, chess.E2)

    assert result.success, result.message
    assert result.contact_count > 0
    assert result.lifted_z > result.initial_z + config.pieces.base_height * 3
    assert result.lifted_z > config.board.board_surface_z + config.pieces.base_height


def test_release_places_pawn_on_target_square() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    initialize_calls = 0
    original_initialize_arm = env.initialize_arm

    def count_initialize_arm() -> None:
        nonlocal initialize_calls
        initialize_calls += 1
        original_initialize_arm()

    env.initialize_arm = count_initialize_arm

    result = run_release_scenario(env, config, chess.E4)

    assert result.success, result.message
    assert initialize_calls == 1
    assert result.drift < config.health.piece_drift_threshold_m
    assert result.tilt_deg < config.health.piece_tilt_threshold_deg
    assert result.linear_speed < config.waypoint.released_piece_velocity_threshold


def test_execute_plan_moves_one_piece_e2_to_e4() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    mapper = CoordinateMapper(config.board)
    source = mapper.square_to_world(chess.E2)
    destination = mapper.square_to_world(chess.E4)
    plan = WaypointPlan(
        piece_body="piece_w_P_3",
        source_pos=source,
        destination_pos=destination,
        hover_z=source[2] + config.waypoint.safe_hover_clearance,
        grasp_z=source[2] + config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.grasp_z_offset,
        release_z=destination[2] + config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.release_z_offset,
        crane_orientation=np.array(config.arm.crane_down_quat),
    )
    health_runner = HealthCheckRunner(env=env, registry=None, config=config)
    health_runner.register(CheckContext.POST_GRASP, check_grasp_contact)
    health_runner.register(CheckContext.DURING_CARRY, check_grasp_contact)
    health_runner.register(CheckContext.DURING_CARRY, check_piece_not_slipping)
    health_runner.register(CheckContext.POST_RELEASE, check_piece_released)

    result = MotionExecutor(env, config.waypoint, config.arm, health_runner=health_runner).execute_plan(plan)
    piece_pos = env.get_body_pos(plan.piece_body)
    piece_linear, _ = env.get_body_vel(plan.piece_body)

    assert result.success, result.error_message
    assert np.linalg.norm(piece_pos[:2] - destination[:2]) < config.health.piece_drift_threshold_m
    assert np.linalg.norm(piece_linear) < config.waypoint.released_piece_velocity_threshold
