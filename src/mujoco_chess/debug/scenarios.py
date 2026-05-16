from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.health.checks import check_grasp_contact, check_piece_not_slipping, check_piece_released
from mujoco_chess.health.runner import CheckContext, HealthCheckContext, HealthCheckRunner
from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.motion.gripper import GripperController
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import AppConfig


@dataclass(frozen=True)
class ReachabilityResult:
    target: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class GraspScenarioResult:
    square: chess.Square
    body_name: str
    success: bool
    initial_z: float
    lifted_z: float
    contact_count: int
    message: str = ""


@dataclass(frozen=True)
class ReleaseScenarioResult:
    source_square: chess.Square
    target_square: chess.Square
    body_name: str
    success: bool
    drift: float
    tilt_deg: float
    linear_speed: float
    message: str = ""


class ReachabilityChecker:
    def __init__(self, env, config: AppConfig) -> None:
        self.env = env
        self.config = config
        self.mapper = CoordinateMapper(config.board)

    def check_level1(self) -> list[ReachabilityResult]:
        """Check all 64 board squares are reachable by the arm at hover height.

        Off-board slots (graveyard, pawn storage, reserve) use teleportation
        and are not included in this check.
        """
        results: list[ReachabilityResult] = []
        hover = self.config.waypoint.safe_hover_clearance

        self.env.initialize_arm()
        # Snake scan: even ranks go a→h, odd ranks go h→a, minimising consecutive jumps.
        first = True
        for rank in range(8):
            files = range(8) if rank % 2 == 0 else range(7, -1, -1)
            for file in files:
                square = chess.square(file, rank)
                pos = self.mapper.square_to_world(square)
                pos[2] += hover
                results.append(self._check_point(chess.square_name(square), pos, first_check=first))
                first = False

        return results

    def check_level2(self) -> list[ReachabilityResult]:
        self.env.initialize_arm()
        sample_squares = [chess.A1, chess.H1, chess.A8, chess.H8, chess.D4, chess.E5]
        results: list[ReachabilityResult] = []
        first = True
        for square in sample_squares:
            center = self.mapper.square_to_world(square)
            hover = center.copy()
            hover[2] += self.config.waypoint.safe_hover_clearance
            grasp = center.copy()
            grasp[2] += self.config.pieces.base_height + self.config.pieces.shaft_height / 2 + self.config.waypoint.grasp_z_offset
            for suffix, target in (("hover", hover), ("grasp", grasp), ("return_hover", hover)):
                results.append(self._check_point(f"{chess.square_name(square)}:{suffix}", target, first_check=first))
                first = False
        return results

    def _check_point(self, name: str, target: np.ndarray, first_check: bool = False) -> ReachabilityResult:
        steps = 3000 if first_check else 2000
        self.env.set_ee_target(target)
        self.env.step(steps)
        # Measure from the weld body (gripper_link) since that is what the mocap constraint
        # directly controls; the grip site sits 0.02 m ahead of it along local +X.
        actual = self.env.get_body_pos(self.config.arm.ee_weld_body_name)
        error = float(np.linalg.norm(actual - target))
        passed = error <= self.config.waypoint.position_tolerance
        return ReachabilityResult(name, passed, "" if passed else f"error={error:.4f}")


def test_grasp(
    env,
    config: AppConfig,
    square: chess.Square,
    executor: MotionExecutor | None = None,
) -> GraspScenarioResult:
    mapper = CoordinateMapper(config.board)
    registry = PhysicalPieceRegistry(board=chess.Board())
    registry.initialize({})
    record = registry.get_piece_at(square)
    if record is None:
        raise ValueError(f"No starting piece at {chess.square_name(square)}")

    executor = executor or MotionExecutor(env, config.waypoint, config.arm)
    gripper = GripperController(env, config.arm)
    health_runner = HealthCheckRunner(env=env, registry=registry, config=config)
    health_runner.register(CheckContext.POST_GRASP, check_grasp_contact)
    health_runner.register(CheckContext.DURING_CARRY, check_grasp_contact)
    health_runner.register(CheckContext.DURING_CARRY, check_piece_not_slipping)
    center = mapper.square_to_world(square)
    hover = center.copy()
    hover[2] += config.waypoint.safe_hover_clearance
    grasp = center.copy()
    grasp[2] += config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.grasp_z_offset

    initial_z = float(env.get_body_pos(record.body_name)[2])
    gripper.command_open()
    env.step(config.waypoint.gripper_settle_steps)
    for target in (hover, grasp):
        result = executor.move_to(target, timeout_steps=config.waypoint.stage_timeout_steps * 4)
        if not result.success:
            return GraspScenarioResult(square, record.body_name, False, initial_z, float(env.get_body_pos(record.body_name)[2]), 0, result.error_message or "motion failed")

    gripper.command_grasp()
    env.step(config.waypoint.gripper_settle_steps)
    contacts = _gripper_piece_contacts(env, record.body_name)
    post_grasp = health_runner.run_checks(
        CheckContext.POST_GRASP,
        HealthCheckContext(target_piece_body=record.body_name, held_piece_body=record.body_name),
    )
    if not health_runner.all_passed(post_grasp):
        return GraspScenarioResult(square, record.body_name, False, initial_z, float(env.get_body_pos(record.body_name)[2]), contacts, _health_message(post_grasp))
    result = executor.move_to(hover, timeout_steps=config.waypoint.stage_timeout_steps * 4)
    lifted_z = float(env.get_body_pos(record.body_name)[2])
    during_carry = health_runner.run_checks(
        CheckContext.DURING_CARRY,
        HealthCheckContext(target_piece_body=record.body_name, held_piece_body=record.body_name),
    )
    lift = lifted_z - initial_z
    success = result.success and contacts > 0 and lift > config.pieces.base_height * 3 and health_runner.all_passed(during_carry)
    message = "" if success else f"lift={lift:.4f}, contacts={contacts}, health={_health_message(during_carry)}"
    return GraspScenarioResult(square, record.body_name, success, initial_z, lifted_z, contacts, message)


def test_release(
    env,
    config: AppConfig,
    target_square: chess.Square,
    source_square: chess.Square = chess.E2,
) -> ReleaseScenarioResult:
    executor = MotionExecutor(env, config.waypoint, config.arm)
    grasp = test_grasp(env, config, source_square, executor=executor)
    if not grasp.success:
        return ReleaseScenarioResult(
            source_square,
            target_square,
            grasp.body_name,
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            grasp.message,
        )

    mapper = CoordinateMapper(config.board)
    gripper = GripperController(env, config.arm)
    health_runner = HealthCheckRunner(env=env, registry=None, config=config)
    health_runner.register(CheckContext.POST_RELEASE, check_piece_released)
    center = mapper.square_to_world(target_square)
    held_offset_xy = env.get_body_pos(grasp.body_name)[:2] - env.get_body_pos(config.arm.ee_weld_body_name)[:2]
    release_center = center.copy()
    release_center[:2] -= held_offset_xy
    hover = center.copy()
    hover[:2] = release_center[:2]
    hover[2] += config.waypoint.safe_hover_clearance
    release = release_center.copy()
    release[2] += config.pieces.base_height + config.pieces.shaft_height / 2 + config.waypoint.release_z_offset

    for target in (hover, release):
        success, message = _move_to_while_holding(env, config, gripper, target, timeout_steps=config.waypoint.stage_timeout_steps * 10)
        if not success:
            pos = env.get_body_pos(grasp.body_name)
            return ReleaseScenarioResult(
                source_square,
                target_square,
                grasp.body_name,
                False,
                float(np.linalg.norm(pos[:2] - center[:2])),
                _tilt_deg(env, grasp.body_name),
                _linear_speed(env, grasp.body_name),
                message,
            )

    gripper.command_open()
    env.step(config.waypoint.gripper_settle_steps)
    result = executor.move_to(hover, timeout_steps=config.waypoint.stage_timeout_steps * 4)
    env.step(config.waypoint.settle_steps)
    post_release = health_runner.run_checks(
        CheckContext.POST_RELEASE,
        HealthCheckContext(held_piece_body=grasp.body_name),
    )

    pos = env.get_body_pos(grasp.body_name)
    drift = float(np.linalg.norm(pos[:2] - center[:2]))
    tilt = _tilt_deg(env, grasp.body_name)
    linear_speed = _linear_speed(env, grasp.body_name)
    success = (
        result.success
        and drift < config.health.piece_drift_threshold_m
        and tilt < config.health.piece_tilt_threshold_deg
        and linear_speed < config.waypoint.released_piece_velocity_threshold
        and health_runner.all_passed(post_release)
    )
    message = "" if success else f"drift={drift:.4f}, tilt={tilt:.2f}, linear_speed={linear_speed:.4f}, health={_health_message(post_release)}"
    return ReleaseScenarioResult(source_square, target_square, grasp.body_name, success, drift, tilt, linear_speed, message)


def _gripper_piece_contacts(env, piece_body: str) -> int:
    count = 0
    for contact in env.get_contacts():
        bodies = {contact.body1_name, contact.body2_name}
        if piece_body in bodies and any(body.startswith("robot0:") for body in bodies):
            count += 1
    return count


def _health_message(results) -> str:
    failures = [result for result in results if not result.passed]
    return "; ".join(f"{result.check_name}: {result.message}" for result in failures) if failures else "ok"


def _move_to_while_holding(
    env,
    config: AppConfig,
    gripper: GripperController,
    target: np.ndarray,
    timeout_steps: int,
) -> tuple[bool, str]:
    max_step = min(config.waypoint.transit_speed, config.waypoint.descent_speed) / 8.0
    for _ in range(timeout_steps):
        current = env.get_body_pos(config.arm.ee_weld_body_name)
        linear_vel, angular_vel = env.get_body_vel(config.arm.ee_weld_body_name)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if (
            distance < config.waypoint.position_tolerance
            and float(np.linalg.norm(linear_vel)) < config.waypoint.velocity_threshold_linear
            and float(np.linalg.norm(angular_vel)) < config.waypoint.velocity_threshold_angular
        ):
            return True, ""
        next_target = target if distance <= max_step else current + delta / distance * max_step
        gripper.command_hold()
        env.set_ee_target(next_target)
        env.set_ee_quat(np.array(config.arm.crane_down_quat))
        env.step(1)
    return False, "Timed out while carrying held piece"


def _linear_speed(env, body_name: str) -> float:
    linear, _ = env.get_body_vel(body_name)
    return float(np.linalg.norm(linear))


def _tilt_deg(env, body_name: str) -> float:
    quat = env.get_body_quat(body_name)
    w, x, y, z = quat
    local_z_world = np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])
    local_z_world /= np.linalg.norm(local_z_world)
    cos_angle = float(np.clip(np.dot(local_z_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
