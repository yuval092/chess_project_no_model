from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mujoco_chess.health.runner import CheckContext, HealthCheckContext
from mujoco_chess.motion.gripper import GripperController
from mujoco_chess.motion.waypoint import WaypointPlan, WaypointStage
from mujoco_chess.move_translation.translator import ActionType
from mujoco_chess.utils.config_loader import ArmConfig, WaypointConfig


@dataclass(frozen=True)
class StageResult:
    success: bool
    steps_taken: int
    final_pos: np.ndarray
    final_vel: float
    error_message: str | None = None


@dataclass(frozen=True)
class MotionExecutionResult:
    success: bool
    failed_stage: str | None = None
    error_message: str | None = None


class MotionExecutor:
    def __init__(self, env, config: WaypointConfig, arm_config: ArmConfig, health_runner=None) -> None:
        self.env = env
        self.config = config
        self.arm_config = arm_config
        self.health_runner = health_runner
        self.gripper = GripperController(env, arm_config)
        self._arm_initialized = False

    def move_to(
        self,
        target_pos: np.ndarray,
        timeout_steps: int | None = None,
        crane_orientation: np.ndarray | None = None,
    ) -> StageResult:
        self._ensure_arm_initialized()
        timeout = timeout_steps or self.config.stage_timeout_steps
        for step in range(timeout):
            current = self._controlled_pos()
            linear_vel, angular_vel = self._controlled_vel()
            delta = target_pos - current
            final_vel = float(np.linalg.norm(linear_vel))
            angular_speed = float(np.linalg.norm(angular_vel))
            if (
                float(np.linalg.norm(delta)) < self.config.position_tolerance
                and final_vel < self.config.velocity_threshold_linear
                and angular_speed < self.config.velocity_threshold_angular
            ):
                return StageResult(True, step, current, final_vel)
            self._step_toward(target_pos, crane_orientation=crane_orientation)
        current = self._controlled_pos()
        linear_vel, _ = self._controlled_vel()
        return StageResult(False, timeout, current, float(np.linalg.norm(linear_vel)), "Timed out before reaching target")

    def execute_action(self, action, transaction=None) -> bool:
        del transaction
        if action.action_type == ActionType.TELEPORT:
            self.env.teleport_piece(action.piece_body, action.destination_pos)
            return True
        plan = WaypointPlan(
            action.piece_body,
            action.source_pos,
            action.destination_pos,
            action.hover_z,
            action.grasp_z,
            action.release_z,
            np.array(self.arm_config.crane_down_quat),
        )
        return self.execute_plan(plan).success

    def execute_plan(self, plan: WaypointPlan) -> MotionExecutionResult:
        home = np.array(self.arm_config.home_position, dtype=float)
        source_hover = np.array([plan.source_pos[0], plan.source_pos[1], plan.hover_z])
        source_grasp = np.array([plan.source_pos[0], plan.source_pos[1], plan.grasp_z])

        self.gripper.command_open()
        self.env.step(self.config.gripper_settle_steps)
        result = self._run_move_stage(WaypointStage.MOVE_TO_HOME, home, plan.crane_orientation)
        if not result.success:
            return result
        result = self._run_move_stage(WaypointStage.MOVE_TO_SOURCE_HOVER, source_hover, plan.crane_orientation)
        if not result.success:
            return result
        result = self._run_move_stage(WaypointStage.DESCEND_TO_GRASP, source_grasp, plan.crane_orientation)
        if not result.success:
            return result

        self.gripper.command_grasp()
        self.env.step(self.config.stage_timeout_steps)
        result = self._run_health(CheckContext.POST_GRASP, WaypointStage.CLOSE_GRIPPER, plan.piece_body)
        if not result.success:
            return result

        result = self._run_hold_stage(WaypointStage.ASCEND_WITH_PIECE, source_hover, plan.crane_orientation, plan.piece_body)
        if not result.success:
            return result
        held_offset_xy = self.env.get_body_pos(plan.piece_body)[:2] - self._controlled_pos()[:2]
        destination_center = np.array(plan.destination_pos, dtype=float)
        destination_ee = destination_center.copy()
        destination_ee[:2] -= held_offset_xy
        destination_hover = np.array([destination_ee[0], destination_ee[1], plan.hover_z])
        destination_release = np.array([destination_ee[0], destination_ee[1], plan.release_z])

        for stage, target in (
            (WaypointStage.MOVE_TO_DESTINATION_HOVER, destination_hover),
            (WaypointStage.DESCEND_TO_RELEASE, destination_release),
        ):
            result = self._run_hold_stage(stage, target, plan.crane_orientation, plan.piece_body)
            if not result.success:
                return result

        self.gripper.command_open()
        self.env.step(self.config.gripper_settle_steps)

        result = self._run_move_stage(WaypointStage.ASCEND_AFTER_RELEASE, destination_hover, plan.crane_orientation)
        if not result.success:
            return result
        self.env.step(self.config.gripper_settle_steps)
        result = self._run_health(CheckContext.POST_RELEASE, WaypointStage.ASCEND_AFTER_RELEASE, plan.piece_body)
        if not result.success:
            return result
        result = self._run_move_stage(WaypointStage.RETURN_HOME, home, plan.crane_orientation)
        if not result.success:
            return result
        return MotionExecutionResult(True)

    def _run_move_stage(
        self,
        stage: WaypointStage,
        target: np.ndarray,
        crane_orientation: np.ndarray,
    ) -> MotionExecutionResult:
        result = self.move_to(target, crane_orientation=crane_orientation)
        if not result.success:
            return MotionExecutionResult(False, failed_stage=stage.value, error_message=result.error_message)
        self.env.step(self.config.settle_steps)
        return MotionExecutionResult(True)

    def _run_hold_stage(
        self,
        stage: WaypointStage,
        target: np.ndarray,
        crane_orientation: np.ndarray,
        piece_body: str,
    ) -> MotionExecutionResult:
        result = self._move_to_while_holding(target, crane_orientation)
        if not result.success:
            return MotionExecutionResult(False, failed_stage=stage.value, error_message=result.error_message)
        self.env.step(self.config.settle_steps)
        if stage == WaypointStage.ASCEND_WITH_PIECE:
            health = self._run_health(CheckContext.DURING_CARRY, stage, piece_body)
            if not health.success:
                return health
        return MotionExecutionResult(True)

    def _run_health(self, context: CheckContext, stage: WaypointStage, piece_body: str) -> MotionExecutionResult:
        if self.health_runner is None:
            return MotionExecutionResult(True)
        results = self.health_runner.run_checks(
            context,
            HealthCheckContext(active_stage=stage, target_piece_body=piece_body, held_piece_body=piece_body),
        )
        if self.health_runner.all_passed(results):
            return MotionExecutionResult(True)
        failed = [result for result in results if not result.passed]
        message = "; ".join(f"{result.check_name}: {result.message}" for result in failed)
        return MotionExecutionResult(False, failed_stage=stage.value, error_message=message)

    def _move_to_while_holding(self, target: np.ndarray, crane_orientation: np.ndarray) -> StageResult:
        timeout = self.config.stage_timeout_steps * 10
        max_step = min(self.config.transit_speed, self.config.descent_speed) / 8.0
        hold_position_tolerance = self.config.position_tolerance * 2.0
        for step in range(timeout):
            current = self._controlled_pos()
            linear_vel, angular_vel = self._controlled_vel()
            delta = target - current
            distance = float(np.linalg.norm(delta))
            final_vel = float(np.linalg.norm(linear_vel))
            angular_speed = float(np.linalg.norm(angular_vel))
            if (
                distance < hold_position_tolerance
                and final_vel < self.config.velocity_threshold_linear
                and angular_speed < self.config.velocity_threshold_angular
            ):
                return StageResult(True, step, current, final_vel)
            next_target = target if distance <= max_step else current + delta / distance * max_step
            self.gripper.command_hold()
            self.env.set_ee_target(next_target)
            self.env.set_ee_quat(crane_orientation)
            self.env.step(1)
        current = self._controlled_pos()
        linear_vel, _ = self._controlled_vel()
        return StageResult(False, timeout, current, float(np.linalg.norm(linear_vel)), "Timed out before reaching target while holding piece")

    def _step_toward(self, target: np.ndarray, crane_orientation: np.ndarray | None = None) -> None:
        current = self._controlled_pos()
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist > self.config.transit_speed:
            next_target = current + delta / dist * self.config.transit_speed
        else:
            next_target = target
        self.env.set_ee_target(next_target)
        if hasattr(self.env, "set_ee_quat"):
            orientation = crane_orientation if crane_orientation is not None else np.array(self.arm_config.crane_down_quat)
            self.env.set_ee_quat(orientation)
        self.env.step(1)

    def _controlled_pos(self) -> np.ndarray:
        return self.env.get_body_pos(self.arm_config.ee_weld_body_name)

    def _controlled_vel(self) -> tuple[np.ndarray, np.ndarray]:
        return self.env.get_body_vel(self.arm_config.ee_weld_body_name)

    def _ensure_arm_initialized(self) -> None:
        if not self._arm_initialized and hasattr(self.env, "initialize_arm"):
            self.env.initialize_arm()
            self._arm_initialized = True
