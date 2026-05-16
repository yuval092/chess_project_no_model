from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class HealthCheckResult:
    check_name: str
    passed: bool
    message: str
    details: dict


class HealthCheckFailureError(RuntimeError):
    pass


def pass_result(name: str, message: str = "ok", details: dict | None = None) -> HealthCheckResult:
    return HealthCheckResult(name, True, message, details or {})


def fail_result(name: str, message: str, details: dict | None = None) -> HealthCheckResult:
    return HealthCheckResult(name, False, message, details or {})


def check_piece_not_drifted(env, registry, config, hc_context=None, **kwargs) -> HealthCheckResult:
    threshold = config.health.piece_drift_threshold_m if hasattr(config, "health") else config.piece_drift_threshold_m
    ignored = getattr(hc_context, "ignored_bodies_for_drift", set()) if hc_context else set()
    overrides = getattr(hc_context, "expected_positions_override", {}) if hc_context else {}
    for record in registry.records.values():
        if record.body_name in ignored:
            continue
        expected = overrides.get(record.body_name, record.expected_pos)
        actual = env.get_body_pos(record.body_name)
        drift = float(np.linalg.norm(actual - expected))
        if drift > threshold:
            return fail_result("check_piece_not_drifted", "piece drift exceeded threshold", {"body": record.body_name, "drift": drift, "threshold": threshold})
    return pass_result("check_piece_not_drifted")


def check_piece_not_tilted(env, registry, config, **kwargs) -> HealthCheckResult:
    threshold = config.health.piece_tilt_threshold_deg if hasattr(config, "health") else config.piece_tilt_threshold_deg
    for record in registry.records.values():
        quat = env.get_body_quat(record.body_name)
        world_z = _rotate_vector_by_quat(np.array([0.0, 0.0, 1.0]), quat)
        cos_angle = float(np.clip(np.dot(world_z, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        tilt = float(np.degrees(np.arccos(cos_angle)))
        if tilt > threshold:
            return fail_result("check_piece_not_tilted", "piece tilt exceeded threshold", {"body": record.body_name, "tilt_deg": tilt, "threshold": threshold})
    return pass_result("check_piece_not_tilted")


def check_piece_not_sunk(env, registry, config, **kwargs) -> HealthCheckResult:
    board_surface_z = config.board.board_surface_z
    threshold = config.health.piece_sink_threshold_m
    for record in registry.records.values():
        z = float(env.get_body_pos(record.body_name)[2])
        if z < board_surface_z - threshold:
            return fail_result("check_piece_not_sunk", "piece below board surface threshold", {"body": record.body_name, "z": z, "threshold_z": board_surface_z - threshold})
    return pass_result("check_piece_not_sunk")


def check_piece_not_floating(env, registry, config, **kwargs) -> HealthCheckResult:
    max_z = config.board.board_surface_z + config.pieces.base_height + config.health.piece_float_threshold_m
    for record in registry.records.values():
        z = float(env.get_body_pos(record.body_name)[2])
        if z > max_z:
            return fail_result("check_piece_not_floating", "piece above float threshold", {"body": record.body_name, "z": z, "threshold_z": max_z})
    return pass_result("check_piece_not_floating")


def check_piece_not_fallen(env, registry, config, **kwargs) -> HealthCheckResult:
    min_z = config.board.table_height - 0.05
    for record in registry.records.values():
        z = float(env.get_body_pos(record.body_name)[2])
        if z < min_z:
            return fail_result("check_piece_not_fallen", "piece below table fall threshold", {"body": record.body_name, "z": z, "threshold_z": min_z})
    return pass_result("check_piece_not_fallen")


def check_no_unexpected_velocity(env, registry, config, **kwargs) -> HealthCheckResult:
    threshold = config.health.unexpected_velocity_threshold
    for record in registry.records.values():
        linear, angular = env.get_body_vel(record.body_name)
        speed = float(np.linalg.norm(linear))
        angular_speed = float(np.linalg.norm(angular))
        if speed > threshold or angular_speed > threshold:
            return fail_result("check_no_unexpected_velocity", "piece velocity exceeded threshold", {"body": record.body_name, "linear": speed, "angular": angular_speed, "threshold": threshold})
    return pass_result("check_no_unexpected_velocity")


def check_no_arm_board_collision(env, registry, config, **kwargs) -> HealthCheckResult:
    del registry, config, kwargs
    for contact in env.get_contacts():
        names = {contact.body1_name, contact.body2_name, contact.geom1_name, contact.geom2_name}
        has_arm = any(name.startswith("robot0:") for name in names)
        has_board = any(name.startswith("board") or name.startswith("square_") for name in names)
        if has_arm and has_board:
            return fail_result("check_no_arm_board_collision", "arm-board contact detected", contact.__dict__)
    return pass_result("check_no_arm_board_collision")


def check_no_arm_piece_collision(env, registry, config, target_piece_body: str | None = None, **kwargs) -> HealthCheckResult:
    del registry, config, kwargs
    for contact in env.get_contacts():
        bodies = {contact.body1_name, contact.body2_name}
        has_arm = any(name.startswith("robot0:") for name in bodies)
        piece_bodies = {name for name in bodies if name.startswith("piece")}
        if has_arm and piece_bodies and target_piece_body not in piece_bodies:
            return fail_result("check_no_arm_piece_collision", "arm contacted non-target piece", contact.__dict__)
    return pass_result("check_no_arm_piece_collision")


def check_no_piece_piece_collision(env, registry, config, **kwargs) -> HealthCheckResult:
    del registry, config, kwargs
    for contact in env.get_contacts():
        if contact.body1_name.startswith("piece") and contact.body2_name.startswith("piece"):
            return fail_result("check_no_piece_piece_collision", "piece-piece contact detected", contact.__dict__)
    return pass_result("check_no_piece_piece_collision")


def check_registry_sync(env, registry, config, logical_board=None, **kwargs) -> HealthCheckResult:
    del env, config
    errors = registry.validate_sync(logical_board) if logical_board is not None else []
    return HealthCheckResult("check_registry_sync", not errors, "registry sync" if not errors else "registry mismatch", {"errors": [e.message for e in errors]})


def check_grasp_contact(env, registry, config, hc_context=None, target_piece_body: str | None = None, **kwargs) -> HealthCheckResult:
    del registry, config, kwargs
    piece_body = target_piece_body or getattr(hc_context, "target_piece_body", None) or getattr(hc_context, "held_piece_body", None)
    if piece_body is None:
        return fail_result("check_grasp_contact", "no target piece body provided")
    contact_count = _gripper_piece_contact_count(env, piece_body)
    if contact_count == 0:
        return fail_result("check_grasp_contact", "gripper-piece contact missing", {"body": piece_body})
    return pass_result("check_grasp_contact", details={"body": piece_body, "contacts": contact_count})


def check_piece_not_slipping(env, registry, config, hc_context=None, held_piece_body: str | None = None, **kwargs) -> HealthCheckResult:
    del registry, kwargs
    piece_body = held_piece_body or getattr(hc_context, "held_piece_body", None)
    if piece_body is None:
        return fail_result("check_piece_not_slipping", "no held piece body provided")
    piece_linear, _ = env.get_body_vel(piece_body)
    gripper_linear, _ = env.get_body_vel(config.arm.ee_weld_body_name)
    relative_speed = float(np.linalg.norm(piece_linear - gripper_linear))
    threshold = config.health.slip_velocity_threshold
    if relative_speed > threshold:
        return fail_result("check_piece_not_slipping", "held piece slipping", {"body": piece_body, "relative_speed": relative_speed, "threshold": threshold})
    return pass_result("check_piece_not_slipping", details={"body": piece_body, "relative_speed": relative_speed})


def check_piece_released(env, registry, config, hc_context=None, held_piece_body: str | None = None, **kwargs) -> HealthCheckResult:
    del registry, kwargs
    piece_body = held_piece_body or getattr(hc_context, "held_piece_body", None)
    if piece_body is None:
        return fail_result("check_piece_released", "no released piece body provided")
    contact_count = _gripper_piece_contact_count(env, piece_body)
    linear, _ = env.get_body_vel(piece_body)
    speed = float(np.linalg.norm(linear))
    threshold = config.waypoint.released_piece_velocity_threshold
    if contact_count > 0:
        return fail_result("check_piece_released", "gripper-piece contact remains after release", {"body": piece_body, "contacts": contact_count})
    if speed > threshold:
        return fail_result("check_piece_released", "released piece velocity exceeded threshold", {"body": piece_body, "linear": speed, "threshold": threshold})
    return pass_result("check_piece_released", details={"body": piece_body, "linear": speed})


def _gripper_piece_contact_count(env, piece_body: str) -> int:
    count = 0
    for contact in env.get_contacts():
        bodies = {contact.body1_name, contact.body2_name}
        if piece_body in bodies and any(body.startswith("robot0:") for body in bodies):
            count += 1
    return count


def _rotate_vector_by_quat(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2 * (w * uv + uuv)


def write_failure_report(results: list[HealthCheckResult], context_info: dict, log_dir: str) -> str:
    failure_dir = Path(log_dir) / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    path = failure_dir / f"failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failed_checks": [r.__dict__ for r in results if not r.passed],
        **context_info,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)
