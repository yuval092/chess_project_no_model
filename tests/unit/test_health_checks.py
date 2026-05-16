from __future__ import annotations

import chess
import numpy as np
import pytest

from mujoco_chess.health.checks import (
    HealthCheckFailureError,
    HealthCheckResult,
    check_grasp_contact,
    check_no_unexpected_velocity,
    check_piece_released,
    check_piece_not_drifted,
    check_piece_not_floating,
    check_piece_not_slipping,
    check_piece_not_sunk,
    check_piece_not_tilted,
)
from mujoco_chess.health.runner import HealthCheckContext, HealthCheckRunner
from mujoco_chess.mujoco_env.env import ContactInfo
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.utils.config_loader import load_all_configs


class FakeEnv:
    def __init__(self) -> None:
        self.pos = {"piece_w_P_0": np.array([0.0, 0.0, 0.525])}
        self.quat = {"piece_w_P_0": np.array([1.0, 0.0, 0.0, 0.0])}
        self.vel = {"piece_w_P_0": (np.zeros(3), np.zeros(3))}
        self.contacts = []

    def get_body_pos(self, body):
        return self.pos[body]

    def get_body_quat(self, body):
        return self.quat[body]

    def get_body_vel(self, body):
        return self.vel[body]

    def get_contacts(self):
        return self.contacts


def _registry() -> PhysicalPieceRegistry:
    board = chess.Board.empty()
    board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
    registry = PhysicalPieceRegistry(board=board)
    registry.initialize({"piece_w_P_0": np.array([0.0, 0.0, 0.525])})
    registry.records = {"piece_w_P_0": registry.records["piece_w_P_0"]}
    return registry


def test_piece_health_checks_pass_for_valid_state() -> None:
    config = load_all_configs()
    env = FakeEnv()
    registry = _registry()
    assert check_piece_not_drifted(env, registry, config).passed
    assert check_piece_not_tilted(env, registry, config).passed
    assert check_piece_not_sunk(env, registry, config).passed
    assert check_piece_not_floating(env, registry, config).passed
    assert check_no_unexpected_velocity(env, registry, config).passed


def test_piece_health_checks_fail_for_invalid_state() -> None:
    config = load_all_configs()
    env = FakeEnv()
    registry = _registry()
    env.pos["piece_w_P_0"] = np.array([1.0, 0.0, 0.525])
    assert not check_piece_not_drifted(env, registry, config).passed
    # sunk: below board_surface_z (0.52) minus sink_threshold (0.005) = 0.515
    env.pos["piece_w_P_0"] = np.array([0.0, 0.0, 0.50])
    assert not check_piece_not_sunk(env, registry, config).passed
    # floating: above board_surface_z + base_height + float_threshold = 0.52+0.01+0.02 = 0.55
    env.pos["piece_w_P_0"] = np.array([0.0, 0.0, 0.56])
    assert not check_piece_not_floating(env, registry, config).passed
    env.pos["piece_w_P_0"] = np.array([0.0, 0.0, 0.525])
    env.vel["piece_w_P_0"] = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
    assert not check_no_unexpected_velocity(env, registry, config).passed


def test_health_runner_failure_report(tmp_path) -> None:
    config = load_all_configs()
    logging_config = config.logging.model_copy(update={"log_dir": str(tmp_path)})
    config = type(config)(**{**config.__dict__, "logging": logging_config})
    runner = HealthCheckRunner(config=config)
    results = [HealthCheckResult("x", False, "bad", {})]
    assert not runner.all_passed(results)
    with pytest.raises(HealthCheckFailureError):
        runner.handle_failure(results, {"chess_move": "e2e4"})
    assert list((tmp_path / "failures").glob("failure_*.json"))


def test_grasp_and_release_health_checks() -> None:
    config = load_all_configs()
    env = FakeEnv()
    registry = _registry()
    env.pos[config.arm.ee_weld_body_name] = np.array([0.0, 0.0, 0.8])
    env.vel[config.arm.ee_weld_body_name] = (np.zeros(3), np.zeros(3))

    ctx = HealthCheckContext(target_piece_body="piece_w_P_0", held_piece_body="piece_w_P_0")
    assert not check_grasp_contact(env, registry, config, hc_context=ctx).passed

    env.contacts = [ContactInfo("shaft_piece_w_P_0", "robot0:l_gripper_finger_link", "piece_w_P_0", "robot0:l_gripper_finger_link", 1.0)]
    assert check_grasp_contact(env, registry, config, hc_context=ctx).passed
    assert check_piece_not_slipping(env, registry, config, hc_context=ctx).passed

    env.vel["piece_w_P_0"] = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
    assert not check_piece_not_slipping(env, registry, config, hc_context=ctx).passed

    env.vel["piece_w_P_0"] = (np.zeros(3), np.zeros(3))
    assert not check_piece_released(env, registry, config, hc_context=ctx).passed
    env.contacts = []
    assert check_piece_released(env, registry, config, hc_context=ctx).passed
