from __future__ import annotations

import numpy as np
import pytest

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.mujoco_env.env import MuJoCoEnv
import chess


def test_teleport_piece_moves_to_target() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    env.step(config.waypoint.initial_settle_steps)

    # Teleport the first white pawn to the white graveyard slot 0.
    slots = SlotManager(config.board, config.pieces)
    target = slots.graveyard_slot_world_pos(chess.WHITE, 0)

    env.teleport_piece("piece_w_P_0", target)

    actual = env.get_body_pos("piece_w_P_0")
    assert np.allclose(actual, target, atol=1e-6), f"expected {target}, got {actual}"


def test_teleport_piece_zeroes_velocity() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    env.step(config.waypoint.initial_settle_steps)

    slots = SlotManager(config.board, config.pieces)
    target = slots.graveyard_slot_world_pos(chess.WHITE, 1)

    env.teleport_piece("piece_w_P_1", target)

    # Run one step and verify the piece stays in place (velocity was zeroed).
    env.step(1)
    actual = env.get_body_pos("piece_w_P_1")
    assert np.allclose(actual, target, atol=0.005), f"piece drifted after teleport: {actual}"


def test_teleport_piece_raises_for_non_freejoint_body() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()

    with pytest.raises(ValueError, match="No freejoint"):
        env.teleport_piece("board_frame", np.array([0.0, 0.0, 1.0]))
