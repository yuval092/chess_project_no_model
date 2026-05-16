from __future__ import annotations

import chess
import pytest

from mujoco_chess.board.slot_manager import SlotAllocationError, SlotManager
from mujoco_chess.utils.config_loader import load_all_configs


def test_graveyard_allocates_per_color_and_limits() -> None:
    config = load_all_configs()
    slots = SlotManager(config.board, config.pieces)
    assert [slots.allocate_graveyard_slot(chess.WHITE) for _ in range(3)] == [0, 1, 2]
    assert slots.allocate_graveyard_slot(chess.BLACK) == 0
    for _ in range(13):
        slots.allocate_graveyard_slot(chess.WHITE)
    with pytest.raises(SlotAllocationError):
        slots.allocate_graveyard_slot(chess.WHITE)


def test_slot_positions_and_reset() -> None:
    config = load_all_configs()
    slots = SlotManager(config.board, config.pieces)
    assert not (slots.graveyard_slot_world_pos(chess.WHITE, 0) == slots.graveyard_slot_world_pos(chess.BLACK, 0)).all()
    assert not (slots.pawn_storage_slot_world_pos(chess.WHITE, 0) == slots.graveyard_slot_world_pos(chess.WHITE, 0)).all()
    reserve = slots.get_reserve_slot(chess.WHITE, chess.QUEEN)
    assert reserve == 0
    slots.mark_reserve_used(chess.WHITE, chess.QUEEN, reserve)
    assert slots.get_reserve_slot(chess.WHITE, chess.QUEEN) == 1
    slots.reset()
    assert slots.get_reserve_slot(chess.WHITE, chess.QUEEN) == 0
