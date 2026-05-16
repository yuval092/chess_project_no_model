from __future__ import annotations

import chess
import pytest

from mujoco_chess.ui.event_handler import EventHandler, MoveRequest


def test_pixel_to_square_a1() -> None:
    handler = EventHandler(square_px=64)
    # A1 is file=0, rank=0. In screen coords: x in [0,64), y in [7*64, 8*64)
    sq = handler.pixel_to_square(0, 7 * 64)
    assert sq == chess.A1


def test_pixel_to_square_h8() -> None:
    handler = EventHandler(square_px=64)
    # H8 is file=7, rank=7. In screen coords: x in [7*64, 8*64), y in [0, 64)
    sq = handler.pixel_to_square(7 * 64, 0)
    assert sq == chess.H8


def test_pixel_to_square_out_of_bounds() -> None:
    handler = EventHandler(square_px=64)
    assert handler.pixel_to_square(-1, 0) is None
    assert handler.pixel_to_square(0, -1) is None
    assert handler.pixel_to_square(8 * 64, 0) is None
    assert handler.pixel_to_square(0, 8 * 64) is None


def test_click_two_squares_returns_move_request() -> None:
    handler = EventHandler(square_px=64)
    # Click E2 (file=4, rank=1) → x=4*64, y=6*64
    result1 = handler.click_square(chess.E2)
    assert result1 is None
    assert handler.selected == chess.E2

    # Click E4 (file=4, rank=3) → should return MoveRequest
    result2 = handler.click_square(chess.E4)
    assert isinstance(result2, MoveRequest)
    assert result2.from_sq == chess.E2
    assert result2.to_sq == chess.E4
    assert result2.promotion is None


def test_click_same_square_twice_reselects() -> None:
    handler = EventHandler(square_px=64)
    # Click E2 first time → selects
    result1 = handler.click_square(chess.E2)
    assert result1 is None
    assert handler.selected == chess.E2

    # Click E2 again → deselects (None returned, selected is cleared)
    result2 = handler.click_square(chess.E2)
    assert result2 is None
    assert handler.selected is None


def test_promotion_flag() -> None:
    handler = EventHandler(square_px=64)
    # Simulate a promotion scenario: pawn moves from E7 to E8
    handler.begin_promotion(chess.E7, chess.E8)
    assert handler.promotion_pending is True
    assert handler.promotion_square == chess.E8

    # Complete with queen promotion
    req = handler.set_promotion(chess.QUEEN)
    assert isinstance(req, MoveRequest)
    assert req.from_sq == chess.E7
    assert req.to_sq == chess.E8
    assert req.promotion == chess.QUEEN

    # After completion, promotion state is cleared
    assert handler.promotion_pending is False
    assert handler.promotion_square is None
