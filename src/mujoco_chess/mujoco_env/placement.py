from __future__ import annotations

from dataclasses import dataclass

import chess

from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.utils.config_loader import AppConfig


class PlacementValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bounds2D:
    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def intersects(self, other: "Bounds2D") -> bool:
        return not (
            self.max_x <= other.min_x
            or self.min_x >= other.max_x
            or self.max_y <= other.min_y
            or self.min_y >= other.max_y
        )


@dataclass(frozen=True)
class PlacementCheckResult:
    passed: bool
    errors: list[str]
    bounds: list[Bounds2D]


def compute_layout_bounds(config: AppConfig) -> list[Bounds2D]:
    board = config.board
    slots = SlotManager(config.board, config.pieces)
    bounds = [
        Bounds2D("board", board.board_origin_x, board.board_max_x, board.board_origin_y, board.board_max_y),
        _bounds_from_points("white_graveyard", [slots.graveyard_slot_world_pos(chess.WHITE, i) for i in range(16)], board.graveyard_slot_spacing),
        _bounds_from_points("black_graveyard", [slots.graveyard_slot_world_pos(chess.BLACK, i) for i in range(16)], board.graveyard_slot_spacing),
        _bounds_from_points("white_pawn_storage", [slots.pawn_storage_slot_world_pos(chess.WHITE, i) for i in range(8)], board.pawn_storage_slot_spacing),
        _bounds_from_points("black_pawn_storage", [slots.pawn_storage_slot_world_pos(chess.BLACK, i) for i in range(8)], board.pawn_storage_slot_spacing),
        _bounds_from_points(
            "promotion_reserve",
            [
                slots.reserve_slot_world_pos(color, piece_type, slot)
                for color in (chess.WHITE, chess.BLACK)
                for piece_type in SlotManager.RESERVE_TYPES
                for slot in range(2)
            ],
            board.reserve_slot_spacing,
        ),
    ]
    footprint_half = config.board.square_size * 2
    bounds.append(
        Bounds2D(
            "arm_base",
            config.arm.base_x - footprint_half,
            config.arm.base_x + footprint_half,
            config.arm.base_y - footprint_half,
            config.arm.base_y + footprint_half,
        )
    )
    return bounds


def validate_layout(config: AppConfig) -> PlacementCheckResult:
    bounds = compute_layout_bounds(config)
    errors: list[str] = []
    # Off-board slots (graveyard, pawn storage, reserve) use teleportation;
    # they don't need to be within arm reach, so arm_base overlap is not checked
    # for those areas. Physical non-overlap between areas is still enforced.
    forbidden_pairs = {
        ("board", "white_graveyard"),
        ("board", "black_graveyard"),
        ("board", "white_pawn_storage"),
        ("board", "black_pawn_storage"),
        ("board", "promotion_reserve"),
        ("arm_base", "board"),
        ("white_graveyard", "white_pawn_storage"),
        ("black_graveyard", "black_pawn_storage"),
    }
    by_name = {item.name: item for item in bounds}
    for left_name, right_name in forbidden_pairs:
        left = by_name[left_name]
        right = by_name[right_name]
        if left.intersects(right):
            errors.append(f"{left.name} intersects {right.name}")
    return PlacementCheckResult(not errors, errors, bounds)


def assert_valid_layout(config: AppConfig) -> None:
    result = validate_layout(config)
    if not result.passed:
        raise PlacementValidationError("; ".join(result.errors))


def _bounds_from_points(name: str, points, spacing: float) -> Bounds2D:
    half = spacing / 2
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Bounds2D(name, min(xs) - half, max(xs) + half, min(ys) - half, max(ys) + half)
