from __future__ import annotations

import chess
import numpy as np

from mujoco_chess.utils.config_loader import BoardConfig, PiecesConfig


class SlotAllocationError(Exception):
    pass


class SlotManager:
    RESERVE_TYPES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)

    def __init__(self, board_config: BoardConfig, pieces_config: PiecesConfig) -> None:
        self.board = board_config
        self.pieces = pieces_config
        self.reset()

    def allocate_graveyard_slot(self, color: chess.Color) -> int:
        return self._allocate(self._graveyard_used[color], 16, "graveyard")

    def free_graveyard_slot(self, color: chess.Color, slot_index: int) -> None:
        self._graveyard_used[color].discard(slot_index)

    def graveyard_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray:
        self._validate_index(slot_index, 16)
        spacing = self.board.graveyard_slot_spacing
        col = slot_index % 2
        row = slot_index // 2
        # WHITE: graveyard_white_x_offset to the left of board, BLACK: 1 square to the right.
        # y starts at board_origin_y + 0.5*spacing so the bottom row clears the arm's
        # maximum extension limit at the board-corner (x=±0.42, y=-0.28).
        x0 = self.board.board_origin_x - self.board.graveyard_white_x_offset if color == chess.WHITE else self.board.board_origin_x + 9 * self.board.square_size
        return np.array([x0 + col * spacing, self.board.board_origin_y + (row + 0.5) * spacing, self.board.board_surface_z], dtype=float)

    def allocate_pawn_storage_slot(self, color: chess.Color) -> int:
        return self._allocate(self._pawn_storage_used[color], 8, "pawn storage")

    def free_pawn_storage_slot(self, color: chess.Color, slot_index: int) -> None:
        self._pawn_storage_used[color].discard(slot_index)

    def pawn_storage_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray:
        self._validate_index(slot_index, 8)
        spacing = self.board.pawn_storage_slot_spacing
        total_slots = 16  # 8 white + 8 black share one row behind the board (positive y, near arm)
        color_offset = 0 if color == chess.WHITE else 8
        col = color_offset + slot_index
        board_width = self.board.files * self.board.square_size
        x0 = self.board.board_origin_x + (board_width - (total_slots - 1) * spacing) / 2
        y = self.board.board_max_y + self.board.pawn_storage_front_offset
        return np.array([x0 + col * spacing, y, self.board.board_surface_z], dtype=float)

    def get_reserve_slot(self, color: chess.Color, piece_type: chess.PieceType) -> int | None:
        key = (color, piece_type)
        for slot in range(2):
            if slot not in self._reserve_used[key]:
                return slot
        return None

    def mark_reserve_used(self, color: chess.Color, piece_type: chess.PieceType, slot: int) -> None:
        self._validate_index(slot, 2)
        key = (color, piece_type)
        if slot in self._reserve_used[key]:
            raise SlotAllocationError(f"Reserve slot already used: {color} {piece_type} {slot}")
        self._reserve_used[key].add(slot)

    def reserve_slot_world_pos(self, color: chess.Color, piece_type: chess.PieceType, slot: int) -> np.ndarray:
        self._validate_index(slot, 2)
        if piece_type not in self.RESERVE_TYPES:
            raise ValueError(f"Unsupported reserve piece type: {piece_type}")
        spacing = self.board.reserve_slot_spacing
        color_offset = 0 if color == chess.WHITE else len(self.RESERVE_TYPES) * 2
        type_offset = self.RESERVE_TYPES.index(piece_type) * 2
        column = color_offset + type_offset + slot
        total_columns = len(self.RESERVE_TYPES) * 2 * 2
        board_width = self.board.files * self.board.square_size
        x0 = self.board.board_origin_x + (board_width - (total_columns - 1) * spacing) / 2
        y = self.board.board_max_y + self.board.reserve_area_offset_y
        return np.array([x0 + column * spacing, y, self.board.board_surface_z], dtype=float)

    def reset(self) -> None:
        self._graveyard_used: dict[chess.Color, set[int]] = {chess.WHITE: set(), chess.BLACK: set()}
        self._pawn_storage_used: dict[chess.Color, set[int]] = {chess.WHITE: set(), chess.BLACK: set()}
        self._reserve_used: dict[tuple[chess.Color, chess.PieceType], set[int]] = {
            (color, piece_type): set()
            for color in (chess.WHITE, chess.BLACK)
            for piece_type in self.RESERVE_TYPES
        }

    def _allocate(self, used: set[int], limit: int, label: str) -> int:
        for idx in range(limit):
            if idx not in used:
                used.add(idx)
                return idx
        raise SlotAllocationError(f"No free {label} slots")

    @staticmethod
    def _validate_index(slot_index: int, limit: int) -> None:
        if not 0 <= slot_index < limit:
            raise IndexError(f"Slot index {slot_index} outside 0..{limit - 1}")
