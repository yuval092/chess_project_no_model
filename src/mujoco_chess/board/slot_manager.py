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

    # Z height for all off-board storage areas: on the floor, not on the table.
    _FLOOR_Z: float = 0.01

    def graveyard_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray:
        self._validate_index(slot_index, 16)
        spacing = self.board.graveyard_slot_spacing
        col = slot_index % 2
        row = slot_index // 2
        # Graveyards sit on the floor to the left (white) and right (black) of the table.
        # The table x half-extent matches the computation in xml_generator._add_table().
        border = self.board.square_size / 2
        board_half_x = self.board.files * self.board.square_size / 2
        table_half_x = board_half_x + border + 0.08  # 8 cm clearance, matches xml_generator
        gap = self.board.graveyard_x_gap
        if color == chess.WHITE:
            # Left (x-) side: col 0 closest to table, col 1 one step further left.
            x = -(table_half_x + gap + col * spacing)
        else:
            # Right (x+) side: col 0 closest to table, col 1 one step further right.
            x = table_half_x + gap + col * spacing
        y = self.board.board_origin_y + row * spacing
        return np.array([x, y, self._FLOOR_Z], dtype=float)

    def allocate_pawn_storage_slot(self, color: chess.Color) -> int:
        return self._allocate(self._pawn_storage_used[color], 8, "pawn storage")

    def free_pawn_storage_slot(self, color: chess.Color, slot_index: int) -> None:
        self._pawn_storage_used[color].discard(slot_index)

    def pawn_storage_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray:
        self._validate_index(slot_index, 8)
        spacing = self.board.pawn_storage_slot_spacing
        total_slots = 16  # 8 white + 8 black
        color_offset = 0 if color == chess.WHITE else 8
        col = color_offset + slot_index
        board_width = self.board.files * self.board.square_size
        x0 = self.board.board_origin_x + (board_width - (total_slots - 1) * spacing) / 2
        # Pawn storage is on the floor south of the board (y- direction, away from arm).
        y = self.board.board_origin_y - self.board.pawn_storage_front_offset
        return np.array([x0 + col * spacing, y, self._FLOOR_Z], dtype=float)

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
        # Reserve area is on the floor south of the board (y- direction, away from arm).
        y = self.board.board_origin_y - self.board.reserve_area_offset_y
        return np.array([x0 + column * spacing, y, self._FLOOR_Z], dtype=float)

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
