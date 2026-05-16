from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess
import numpy as np


class PieceStatus(Enum):
    ACTIVE = "ACTIVE"
    CAPTURED = "CAPTURED"
    RESERVE = "RESERVE"
    PROMOTED = "PROMOTED"
    PROMOTION_PAWN_STORED = "PROMOTION_PAWN_STORED"


@dataclass
class PieceRecord:
    body_name: str
    color: chess.Color
    piece_type: chess.PieceType
    status: PieceStatus
    square: chess.Square | None
    graveyard_slot: int | None
    graveyard_color: chess.Color | None
    pawn_storage_slot: int | None
    reserve_slot: int | None
    expected_pos: np.ndarray


@dataclass(frozen=True)
class SyncError:
    message: str
    square: chess.Square | None = None
    body_name: str | None = None


class PhysicalPieceRegistry:
    def __init__(self, env=None, board: chess.Board | None = None) -> None:
        self.env = env
        self.board = board or chess.Board()
        self.records: dict[str, PieceRecord] = {}

    def initialize(self, settled_positions: dict[str, np.ndarray]) -> None:
        self.records.clear()
        counters: dict[tuple[chess.Color, chess.PieceType], int] = {}
        for square, piece in self.board.piece_map().items():
            key = (piece.color, piece.piece_type)
            piece_id = counters.get(key, 0)
            counters[key] = piece_id + 1
            color_char = "w" if piece.color == chess.WHITE else "b"
            type_char = piece.symbol().upper()
            body = f"piece_{color_char}_{type_char}_{piece_id}"
            self.records[body] = PieceRecord(body, piece.color, piece.piece_type, PieceStatus.ACTIVE, square, None, None, None, None, np.array(settled_positions.get(body, np.zeros(3)), dtype=float))
        for color in (chess.WHITE, chess.BLACK):
            color_char = "w" if color == chess.WHITE else "b"
            for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                type_char = chess.piece_symbol(piece_type).upper()
                for slot in range(2):
                    body = f"piece_reserve_{color_char}_{type_char}_{slot}"
                    self.records[body] = PieceRecord(body, color, piece_type, PieceStatus.RESERVE, None, None, None, None, slot, np.array(settled_positions.get(body, np.zeros(3)), dtype=float))

    def get_piece_at(self, square: chess.Square) -> PieceRecord | None:
        for record in self.records.values():
            if record.square == square and record.status in (PieceStatus.ACTIVE, PieceStatus.PROMOTED):
                return record
        return None

    def get_piece_by_body(self, body_name: str) -> PieceRecord | None:
        return self.records.get(body_name)

    def move_piece(self, body_name: str, to_square: chess.Square, new_expected_pos: np.ndarray) -> None:
        rec = self._record(body_name)
        rec.square = to_square
        if rec.status != PieceStatus.PROMOTED:
            rec.status = PieceStatus.ACTIVE
        rec.expected_pos = np.array(new_expected_pos, dtype=float)

    def capture_piece(self, body_name: str, graveyard_color: chess.Color, graveyard_slot: int, slot_pos: np.ndarray) -> None:
        rec = self._record(body_name)
        rec.status = PieceStatus.CAPTURED
        rec.square = None
        rec.graveyard_color = graveyard_color
        rec.graveyard_slot = graveyard_slot
        rec.expected_pos = np.array(slot_pos, dtype=float)

    def store_promoted_pawn(self, pawn_body: str, pawn_storage_slot: int, slot_pos: np.ndarray) -> None:
        rec = self._record(pawn_body)
        rec.status = PieceStatus.PROMOTION_PAWN_STORED
        rec.square = None
        rec.pawn_storage_slot = pawn_storage_slot
        rec.expected_pos = np.array(slot_pos, dtype=float)

    def activate_reserve_piece(self, reserve_body: str, to_square: chess.Square, new_pos: np.ndarray) -> None:
        rec = self._record(reserve_body)
        rec.status = PieceStatus.PROMOTED
        rec.square = to_square
        rec.reserve_slot = None
        rec.expected_pos = np.array(new_pos, dtype=float)

    def reset(self) -> None:
        self.initialize({})

    def get_all_active(self) -> list[PieceRecord]:
        return [r for r in self.records.values() if r.status in (PieceStatus.ACTIVE, PieceStatus.PROMOTED)]

    def get_all_captured(self) -> list[PieceRecord]:
        return [r for r in self.records.values() if r.status == PieceStatus.CAPTURED]

    def get_all_reserve(self) -> list[PieceRecord]:
        return [r for r in self.records.values() if r.status == PieceStatus.RESERVE]

    def validate_sync(self, logical_board: chess.Board) -> list[SyncError]:
        errors: list[SyncError] = []
        occupied = {record.square: record for record in self.get_all_active()}
        for square, piece in logical_board.piece_map().items():
            record = occupied.get(square)
            if record is None:
                errors.append(SyncError("Missing physical piece for logical square", square=square))
            elif record.color != piece.color or record.piece_type != piece.piece_type:
                errors.append(SyncError("Physical piece type/color mismatch", square=square, body_name=record.body_name))
        for square, record in occupied.items():
            piece = logical_board.piece_at(square)
            if piece is None:
                errors.append(SyncError("Physical piece exists on empty logical square", square=square, body_name=record.body_name))
        return errors

    def _record(self, body_name: str) -> PieceRecord:
        if body_name not in self.records:
            raise KeyError(f"Unknown piece body: {body_name}")
        return self.records[body_name]
