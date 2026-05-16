from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class MoveRequest:
    from_sq: chess.Square
    to_sq: chess.Square
    promotion: chess.PieceType | None = None


@dataclass(frozen=True)
class AIPlayRequest:
    pass


@dataclass(frozen=True)
class ResetRequest:
    pass


@dataclass(frozen=True)
class QuitRequest:
    pass


class EventHandler:
    def __init__(self, square_px: int = 64) -> None:
        self.square_px = square_px
        self.selected: chess.Square | None = None
        self.promotion_pending: bool = False
        self.promotion_square: chess.Square | None = None
        self._promotion_from_sq: chess.Square | None = None
        self._promotion_to_sq: chess.Square | None = None

    def pixel_to_square(self, x: int, y: int) -> chess.Square | None:
        file_idx = x // self.square_px
        rank_from_top = y // self.square_px
        if not (0 <= file_idx < 8 and 0 <= rank_from_top < 8):
            return None
        return chess.square(file_idx, 7 - rank_from_top)

    def click_square(self, square: chess.Square) -> MoveRequest | None:
        if self.selected is None:
            self.selected = square
            return None
        if self.selected == square:
            # Re-select the same square: deselect
            self.selected = None
            return None
        request = MoveRequest(self.selected, square)
        self.selected = None
        return request

    def set_promotion(self, piece_type: chess.PieceType) -> MoveRequest:
        """Create a MoveRequest with a promotion piece type."""
        if self._promotion_from_sq is None or self._promotion_to_sq is None:
            raise ValueError("No pending promotion move")
        req = MoveRequest(self._promotion_from_sq, self._promotion_to_sq, promotion=piece_type)
        self.promotion_pending = False
        self.promotion_square = None
        self._promotion_from_sq = None
        self._promotion_to_sq = None
        return req

    def begin_promotion(self, from_sq: chess.Square, to_sq: chess.Square) -> None:
        """Mark that a promotion is pending (pawn reaching back rank)."""
        self.promotion_pending = True
        self.promotion_square = to_sq
        self._promotion_from_sq = from_sq
        self._promotion_to_sq = to_sq
        self.selected = None
