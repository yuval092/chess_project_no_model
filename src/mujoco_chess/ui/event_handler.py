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
        request = MoveRequest(self.selected, square)
        self.selected = None
        return request
