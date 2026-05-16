from __future__ import annotations

from dataclasses import dataclass
import logging

import chess

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChessMoveAnalysis:
    move: chess.Move
    is_capture: bool
    captured_piece_type: chess.PieceType | None
    is_castling: bool
    is_en_passant: bool
    is_promotion: bool
    promotion_piece_type: chess.PieceType | None
    from_square: chess.Square
    to_square: chess.Square
    castling_rook_from: chess.Square | None
    castling_rook_to: chess.Square | None


MoveResult = ChessMoveAnalysis


class ChessEngine:
    def __init__(self) -> None:
        self._board = chess.Board()
        self._move_history: list[chess.Move] = []

    def get_board_state(self) -> chess.Board:
        return self._board.copy(stack=True)

    def get_legal_moves(self) -> list[chess.Move]:
        return list(self._board.legal_moves)

    def is_move_legal(self, move: chess.Move) -> bool:
        return move in self._board.legal_moves

    def apply_move(self, move: chess.Move) -> MoveResult:
        LOGGER.info("Analyzing chess move %s", move.uci())
        if not self.is_move_legal(move):
            raise ValueError(f"Illegal move: {move.uci()}")
        captured_piece = self._captured_piece_for(move)
        rook_from, rook_to = self._castling_rook_squares(move)
        return ChessMoveAnalysis(
            move=move,
            is_capture=self._board.is_capture(move),
            captured_piece_type=captured_piece.piece_type if captured_piece else None,
            is_castling=self._board.is_castling(move),
            is_en_passant=self._board.is_en_passant(move),
            is_promotion=move.promotion is not None,
            promotion_piece_type=move.promotion,
            from_square=move.from_square,
            to_square=move.to_square,
            castling_rook_from=rook_from,
            castling_rook_to=rook_to,
        )

    def commit_move(self, move: chess.Move) -> None:
        LOGGER.info("Committing chess move %s", move.uci())
        if not self.is_move_legal(move):
            raise ValueError(f"Illegal move: {move.uci()}")
        self._board.push(move)
        self._move_history.append(move)

    def is_game_over(self) -> bool:
        return self._board.is_game_over()

    def get_game_result(self) -> str | None:
        return self._board.result() if self._board.is_game_over() else None

    def get_side_to_move(self) -> chess.Color:
        return self._board.turn

    def get_move_history(self) -> list[chess.Move]:
        return list(self._move_history)

    def reset(self) -> None:
        self._board.reset()
        self._move_history.clear()

    def _captured_piece_for(self, move: chess.Move) -> chess.Piece | None:
        if self._board.is_en_passant(move):
            captured_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            return self._board.piece_at(captured_square)
        return self._board.piece_at(move.to_square)

    def _castling_rook_squares(self, move: chess.Move) -> tuple[chess.Square | None, chess.Square | None]:
        if not self._board.is_castling(move):
            return None, None
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            return chess.square(7, rank), chess.square(5, rank)
        return chess.square(0, rank), chess.square(3, rank)
