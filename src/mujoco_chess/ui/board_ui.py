from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import chess

from mujoco_chess.ui.event_handler import (
    AIPlayRequest,
    EventHandler,
    MoveRequest,
    QuitRequest,
    ResetRequest,
)

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

_LIGHT_SQUARE = (220, 205, 175)
_DARK_SQUARE = (82, 115, 92)
_SIDEBAR_BG = (32, 32, 32)
_BTN_BG = (60, 60, 80)
_BTN_HOVER = (90, 90, 120)
_BTN_TEXT = (230, 230, 230)
_WHITE_PIECE_CIRCLE = (230, 220, 200)
_BLACK_PIECE_CIRCLE = (50, 40, 35)
_WHITE_PIECE_LETTER = (30, 30, 30)
_BLACK_PIECE_LETTER = (220, 210, 200)

_PIECE_LETTERS = {
    chess.PAWN: "P",
    chess.ROOK: "R",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.QUEEN: "Q",
    chess.KING: "K",
}

_PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


class BoardUI:
    def __init__(self, event_queue: queue.Queue | None = None, state_queue: queue.Queue | None = None, square_px: int = 64) -> None:
        self.event_queue = event_queue
        self.state_queue = state_queue
        self.square_px = square_px
        self.board: chess.Board = chess.Board()
        self.promotion_pending = False
        self._promo_from_sq: chess.Square | None = None
        self._promo_to_sq: chess.Square | None = None
        self._event_handler = EventHandler(square_px=square_px)
        self._last_move: str = ""
        self._status: str = "White to move"
        self._running = False

    def run(self) -> None:
        import os
        try:
            import pygame
        except Exception as exc:
            raise RuntimeError("pygame is required to run the board UI") from exc

        # Handle headless environments
        if "SDL_VIDEODRIVER" not in os.environ:
            pass  # let pygame pick driver

        try:
            pygame.init()
        except Exception as exc:
            LOGGER.warning("pygame.init() failed (headless?): %s", exc)
            return

        sidebar_w = 220
        board_px = self.square_px * 8
        try:
            screen = pygame.display.set_mode((board_px + sidebar_w, board_px))
            pygame.display.set_caption("MuJoCo Chess")
        except Exception as exc:
            LOGGER.warning("pygame display init failed (headless?): %s", exc)
            pygame.quit()
            return

        font = pygame.font.SysFont("monospace", 18, bold=True)
        small_font = pygame.font.SysFont("monospace", 14)
        clock = pygame.time.Clock()
        self._running = True

        while self._running:
            # Drain state queue
            try:
                while True:
                    board_snapshot = self.state_queue.get_nowait()
                    self.board = board_snapshot
                    self._update_status()
            except (queue.Empty, AttributeError):
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    if self.event_queue is not None:
                        self.event_queue.put(QuitRequest())
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos, board_px, sidebar_w)

            screen.fill(_SIDEBAR_BG)
            self._draw_board(screen)
            self._draw_pieces(screen, font)
            self._draw_sidebar(screen, font, small_font, board_px, sidebar_w)
            if self.promotion_pending:
                self._draw_promotion_popup(screen, font, board_px, sidebar_w)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _update_status(self) -> None:
        board = self.board
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            self._status = f"Checkmate! {winner} wins"
        elif board.is_stalemate():
            self._status = "Stalemate"
        elif board.is_insufficient_material():
            self._status = "Draw (material)"
        elif board.is_check():
            side = "White" if board.turn == chess.WHITE else "Black"
            self._status = f"{side} in check"
        else:
            side = "White" if board.turn == chess.WHITE else "Black"
            self._status = f"{side} to move"

        if board.move_stack:
            self._last_move = board.peek().uci()

    def _draw_board(self, screen) -> None:
        import pygame
        sq = self.square_px
        for file_idx in range(8):
            for rank_idx in range(8):
                color = _LIGHT_SQUARE if (file_idx + rank_idx) % 2 == 0 else _DARK_SQUARE
                pygame.draw.rect(screen, color, (file_idx * sq, (7 - rank_idx) * sq, sq, sq))

    def _draw_pieces(self, screen, font) -> None:
        import pygame
        sq = self.square_px
        radius = sq // 3
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            cx = file_idx * sq + sq // 2
            cy = (7 - rank_idx) * sq + sq // 2
            if piece.color == chess.WHITE:
                pygame.draw.circle(screen, _WHITE_PIECE_CIRCLE, (cx, cy), radius)
                letter_color = _WHITE_PIECE_LETTER
            else:
                pygame.draw.circle(screen, _BLACK_PIECE_CIRCLE, (cx, cy), radius)
                letter_color = _BLACK_PIECE_LETTER
            letter = _PIECE_LETTERS.get(piece.piece_type, "?")
            text_surf = font.render(letter, True, letter_color)
            text_rect = text_surf.get_rect(center=(cx, cy))
            screen.blit(text_surf, text_rect)

    def _draw_sidebar(self, screen, font, small_font, board_px: int, sidebar_w: int) -> None:
        import pygame
        x0 = board_px + 5
        y = 10

        # Status
        text = font.render("Status:", True, (180, 180, 180))
        screen.blit(text, (x0, y))
        y += 24
        for line in self._wrap(self._status, small_font, sidebar_w - 10):
            surf = small_font.render(line, True, (230, 230, 230))
            screen.blit(surf, (x0, y))
            y += 18
        y += 8

        # Last move
        text = font.render("Last move:", True, (180, 180, 180))
        screen.blit(text, (x0, y))
        y += 24
        lm_surf = small_font.render(self._last_move or "—", True, (230, 230, 230))
        screen.blit(lm_surf, (x0, y))
        y += 30

        # AI plays for me button
        btn_rect = pygame.Rect(board_px + 10, y, sidebar_w - 20, 36)
        pygame.draw.rect(screen, _BTN_BG, btn_rect, border_radius=6)
        ai_label = font.render("AI plays for me", True, _BTN_TEXT)
        screen.blit(ai_label, ai_label.get_rect(center=btn_rect.center))
        self._ai_btn_rect = btn_rect
        y += 46

        # Reset button
        btn_rect2 = pygame.Rect(board_px + 10, y, sidebar_w - 20, 36)
        pygame.draw.rect(screen, _BTN_BG, btn_rect2, border_radius=6)
        reset_label = font.render("Reset", True, _BTN_TEXT)
        screen.blit(reset_label, reset_label.get_rect(center=btn_rect2.center))
        self._reset_btn_rect = btn_rect2

    def _draw_promotion_popup(self, screen, font, board_px: int, sidebar_w: int) -> None:
        import pygame
        sq = self.square_px
        popup_w = sq * 4
        popup_h = sq + 20
        px = (board_px - popup_w) // 2
        py = (sq * 8 - popup_h) // 2
        pygame.draw.rect(screen, (40, 40, 40), (px - 4, py - 4, popup_w + 8, popup_h + 8))
        pygame.draw.rect(screen, (220, 180, 80), (px - 4, py - 4, popup_w + 8, popup_h + 8), 2)
        self._promo_btn_rects = []
        for i, pt in enumerate(_PROMO_TYPES):
            btn_rect = pygame.Rect(px + i * sq, py + 10, sq, sq)
            pygame.draw.rect(screen, _BTN_BG, btn_rect)
            letter = _PIECE_LETTERS[pt]
            surf = font.render(letter, True, _BTN_TEXT)
            screen.blit(surf, surf.get_rect(center=btn_rect.center))
            self._promo_btn_rects.append((btn_rect, pt))

    def _handle_click(self, pos, board_px: int, sidebar_w: int) -> None:
        x, y = pos

        # Promotion popup takes priority
        if self.promotion_pending:
            for btn_rect, pt in getattr(self, "_promo_btn_rects", []):
                if btn_rect.collidepoint(x, y):
                    self._complete_promotion(pt)
            return

        # Sidebar buttons
        if x >= board_px:
            if hasattr(self, "_ai_btn_rect") and self._ai_btn_rect.collidepoint(x, y):
                if self.event_queue is not None:
                    self.event_queue.put(AIPlayRequest())
            elif hasattr(self, "_reset_btn_rect") and self._reset_btn_rect.collidepoint(x, y):
                if self.event_queue is not None:
                    self.event_queue.put(ResetRequest())
            return

        # Board click
        square = self._event_handler.pixel_to_square(x, y)
        if square is None:
            return

        # Check if this would be a promotion move
        if self._event_handler.selected is not None:
            from_sq = self._event_handler.selected
            piece = self.board.piece_at(from_sq)
            if (
                piece is not None
                and piece.piece_type == chess.PAWN
                and (
                    (piece.color == chess.WHITE and chess.square_rank(square) == 7)
                    or (piece.color == chess.BLACK and chess.square_rank(square) == 0)
                )
            ):
                # Check it's a legal promotion (any promotion piece)
                legal_promo = any(
                    m.from_square == from_sq and m.to_square == square and m.promotion is not None
                    for m in self.board.legal_moves
                )
                if legal_promo:
                    self._promo_from_sq = from_sq
                    self._promo_to_sq = square
                    self.promotion_pending = True
                    self._event_handler.selected = None
                    return

        result = self._event_handler.click_square(square)
        if result is not None and self.event_queue is not None:
            self.event_queue.put(result)

    def _complete_promotion(self, piece_type: chess.PieceType) -> None:
        if self._promo_from_sq is not None and self._promo_to_sq is not None:
            req = MoveRequest(self._promo_from_sq, self._promo_to_sq, promotion=piece_type)
            if self.event_queue is not None:
                self.event_queue.put(req)
        self.promotion_pending = False
        self._promo_from_sq = None
        self._promo_to_sq = None

    @staticmethod
    def _wrap(text: str, font, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def start_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t
