from __future__ import annotations

from abc import ABC, abstractmethod
import random

import chess

from mujoco_chess.utils.config_loader import GameConfig


class MoveSelector(ABC):
    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class RandomMoveSelector(MoveSelector):
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)


class HeuristicMoveSelector(MoveSelector):
    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        captures = [move for move in legal_moves if board.is_capture(move)]
        return random.choice(captures or legal_moves)


def create_move_selector(game_config: GameConfig) -> MoveSelector:
    if game_config.selector_type == "random":
        return RandomMoveSelector()
    if game_config.selector_type == "heuristic":
        return HeuristicMoveSelector()
    raise ValueError(f"Unsupported selector type: {game_config.selector_type}")
