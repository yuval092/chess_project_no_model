from __future__ import annotations

import copy
import logging

import chess

from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.move_translation.translator import MoveExecutionTransaction, MoveTranslator, PartialMoveFailureError
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry

LOGGER = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, chess_engine: ChessEngine, registry: PhysicalPieceRegistry, translator: MoveTranslator, executor=None) -> None:
        self.chess_engine = chess_engine
        self.registry = registry
        self.translator = translator
        self.executor = executor

    def execute_chess_move(self, move: chess.Move) -> bool:
        if not self.chess_engine.is_move_legal(move):
            LOGGER.warning("Rejected illegal move %s", move.uci())
            return False
        board = self.chess_engine.get_board_state()
        self.chess_engine.apply_move(move)
        actions = self.translator.translate(move, board)
        transaction = MoveExecutionTransaction(
            chess_move=move,
            actions=actions,
            completed_action_indices=[],
            registry_snapshot=copy.deepcopy(self.registry.records),
        )
        if self.executor is not None:
            for index, action in enumerate(actions):
                result = self.executor.execute_action(action, transaction=transaction)
                if not result:
                    raise PartialMoveFailureError(
                        f"Physical action {index} failed for move {move.uci()}; logical board and registry were not committed",
                        transaction,
                        index,
                    )
                transaction.completed_action_indices.append(index)
                transaction.pending_expected_positions[action.piece_body] = action.destination_pos.copy()
        self._commit_registry_updates(actions)
        self.chess_engine.commit_move(move)
        transaction.committed = True
        return True

    def _commit_registry_updates(self, actions) -> None:
        for action in actions:
            if action.graveyard_slot is not None:
                self.registry.capture_piece(action.piece_body, action.graveyard_color, action.graveyard_slot, action.destination_pos)
            elif action.pawn_storage_slot is not None:
                self.registry.store_promoted_pawn(action.piece_body, action.pawn_storage_slot, action.destination_pos)
            elif action.reserve_piece and action.destination_square is not None:
                self.registry.activate_reserve_piece(action.piece_body, action.destination_square, action.destination_pos)
            elif action.destination_square is not None:
                self.registry.move_piece(action.piece_body, action.destination_square, action.destination_pos)
