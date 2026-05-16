from __future__ import annotations

import copy
import logging
import queue

import chess

from mujoco_chess.chess_logic.engine import ChessEngine
from mujoco_chess.move_translation.translator import MoveExecutionTransaction, MoveTranslator, PartialMoveFailureError
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry
from mujoco_chess.ui.event_handler import AIPlayRequest, MoveRequest, QuitRequest, ResetRequest

LOGGER = logging.getLogger(__name__)


class GameLoop:
    def __init__(
        self,
        chess_engine: ChessEngine,
        registry: PhysicalPieceRegistry,
        translator: MoveTranslator,
        executor=None,
        move_selector=None,
        human_color: chess.Color | None = chess.WHITE,
        event_queue: queue.Queue | None = None,
        state_queue: queue.Queue | None = None,
        slot_manager=None,
        mapper=None,
        health_runner=None,
    ) -> None:
        self.chess_engine = chess_engine
        self.registry = registry
        self.translator = translator
        self.executor = executor
        self.move_selector = move_selector
        self.human_color = human_color
        self.event_queue = event_queue
        self.state_queue = state_queue
        self.slot_manager = slot_manager
        self.mapper = mapper
        self.health_runner = health_runner

    # ------------------------------------------------------------------
    # Main game loop
    # ------------------------------------------------------------------

    def run(self, max_moves: int | None = None) -> str | None:
        """Main game loop. Returns game result string or None."""
        move_count = 0
        self._send_state_to_ui()
        while True:
            if self.chess_engine.is_game_over():
                result = self.chess_engine.get_game_result()
                self._handle_game_over(result)
                return result

            if max_moves is not None and move_count >= max_moves:
                LOGGER.info("Max moves (%d) reached", max_moves)
                return None

            if self._is_human_turn():
                if self.event_queue is None:
                    LOGGER.info("Human turn but no event_queue — stopping loop")
                    break
                # Poll for UI event
                try:
                    event = self.event_queue.get_nowait()
                except queue.Empty:
                    continue
                handled = self._dispatch_event(event)
                if handled == "quit":
                    break
                if handled == "move":
                    move_count += 1
                    self._send_state_to_ui()
            else:
                # AI turn (or human_color is None → AI plays both sides)
                if self.move_selector is None:
                    LOGGER.warning("AI turn but no move_selector configured — stopping loop")
                    break
                board = self.chess_engine.get_board_state()
                move = self.move_selector.choose_move(board)
                success = self._execute_with_error_handling(move)
                if success:
                    move_count += 1
                    self._send_state_to_ui()

        return self.chess_engine.get_game_result()

    def _dispatch_event(self, event) -> str | None:
        """Handle a single UI event. Returns 'quit', 'move', or None."""
        if isinstance(event, QuitRequest):
            return "quit"
        if isinstance(event, ResetRequest):
            self.reset()
            self._send_state_to_ui()
            return None
        if isinstance(event, AIPlayRequest):
            if self.move_selector is not None:
                board = self.chess_engine.get_board_state()
                move = self.move_selector.choose_move(board)
                success = self._execute_with_error_handling(move)
                if success:
                    return "move"
            return None
        if isinstance(event, MoveRequest):
            move = chess.Move(event.from_sq, event.to_sq, promotion=event.promotion)
            success = self._execute_with_error_handling(move)
            if success:
                return "move"
        return None

    def _execute_with_error_handling(self, move: chess.Move) -> bool:
        """Execute a move, handling PartialMoveFailureError gracefully."""
        try:
            return self.execute_chess_move(move)
        except PartialMoveFailureError as exc:
            LOGGER.error("Partial move failure for %s: %s", move.uci(), exc)
            return False

    def _is_human_turn(self) -> bool:
        if self.human_color is None:
            return False
        return self.chess_engine.get_side_to_move() == self.human_color

    def _process_ui_events(self) -> None:
        """Drain all pending UI events from the queue."""
        if self.event_queue is None:
            return
        while True:
            try:
                event = self.event_queue.get_nowait()
                self._dispatch_event(event)
            except queue.Empty:
                break

    def _send_state_to_ui(self) -> None:
        """Put a board snapshot on the state_queue for UI consumption."""
        if self.state_queue is None:
            return
        try:
            self.state_queue.put_nowait(self.chess_engine.get_board_state())
        except queue.Full:
            pass

    def _handle_game_over(self, result: str | None) -> None:
        LOGGER.info("Game over: %s", result)
        self._send_state_to_ui()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset chess engine, slot manager, and registry to starting state."""
        self.chess_engine.reset()
        if self.slot_manager is not None:
            self.slot_manager.reset()
        if self.mapper is not None:
            board = self.chess_engine.get_board_state()
            positions = {}
            for square, piece in board.piece_map().items():
                # Build body name using the same naming convention as registry
                key = (piece.color, piece.piece_type)
                color_char = "w" if piece.color == chess.WHITE else "b"
                type_char = piece.symbol().upper()
                positions[(piece.color, piece.piece_type)] = positions.get((piece.color, piece.piece_type), -1) + 1
                piece_id = positions[(piece.color, piece.piece_type)]
                body = f"piece_{color_char}_{type_char}_{piece_id}"
                positions[body] = self.mapper.square_to_world(square)
            # Re-initialize registry with starting positions from mapper
            settled: dict[str, object] = {}
            counters: dict[tuple, int] = {}
            for square, piece in board.piece_map().items():
                key = (piece.color, piece.piece_type)
                piece_id = counters.get(key, 0)
                counters[key] = piece_id + 1
                color_char = "w" if piece.color == chess.WHITE else "b"
                type_char = piece.symbol().upper()
                body = f"piece_{color_char}_{type_char}_{piece_id}"
                settled[body] = self.mapper.square_to_world(square)
            self.registry.initialize(settled)
        else:
            self.registry.initialize({})

    # ------------------------------------------------------------------
    # Core move execution (existing)
    # ------------------------------------------------------------------

    def execute_chess_move(self, move: chess.Move) -> bool:
        if not self.chess_engine.is_move_legal(move):
            LOGGER.warning("Rejected illegal move %s", move.uci())
            return False

        if self.health_runner is not None:
            from mujoco_chess.health.runner import CheckContext
            self.health_runner.run_checks(CheckContext.START_OF_TURN)

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

        if self.health_runner is not None:
            from mujoco_chess.health.runner import CheckContext
            self.health_runner.run_checks(CheckContext.END_OF_TURN)

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
