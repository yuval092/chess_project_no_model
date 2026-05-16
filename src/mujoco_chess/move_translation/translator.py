from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess
import numpy as np

from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.registry.piece_registry import PhysicalPieceRegistry, PieceStatus
from mujoco_chess.utils.config_loader import PiecesConfig, WaypointConfig


class TranslationError(Exception):
    pass


class PromotionReserveExhaustedError(TranslationError):
    pass


class PartialMoveFailureError(RuntimeError):
    def __init__(self, message: str, transaction: "MoveExecutionTransaction", failed_action_index: int) -> None:
        super().__init__(message)
        self.transaction = transaction
        self.failed_action_index = failed_action_index


class ActionType(Enum):
    MOVE_PIECE = "MOVE_PIECE"
    TELEPORT = "TELEPORT"


@dataclass(frozen=True)
class PhysicalAction:
    action_type: ActionType
    piece_body: str
    source_pos: np.ndarray
    destination_pos: np.ndarray
    hover_z: float
    grasp_z: float
    release_z: float
    destination_square: chess.Square | None = None
    graveyard_color: chess.Color | None = None
    graveyard_slot: int | None = None
    pawn_storage_slot: int | None = None
    reserve_piece: bool = False


@dataclass
class MoveExecutionTransaction:
    chess_move: chess.Move
    actions: list[PhysicalAction]
    completed_action_indices: list[int]
    registry_snapshot: dict
    pending_expected_positions: dict[str, np.ndarray] = field(default_factory=dict)
    pending_statuses: dict[str, PieceStatus] = field(default_factory=dict)
    committed: bool = False


class MoveTranslator:
    def __init__(
        self,
        registry: PhysicalPieceRegistry,
        mapper: CoordinateMapper,
        slot_manager: SlotManager,
        waypoint_config: WaypointConfig,
        pieces_config: PiecesConfig | None = None,
    ) -> None:
        self.registry = registry
        self.mapper = mapper
        self.slot_manager = slot_manager
        self.waypoint = waypoint_config
        self.pieces = pieces_config

    def translate(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        if board.is_castling(move):
            return self._translate_castling(move, board)
        if board.is_en_passant(move):
            return self._translate_en_passant(move, board)
        if move.promotion is not None:
            return self._translate_promotion(move, board)
        if board.is_capture(move):
            return self._translate_capture(move, board)
        return self._translate_normal(move, board)

    def _translate_normal(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        del board
        record = self._piece_at(move.from_square)
        return [self._move_action(record.body_name, move.from_square, move.to_square, destination_square=move.to_square)]

    def _translate_capture(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        captured = self._piece_at(move.to_square)
        slot = self.slot_manager.allocate_graveyard_slot(captured.color)
        graveyard_pos = self.slot_manager.graveyard_slot_world_pos(captured.color, slot)
        attacker = self._piece_at(move.from_square)
        return [
            self._raw_action(captured.body_name, self.mapper.square_to_world(move.to_square), graveyard_pos, graveyard_color=captured.color, graveyard_slot=slot),
            self._move_action(attacker.body_name, move.from_square, move.to_square, destination_square=move.to_square),
        ]

    def _translate_castling(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
        else:
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
        king = self._piece_at(move.from_square)
        rook = self._piece_at(rook_from)
        return [
            self._move_action(king.body_name, move.from_square, move.to_square, destination_square=move.to_square),
            self._move_action(rook.body_name, rook_from, rook_to, destination_square=rook_to),
        ]

    def _translate_en_passant(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        captured_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        captured = self._piece_at(captured_sq)
        slot = self.slot_manager.allocate_graveyard_slot(captured.color)
        graveyard_pos = self.slot_manager.graveyard_slot_world_pos(captured.color, slot)
        attacker = self._piece_at(move.from_square)
        return [
            self._raw_action(captured.body_name, self.mapper.square_to_world(captured_sq), graveyard_pos, graveyard_color=captured.color, graveyard_slot=slot),
            self._move_action(attacker.body_name, move.from_square, move.to_square, destination_square=move.to_square),
        ]

    def _translate_promotion(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]:
        pawn = self._piece_at(move.from_square)
        assert move.promotion is not None
        reserve_slot = self.slot_manager.get_reserve_slot(pawn.color, move.promotion)
        if reserve_slot is None:
            raise PromotionReserveExhaustedError("No reserve piece available for requested promotion")
        self.slot_manager.mark_reserve_used(pawn.color, move.promotion, reserve_slot)
        pawn_slot = self.slot_manager.allocate_pawn_storage_slot(pawn.color)
        reserve_body = f"piece_reserve_{'w' if pawn.color == chess.WHITE else 'b'}_{chess.piece_symbol(move.promotion).upper()}_{reserve_slot}"
        actions: list[PhysicalAction] = []
        if board.is_capture(move):
            captured = self._piece_at(move.to_square)
            slot = self.slot_manager.allocate_graveyard_slot(captured.color)
            actions.append(self._raw_action(captured.body_name, self.mapper.square_to_world(move.to_square), self.slot_manager.graveyard_slot_world_pos(captured.color, slot), graveyard_color=captured.color, graveyard_slot=slot))
        actions.append(self._raw_action(pawn.body_name, self.mapper.square_to_world(move.from_square), self.slot_manager.pawn_storage_slot_world_pos(pawn.color, pawn_slot), pawn_storage_slot=pawn_slot))
        actions.append(self._raw_action(reserve_body, self.slot_manager.reserve_slot_world_pos(pawn.color, move.promotion, reserve_slot), self.mapper.square_to_world(move.to_square), destination_square=move.to_square, reserve_piece=True))
        return actions

    def _move_action(self, body_name: str, from_square: chess.Square, to_square: chess.Square, **kwargs) -> PhysicalAction:
        return self._raw_action(body_name, self.mapper.square_to_world(from_square), self.mapper.square_to_world(to_square), **kwargs)

    def _raw_action(self, body_name: str, source_pos: np.ndarray, destination_pos: np.ndarray, **kwargs) -> PhysicalAction:
        is_teleport = "graveyard_slot" in kwargs or "pawn_storage_slot" in kwargs or kwargs.get("reserve_piece", False)
        action_type = ActionType.TELEPORT if is_teleport else ActionType.MOVE_PIECE
        hover_z = max(source_pos[2], destination_pos[2]) + self.waypoint.safe_hover_clearance
        piece_midshaft = 0.0 if self.pieces is None else self.pieces.base_height + self.pieces.shaft_height / 2
        return PhysicalAction(
            action_type,
            body_name,
            source_pos,
            destination_pos,
            hover_z,
            source_pos[2] + piece_midshaft + self.waypoint.grasp_z_offset,
            destination_pos[2] + piece_midshaft + self.waypoint.release_z_offset,
            **kwargs,
        )

    def _piece_at(self, square: chess.Square):
        record = self.registry.get_piece_at(square)
        if record is None:
            raise TranslationError(f"No physical piece at {chess.square_name(square)}")
        return record
