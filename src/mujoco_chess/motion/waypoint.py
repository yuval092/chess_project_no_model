from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WaypointStage(Enum):
    MOVE_TO_HOME = "MOVE_TO_HOME"
    MOVE_TO_SOURCE_HOVER = "MOVE_TO_SOURCE_HOVER"
    DESCEND_TO_GRASP = "DESCEND_TO_GRASP"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    ASCEND_WITH_PIECE = "ASCEND_WITH_PIECE"
    MOVE_TO_DESTINATION_HOVER = "MOVE_TO_DESTINATION_HOVER"
    DESCEND_TO_RELEASE = "DESCEND_TO_RELEASE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    ASCEND_AFTER_RELEASE = "ASCEND_AFTER_RELEASE"
    RETURN_HOME = "RETURN_HOME"


@dataclass(frozen=True)
class WaypointPlan:
    piece_body: str
    source_pos: np.ndarray
    destination_pos: np.ndarray
    hover_z: float
    grasp_z: float
    release_z: float
    crane_orientation: np.ndarray
