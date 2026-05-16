from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigValidationError(Exception):
    """Raised when application configuration is missing or inconsistent."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _rgba(value: list[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("RGBA values must contain exactly 4 floats")
    return value


def _vec3(value: list[float]) -> list[float]:
    if len(value) != 3:
        raise ValueError("3D vectors must contain exactly 3 floats")
    return value


class BoardConfig(StrictModel):
    square_size: float = Field(default=0.07, gt=0)
    files: int = 8
    ranks: int = 8
    board_thickness: float = Field(default=0.02, gt=0)
    board_origin_x: float = 0.0
    board_origin_y: float = 0.0
    board_origin_z: float = 0.0
    light_square_rgba: list[float]
    dark_square_rgba: list[float]
    board_rgba: list[float]
    table_height: float
    graveyard_slot_spacing: float = Field(default=0.075, gt=0)
    graveyard_x_gap: float = Field(default=0.05, gt=0)
    pawn_storage_slot_spacing: float = Field(default=0.05, gt=0)
    pawn_storage_front_offset: float = Field(default=0.20, gt=0)
    reserve_slot_spacing: float = Field(default=0.05, gt=0)
    reserve_area_offset_y: float = Field(default=0.27, gt=0)

    _light_rgba = field_validator("light_square_rgba")(_rgba)
    _dark_rgba = field_validator("dark_square_rgba")(_rgba)
    _board_rgba = field_validator("board_rgba")(_rgba)

    @property
    def board_surface_z(self) -> float:
        return self.board_origin_z + self.board_thickness

    @property
    def board_max_x(self) -> float:
        return self.board_origin_x + self.files * self.square_size

    @property
    def board_max_y(self) -> float:
        return self.board_origin_y + self.ranks * self.square_size


class PiecesConfig(StrictModel):
    base_radius: float = Field(gt=0)
    base_height: float = Field(gt=0)
    shaft_radius: float = Field(gt=0)
    shaft_height: float = Field(gt=0)
    mass: float = Field(gt=0)
    friction: list[float]
    solimp: list[float]
    solref: list[float]
    white_rgba: list[float]
    black_rgba: list[float]
    stl_scale: float = Field(default=0.0, ge=0)
    stl_placeholders: bool = True

    @field_validator("friction")
    @classmethod
    def _friction(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("friction must contain exactly 3 floats")
        return value

    _white_rgba = field_validator("white_rgba")(_rgba)
    _black_rgba = field_validator("black_rgba")(_rgba)

    @property
    def total_height(self) -> float:
        return self.base_height + self.shaft_height


class ArmConfig(StrictModel):
    base_x: float
    base_y: float
    base_z: float
    base_yaw: float = 0.0
    base_body_name: str = "robot0:base_link"
    home_position: list[float]
    gripper_open_width: float = Field(default=0.05, gt=0)
    gripper_grasp_width: float = Field(default=0.022, gt=0)
    gripper_hold_width: float = Field(gt=0)
    ee_site_name: str
    ee_weld_body_name: str
    gripper_left_actuator: str
    gripper_right_actuator: str
    mocap_body_name: str = "ee_target"
    crane_down_quat: list[float]
    initial_qpos: dict[str, float] = Field(default_factory=dict)

    _home_position = field_validator("home_position")(_vec3)

    @field_validator("crane_down_quat")
    @classmethod
    def _quat(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("quaternion must contain exactly 4 floats")
        return value


class WaypointConfig(StrictModel):
    safe_hover_clearance: float = Field(gt=0)
    descent_speed: float = Field(gt=0)
    transit_speed: float = Field(gt=0)
    ascent_speed: float = Field(gt=0)
    position_tolerance: float = Field(gt=0)
    velocity_threshold_linear: float = Field(gt=0)
    velocity_threshold_angular: float = Field(gt=0)
    held_piece_velocity_threshold: float = Field(gt=0)
    settle_steps: int = Field(gt=0)
    stage_timeout_steps: int = Field(gt=0)
    grasp_z_offset: float
    release_z_offset: float
    piece_settle_steps: int = Field(gt=0)
    arm_settle_steps: int = Field(gt=0)
    initial_settle_steps: int = Field(gt=0)
    gripper_settle_steps: int = Field(gt=0)
    robot_joint_velocity_threshold: float = Field(gt=0)
    gripper_joint_velocity_threshold: float = Field(gt=0)
    released_piece_velocity_threshold: float = Field(gt=0)


class HealthConfig(StrictModel):
    piece_tilt_threshold_deg: float = Field(gt=0)
    piece_drift_threshold_m: float = Field(gt=0)
    piece_sink_threshold_m: float = Field(gt=0)
    piece_float_threshold_m: float = Field(gt=0)
    unexpected_velocity_threshold: float = Field(gt=0)
    grasp_contact_required: bool
    slip_velocity_threshold: float = Field(gt=0)
    collision_contact_threshold: float = Field(gt=0)


class LoggingConfig(StrictModel):
    log_dir: str
    app_log_level: str
    simulation_log_level: str
    motion_log_level: str
    health_log_level: str
    debug_log_level: str
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=0)


class GameConfig(StrictModel):
    human_color: Literal["white", "black"] = "white"
    selector_type: Literal["random", "heuristic"] = "heuristic"


class DebugConfig(StrictModel):
    enabled: bool
    show_square_centers: bool
    show_waypoint_path: bool
    highlight_target_piece: bool
    pause_before_stage: bool
    step_mode: bool


@dataclass(frozen=True)
class AppConfig:
    board: BoardConfig
    pieces: PiecesConfig
    arm: ArmConfig
    waypoint: WaypointConfig
    health: HealthConfig
    logging: LoggingConfig
    game: GameConfig
    debug: DebugConfig
    root_dir: Path


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "board": BoardConfig,
    "pieces": PiecesConfig,
    "arm": ArmConfig,
    "waypoint": WaypointConfig,
    "health": HealthConfig,
    "logging": LoggingConfig,
    "game": GameConfig,
    "debug": DebugConfig,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigValidationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Config file must contain a mapping: {path}")
    return data


def _instantiate(model: type[BaseModel], path: Path) -> BaseModel:
    try:
        return model.model_validate(_load_yaml(path))
    except Exception as exc:
        raise ConfigValidationError(f"Invalid config {path}: {exc}") from exc


def _validate_cross_config(config: AppConfig) -> None:
    board = config.board
    pieces = config.pieces
    arm = config.arm
    waypoint = config.waypoint

    checks = [
        (board.files == 8 and board.ranks == 8, "board.files and board.ranks must both be 8"),
        (board.square_size > 0, "board.square_size must be > 0"),
        (
            waypoint.safe_hover_clearance > pieces.total_height,
            "waypoint.safe_hover_clearance must be greater than total piece height",
        ),
        (
            arm.gripper_open_width < board.square_size,
            "arm.gripper_open_width must be smaller than board.square_size",
        ),
        (
            arm.gripper_grasp_width > pieces.shaft_radius * 2,
            "arm.gripper_grasp_width must be larger than piece shaft diameter",
        ),
        (2 * 8 >= 16, "graveyard layout must provide at least 16 slots per color"),
        (waypoint.initial_settle_steps > 0, "waypoint.initial_settle_steps must be > 0"),
        (waypoint.gripper_settle_steps > 0, "waypoint.gripper_settle_steps must be > 0"),
    ]
    for passed, message in checks:
        if not passed:
            raise ConfigValidationError(message)

    stl_dir = config.root_dir / "assets" / "stl"
    for name in ("pawn", "rook", "knight", "bishop", "queen", "king"):
        if not (stl_dir / f"{name}.stl").exists() and not pieces.stl_placeholders:
            raise ConfigValidationError(f"Missing STL asset and placeholders disabled: {name}.stl")


def load_all_configs(root_dir: str | Path | None = None) -> AppConfig:
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    config_dir = root / "configs"
    loaded: dict[str, BaseModel] = {}
    for name, model in CONFIG_MODELS.items():
        loaded[name] = _instantiate(model, config_dir / f"{name}.yaml")
    app_config = AppConfig(
        board=loaded["board"],  # type: ignore[arg-type]
        pieces=loaded["pieces"],  # type: ignore[arg-type]
        arm=loaded["arm"],  # type: ignore[arg-type]
        waypoint=loaded["waypoint"],  # type: ignore[arg-type]
        health=loaded["health"],  # type: ignore[arg-type]
        logging=loaded["logging"],  # type: ignore[arg-type]
        game=loaded["game"],  # type: ignore[arg-type]
        debug=loaded["debug"],  # type: ignore[arg-type]
        root_dir=root,
    )
    _validate_cross_config(app_config)
    return app_config
