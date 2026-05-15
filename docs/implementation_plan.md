# MuJoCo Robotic Chess — Implementation Plan

This document is the step-by-step implementation plan for the MuJoCo Robotic Chess project.
It is derived from `plan.md` and is intended to be executed by an AI coding agent.

---

## Design Decisions (Resolved Before Implementation)

The following open questions from section 21 of `plan.md` are resolved here.

### UI Framework — Pygame

**Choice:** Pygame

**Rationale:** Pygame runs in its own thread-friendly event loop, is straightforward to draw a
2D chess board on, has no heavyweight framework dependencies, integrates easily with a MuJoCo
passive viewer that runs in a separate thread, and is simple enough to maintain. PyQt and Dear
PyGui are heavier; Tkinter struggles with real-time rendering integration.

**Integration pattern:** The MuJoCo viewer runs via `mujoco.viewer.launch_passive` in the main
thread. Pygame runs in a secondary thread, communicating with the game loop via a thread-safe
queue. The game loop owns all state; UI only reads immutable state snapshots and posts move
requests.

**Threading note:** Pygame behavior on a secondary thread can be platform-dependent. If runtime
issues appear, the fallback is to run Pygame in the main thread and drive the MuJoCo passive
viewer's `sync()` call from the game loop. Document which model is in use. Either way, the UI
must only receive immutable snapshots — never a shared mutable `chess.Board`.

### Physical Piece Shape — Cylinder + Capsule Collision Geometry

**Choice:** Each piece is a short, wide cylinder (base) topped by a thin capsule (shaft), all
defined with primitive MuJoCo geoms. Exact dimensions are config-driven.

- Base cylinder: radius ~0.018 m, height ~0.008 m — provides wide stable footing.
- Shaft capsule: radius ~0.010 m, height ~0.040 m — consistent grasp zone for all piece types.
- Total piece height before STL decoration: ~0.050 m (configurable).
- STL mesh geom added as `contype="0" conaffinity="0"` (visual only).

**Rationale:** Wide base reduces tipping. Consistent shaft radius across all piece types means
one gripper width works for everything. Primitives produce predictable contact normals.

### Promotion Strategy — Reserve Area + Separate Pawn Storage

**Choice:** A promotion reserve area holds spare pieces (Q, R, B, N) for each color — 4 types
× 2 colors × 2 slots each = 16 reserve slots (accounts for two same-type promotions per color).

These 16 reserve pieces are **physical MuJoCo bodies** present in the XML from the start. They
are generated alongside the 32 active starting pieces (see Milestone 8) and registered with
status `RESERVE` in the `PhysicalPieceRegistry`.

Promoted pawns are moved to a **dedicated pawn-storage area** — separate from the
captured-piece graveyard. This is required because the 16-slot per-color graveyard is
sized for the 16 capturable pieces of that color; mixing in stored pawns would overflow it.

- Pawn-storage area: up to 8 slots per color, placed near the reserve area.
- Captured-piece graveyard: 16 slots per color (for pieces captured by the opponent).
- Reserve area: 16 spare pieces for promotion.

Physical promotion sequence:
1. If the destination square contains an enemy piece (promotion-capture), move that piece to
   its color's graveyard first.
2. Move the promoting pawn to the pawn-storage area (its own color's pawn slots).
3. Move the selected reserve piece from the reserve area to the promotion destination square.
4. Only after all physical actions succeed, commit the logical chess move.

**v1 limitation (Option B — documented against the original requirement):** At most 2
promoted pieces of each type per color are supported (e.g. at most 2 promoted queens per
color). This means the simulator handles promotion as a move type but cannot support every
theoretically legal chess game. Games requiring a third same-type promotion are rejected with
`PromotionReserveExhaustedError` and a clear message. The original requirement asked for a
full chess game, so this is an explicit known limitation against that requirement. Future
versions can expand the reserve to support up to 8 promotions of each type per color.

### Graveyard and Pawn-Storage Layout

**Choice:** Three storage areas beside the board (all opposite the arm base side):

- **White captured-piece graveyard** (16 slots): left of board (negative X side), 2×8 grid.
  Contains white pieces captured by Black.
- **Black captured-piece graveyard** (16 slots): right of board (positive X side), 2×8 grid.
  Contains black pieces captured by White.
- **Pawn-storage area** (8 slots per color, total 16 slots): placed adjacent to the graveyard
  strips (e.g. further out on each X side), 1×8 grid per color.
  Contains promoted pawns awaiting end of game.

All slot spacings configurable via `board.yaml` → `graveyard_slot_spacing` (default 0.075 m).
All platforms at the same height as the board surface.

**Rationale:** Separating pawn storage from the captured-piece graveyard prevents capacity
overflow. A full game can produce at most 16 captured pieces + 8 stored pawns per color.
The 16-slot graveyard is for captures only; stored pawns go to their own area.
Slot allocation is deterministic: filled in order from slot 0 upward per color.

### Arm and Board Placement

**Choice:** The Fetch arm base is placed behind the board (positive Y side from White's
perspective), centered on the board's X midpoint, at a slightly elevated mount height.

Specific placement parameters (all configurable in `configs/arm.yaml`):
- `arm_base_x`: board_center_x (centered)
- `arm_base_y`: board_max_y + 0.15 m (behind board)
- `arm_base_z`: table_height (same table surface)
- `home_position`: directly above arm base at safe hover height

After placement, the reachability validation tool (Milestone 15) must confirm that every board
square, graveyard slot, and reserve slot is reachable. If not, these parameters will be adjusted.

**Fetch arm asset source:** Use the **Fetch robot** XML from the
[Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) `FetchPickAndPlace`
environment (`gymnasium_robotics/envs/fetch/`). Do **not** use the Franka/Panda robot —
it is a different arm with different geometry, kinematics, and actuator names. Copy the
relevant MJCF files into `assets/robot/`. The following must be identified and recorded
in `docs/environment.md` before Milestone 13 begins:
- Exact body name of the robot base (typically `robot0:base`).
- Name of the **body** (not site) that is closest to the gripper center and suitable for the
  mocap weld (typically `robot0:gripper_link` or `robot0:r_gripper_finger_joint` parent).
  Record as `ee_weld_body_name` in `ArmConfig`.
- Name of the end-effector **site** used only for measuring gripper world position
  (typically `robot0:grip_site`). Record as `ee_site_name` in `ArmConfig`.
- Actuator names for the two gripper fingers (`robot0:l_gripper_finger_joint`,
  `robot0:r_gripper_finger_joint` or equivalent).
- How the mocap weld is set up: a `mocap` body named `ee_target` is added to the XML, and
  an `equality → weld` constraint links `ee_target` to `ee_weld_body_name` (a body, never a
  site — MuJoCo weld constraints connect bodies).
- Confirm no goal/target site exists in the sourced XML; remove any if found.

### Grasp Strategy — Mid-Shaft Grasp

**Choice:** The gripper targets the midpoint of the piece shaft (capsule geom). Grasp height is
calculated as:

```
grasp_z = board_top_z + piece_base_height + (piece_shaft_height / 2)
```

Gripper open width: **0.05 m** (wider than shaft diameter of 0.020 m to clear the piece, and
below the 0.07 m square size so fingers do not clip neighboring pieces during descent).
The total lateral envelope of the open gripper — including finger bodies, not just tip
separation — must also fit within the 0.07 m square clearance envelope; verify this
experimentally with the reachability tool.

Gripper closed/grasp width: **0.022 m** (slightly wider than shaft diameter of 0.020 m to
produce reliable squeeze without crushing).

All values are in `configs/arm.yaml` and `configs/waypoint.yaml`.

---

## Project Structure to Create

```
mujoco_chess/
├── main.py                          # Entry point
├── pyproject.toml
├── README.md
├── configs/
│   ├── board.yaml
│   ├── pieces.yaml
│   ├── arm.yaml
│   ├── waypoint.yaml
│   ├── health.yaml
│   ├── logging.yaml
│   ├── game.yaml                    # NEW: human color, selector type
│   └── debug.yaml
├── assets/
│   ├── stl/                         # STL meshes (placeholders acceptable in v1)
│   ├── robot/                       # Fetch arm XML assets (copied from Menagerie)
│   └── textures/
├── generated/
│   └── chess_env.xml                # Auto-generated, never edit by hand
├── logs/
│   ├── app.log
│   ├── simulation.log
│   ├── motion.log
│   ├── health_checks.log
│   ├── debug.log
│   └── failures/
├── src/
│   └── mujoco_chess/
│       ├── __init__.py
│       ├── app/
│       │   ├── __init__.py
│       │   ├── game_loop.py         # Orchestrates all layers
│       │   └── startup.py           # Config load, XML gen, MuJoCo init sequence
│       ├── chess_logic/
│       │   ├── __init__.py
│       │   ├── engine.py            # ChessEngine wrapper around python-chess
│       │   └── move_selector.py     # MoveSelector interface + RandomSelector
│       ├── board/
│       │   ├── __init__.py
│       │   ├── coordinate_mapper.py # Square <-> MuJoCo world coords
│       │   └── slot_manager.py      # Graveyard + reserve slot allocation
│       ├── move_translation/
│       │   ├── __init__.py
│       │   └── translator.py        # Chess move -> list[PhysicalAction]
│       ├── registry/
│       │   ├── __init__.py
│       │   └── piece_registry.py    # PhysicalPieceRegistry
│       ├── mujoco_env/
│       │   ├── __init__.py
│       │   ├── xml_generator.py     # Generates chess_env.xml from configs
│       │   └── env.py               # MuJoCoEnv: load, step, query, viewer
│       ├── motion/
│       │   ├── __init__.py
│       │   ├── waypoint.py          # WaypointStage enum + WaypointPlan dataclass
│       │   ├── executor.py          # MotionExecutor: runs waypoint plans
│       │   └── gripper.py           # GripperController
│       ├── health/
│       │   ├── __init__.py
│       │   ├── checks.py            # Individual check functions
│       │   └── runner.py            # HealthCheckRunner: schedules and runs checks
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── board_ui.py          # Pygame chess board display
│       │   └── event_handler.py     # Translates Pygame events to game events
│       ├── debug/
│       │   ├── __init__.py
│       │   ├── cli.py               # CLI entry point for debug commands
│       │   ├── scenarios.py         # Named debug scenarios
│       │   ├── recovery.py          # Developer-only recovery tools
│       │   └── visual_markers.py    # Optional MuJoCo visual debug markers
│       ├── logging_setup/
│       │   ├── __init__.py
│       │   └── setup.py             # Configures all loggers from logging.yaml
│       └── utils/
│           ├── __init__.py
│           └── config_loader.py     # YAML load + Pydantic/dataclass validation
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_coordinate_mapper.py
│   │   ├── test_slot_manager.py
│   │   ├── test_chess_engine.py
│   │   ├── test_move_translator.py
│   │   ├── test_piece_registry.py
│   │   └── test_health_checks.py
│   ├── integration/
│   │   ├── test_move_pipeline.py
│   │   ├── test_capture_pipeline.py
│   │   ├── test_special_moves.py
│   │   └── test_game_reset.py
│   ├── simulation/
│   │   ├── test_xml_generation.py
│   │   ├── test_env_load.py
│   │   ├── test_piece_stability.py
│   │   ├── test_reachability.py
│   │   └── test_physical_moves.py
│   └── regression/
│       └── .gitkeep
└── docs/
    ├── implementation_plan.md       # This file
    ├── architecture.md
    ├── environment.md
    ├── waypoint_algorithm.md
    ├── chess_move_translation.md
    ├── health_checks.md
    ├── debugging.md
    ├── testing.md
    └── development_guide.md
```

---

## Milestones

Each milestone ends with a validation gate. Do not proceed to the next milestone until the gate
passes. Every milestone that touches MuJoCo must have at least one debug command that can be
run to visually or programmatically verify it. Debug commands are added **incrementally** as
each capability is built — the CLI skeleton is created at Milestone 10 and extended at each
relevant later milestone.

---

### Milestone 1 — Project Skeleton

**Goal:** Create the directory structure and Python package wiring so all imports resolve.

**Steps:**

1. Create the directory tree shown in the Project Structure section above using `mkdir -p`.
2. Create all `__init__.py` files.
3. Create `pyproject.toml`:
   - Package name: `mujoco_chess`
   - Python requirement: `>=3.10`
   - Dependencies: `mujoco`, `python-chess`, `pygame`, `pyyaml`, `pydantic>=2`, `numpy`
   - Dev dependencies: `pytest`, `pytest-cov`
   - Entry points:
     - `mujoco-chess = mujoco_chess.app.startup:main`
     - `mujoco-chess-debug = mujoco_chess.debug.cli:main`
4. Create `main.py` as a thin shim: `from mujoco_chess.app.startup import main; main()`.
5. Create stub files (empty classes / `pass` bodies) for every module listed in the structure.
   This ensures imports work before implementation.
6. Create `configs/` YAML files with placeholder values (filled in during Milestone 2).
7. Create `generated/`, `logs/`, `logs/failures/`, `assets/stl/`, `assets/robot/`,
   `assets/textures/` directories.
8. Create `README.md` with one-paragraph project description.

**Validation gate:**
- `pip install -e .` succeeds with no import errors.
- `python -c "import mujoco_chess"` succeeds.
- `pytest tests/` collects zero tests (no failures because no tests yet).

---

### Milestone 2 — Config Loading and Validation

**Goal:** All config files are loadable, validated, and accessible throughout the codebase.

**Steps:**

1. **Define config schemas** in `src/mujoco_chess/utils/config_loader.py` using Pydantic v2
   `BaseModel`. Create one model per config file:

   - `BoardConfig` (from `configs/board.yaml`):
     - `square_size: float` (must be > 0, default 0.07)
     - `files: int = 8`
     - `ranks: int = 8`
     - `board_thickness: float` (default 0.02)
     - `board_origin_x: float`
     - `board_origin_y: float`
     - `board_origin_z: float` (table surface Z)
     - `light_square_rgba: list[float]` (4 floats)
     - `dark_square_rgba: list[float]` (4 floats)
     - `board_rgba: list[float]` (4 floats, board frame)
     - `table_height: float`
     - `graveyard_slot_spacing: float` (default 0.075 m — space between captured-piece slots)
     - `pawn_storage_slot_spacing: float` (default 0.075 m — space between pawn-storage slots)
     - `reserve_slot_spacing: float` (default 0.075 m — space between reserve slots)
     - `reserve_area_offset_y: float` (distance behind board_max_y for reserve area)

   - `PiecesConfig` (from `configs/pieces.yaml`):
     - `base_radius: float`
     - `base_height: float`
     - `shaft_radius: float`
     - `shaft_height: float`
     - `mass: float`
     - `friction: list[float]` (3 floats: sliding, torsional, rolling)
     - `solimp: list[float]`
     - `solref: list[float]`
     - `white_rgba: list[float]`
     - `black_rgba: list[float]`
     - `stl_scale: float`

   - `ArmConfig` (from `configs/arm.yaml`):
     - `base_x: float`
     - `base_y: float`
     - `base_z: float`
     - `home_position: list[float]` (3 floats: x, y, z)
     - `gripper_open_width: float` (default 0.05 m — must be < board.square_size)
     - `gripper_grasp_width: float` (default 0.022 m — must be > pieces.shaft_radius * 2)
     - `gripper_hold_width: float`
     - `ee_site_name: str` (name of the end-effector **site** used for position measurement)
     - `ee_weld_body_name: str` (name of the **body** welded to the mocap target — NOT a site)
     - `gripper_left_actuator: str` (actuator name for left finger)
     - `gripper_right_actuator: str` (actuator name for right finger)
     - `mocap_body_name: str = "ee_target"`
     - `crane_down_quat: list[float]` (4 floats: quaternion for downward-facing gripper orientation)

   - `WaypointConfig` (from `configs/waypoint.yaml`):
     - `safe_hover_clearance: float` (meters above tallest piece)
     - `descent_speed: float` (m per sim step)
     - `transit_speed: float`
     - `ascent_speed: float`
     - `position_tolerance: float`
     - `velocity_threshold_linear: float`
     - `velocity_threshold_angular: float`
     - `held_piece_velocity_threshold: float`
     - `settle_steps: int` (steps per settle window)
     - `stage_timeout_steps: int`
     - `grasp_z_offset: float`
     - `release_z_offset: float`
     - `initial_settle_steps: int` (steps for the startup settle phase before gameplay)
     - `gripper_settle_steps: int` (steps for gradual gripper open/close interpolation)
     - `robot_joint_velocity_threshold: float` (max allowed joint velocity for complete-halt)
     - `gripper_joint_velocity_threshold: float` (max allowed gripper finger velocity)
     - `released_piece_velocity_threshold: float` (max velocity for released piece after placement)

   - `HealthConfig` (from `configs/health.yaml`):
     - `piece_tilt_threshold_deg: float`
     - `piece_drift_threshold_m: float`
     - `piece_sink_threshold_m: float`
     - `piece_float_threshold_m: float`
     - `unexpected_velocity_threshold: float`
     - `grasp_contact_required: bool`
     - `slip_velocity_threshold: float`
     - `collision_contact_threshold: float`

   - `LoggingConfig` (from `configs/logging.yaml`):
     - `log_dir: str`
     - `app_log_level: str`
     - `simulation_log_level: str`
     - `motion_log_level: str`
     - `health_log_level: str`
     - `debug_log_level: str`
     - `max_bytes: int`
     - `backup_count: int`

   - `GameConfig` (from `configs/game.yaml`):
     - `human_color: str` (`"white"` or `"black"`, default `"white"`)
     - `selector_type: str` (`"random"` or `"heuristic"`, default `"heuristic"`)

   - `DebugConfig` (from `configs/debug.yaml`):
     - `enabled: bool`
     - `show_square_centers: bool`
     - `show_waypoint_path: bool`
     - `highlight_target_piece: bool`
     - `pause_before_stage: bool`
     - `step_mode: bool`

2. **Implement `load_all_configs()`** in `config_loader.py`:
   - Load each YAML file.
   - Instantiate the corresponding Pydantic model.
   - Run cross-config validation assertions:
     - `board.files == 8 and board.ranks == 8`
     - `board.square_size > 0`
     - `waypoint.safe_hover_clearance > pieces.base_height + pieces.shaft_height`
     - `arm.gripper_open_width < board.square_size` (0.05 < 0.07 ✓)
     - `arm.gripper_grasp_width > pieces.shaft_radius * 2`
     - Graveyard has >= 16 slots per color (derived: `2 * 8 = 16` slots per strip).
     - All STL paths in `assets/stl/` either exist or are marked as placeholder.
     - `waypoint.initial_settle_steps > 0`
     - `waypoint.gripper_settle_steps > 0`
   - Raise `ConfigValidationError` with a clear message on failure.
   - Return an `AppConfig` dataclass grouping all sub-configs.

3. **Fill in YAML files** with correct default values matching the design decisions above.

4. **Write tests** in `tests/unit/test_config.py`:
   - Test that valid YAML loads without error.
   - Test that each invalid config value raises `ConfigValidationError`.
   - Test each cross-config validation rule (including the gripper width check).
   - Test that `gripper_open_width = 0.08` with `square_size = 0.07` raises an error.
   - Test that missing `initial_settle_steps` or `gripper_settle_steps` raises an error.

**Validation gate:**
- All unit tests in `test_config.py` pass.
- `python -c "from mujoco_chess.utils.config_loader import load_all_configs; load_all_configs()"` succeeds.

---

### Milestone 3 — Chess Logic Wrapper

**Goal:** A clean `ChessEngine` class that wraps `python-chess` and exposes only the API the
rest of the system needs.

**Steps:**

1. **Implement `ChessEngine`** in `src/mujoco_chess/chess_logic/engine.py`:

   ```python
   class ChessEngine:
       def __init__(self) -> None: ...
       def get_board_state(self) -> chess.Board: ...
       def get_legal_moves(self) -> list[chess.Move]: ...
       def is_move_legal(self, move: chess.Move) -> bool: ...
       def apply_move(self, move: chess.Move) -> MoveResult: ...
       def commit_move(self, move: chess.Move) -> None: ...
       def is_game_over(self) -> bool: ...
       def get_game_result(self) -> str | None: ...
       def get_side_to_move(self) -> chess.Color: ...
       def get_move_history(self) -> list[chess.Move]: ...
       def reset(self) -> None: ...
   ```

   - `ChessMoveAnalysis` is a dataclass: `{ move, is_capture, captured_piece_type,
     is_castling, is_en_passant, is_promotion, promotion_piece_type, from_square,
     to_square, castling_rook_from, castling_rook_to }`.
   - `apply_move` must NOT push to the internal board. It returns a `ChessMoveAnalysis` for
     inspection. The caller decides when to commit (after physical success).
   - `commit_move(move)` actually pushes the move to the internal `chess.Board`.
   - Do not expose `chess.Board` internals beyond `get_board_state()`.
   - Log every `apply_move` and `commit_move` call.

2. **Implement `MoveSelector`** interface and `RandomMoveSelector` in
   `src/mujoco_chess/chess_logic/move_selector.py`:

   ```python
   class MoveSelector(ABC):
       @abstractmethod
       def choose_move(self, board: chess.Board) -> chess.Move: ...

   class RandomMoveSelector(MoveSelector):
       def choose_move(self, board: chess.Board) -> chess.Move:
           return random.choice(list(board.legal_moves))
   ```

3. **Write tests** in `tests/unit/test_chess_engine.py`:
   - Initial board state is the standard starting position.
   - Legal move list for starting position has 20 moves.
   - `is_move_legal(e2e4)` returns True on starting position.
   - `is_move_legal(e2e5)` returns False.
   - `apply_move(e2e4)` returns correct `ChessMoveAnalysis` without mutating the board.
   - `commit_move(e2e4)` mutates the board.
   - Capture detection works.
   - Castling detection works.
   - En passant detection works.
   - Promotion detection works.
   - `is_game_over()` returns False for starting position.
   - `reset()` returns to starting position.

**Validation gate:**
- All tests in `test_chess_engine.py` pass.

---

### Milestone 4 — Move Selector Interface

**Goal:** The move selector strategy is wired in and can be swapped at startup.

**Steps:**

1. Add a `HeuristicMoveSelector` that prefers captures over non-captures, using
   `python-chess` `board.is_capture(move)`. Falls back to random.
2. Implement `create_move_selector(game_config: GameConfig) -> MoveSelector` factory function
   that reads `game_config.selector_type` and returns the correct implementation.
3. Write one test per selector confirming it returns a legal move from the starting position.

**Validation gate:**
- Both selectors return legal moves in tests.
- Factory function constructs the correct type from config.

---

### Milestone 5 — Board Coordinate Mapper

**Goal:** Bidirectional, config-driven mapping between chess notation and MuJoCo world
coordinates.

**Steps:**

1. **Implement `CoordinateMapper`** in `src/mujoco_chess/board/coordinate_mapper.py`:

   ```python
   class CoordinateMapper:
       def __init__(self, board_config: BoardConfig) -> None: ...
       def square_to_world(self, square: chess.Square) -> np.ndarray: ...
           # Returns (x, y, z) center of the square's top surface in world coords.
       def world_to_square(self, pos: np.ndarray) -> chess.Square | None: ...
           # Returns nearest square, or None if outside board.
       def square_to_file_rank(self, square: chess.Square) -> tuple[int, int]: ...
       def hover_position(self, square: chess.Square, hover_z: float) -> np.ndarray: ...
       def grasp_position(self, square: chess.Square, grasp_z: float) -> np.ndarray: ...
   ```

   **Orientation convention (must be consistent across the entire project):**
   - White plays from rank 1 (Y = board_origin_y). Rank 8 is at Y = board_origin_y + 7 * square_size.
   - File a is at X = board_origin_x. File h is at X = board_origin_x + 7 * square_size.
   - Square center world-coordinate formula:
     ```
     x = board_origin_x + file_index * square_size + square_size / 2
     y = board_origin_y + rank_index * square_size + square_size / 2
     z = board_origin_z + board_thickness   # top surface in world coords
     ```
   - `file_index = chess.square_file(square)`  (0 = a, 7 = h)
   - `rank_index = chess.square_rank(square)`  (0 = rank 1, 7 = rank 8)

   **Important:** `square_to_world` returns **world coordinates**. These are used for
   end-effector targets and slot positions. They must NOT be used directly as geom `pos`
   attributes inside a non-root MuJoCo body — see Milestone 7 for the XML coordinate rule.

2. Document the orientation convention as a docstring at the top of the module AND in
   `docs/environment.md`.

3. **Write tests** in `tests/unit/test_coordinate_mapper.py`:
   - `square_to_world(chess.A1)` returns expected (x, y, z) given known config values.
   - `square_to_world(chess.H8)` returns expected values.
   - `world_to_square` round-trips correctly for all 64 squares.
   - `world_to_square` returns None for a position 1 m off the board.
   - Hover and grasp positions differ from square centers by correct Z offsets.

**Validation gate:**
- All tests pass.
- Round-trip test passes for all 64 squares.

---

### Milestone 6 — Graveyard and Promotion Slot Manager

**Goal:** Config-driven, deterministic slot allocation for graveyard and promotion reserve.

**Steps:**

1. **Implement `SlotManager`** in `src/mujoco_chess/board/slot_manager.py`:

   ```python
   class SlotManager:
       def __init__(self, board_config: BoardConfig, pieces_config: PiecesConfig) -> None: ...

       # Graveyard
       def allocate_graveyard_slot(self, color: chess.Color) -> int: ...
           # Returns slot index for the given color (raises if full).
           # Slot indices are per-color: 0–15 for white, 0–15 for black.
       def free_graveyard_slot(self, color: chess.Color, slot_index: int) -> None: ...
           # Frees a slot; color is required to disambiguate the two strips.
           # Used on reset only.
       def graveyard_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray: ...
           # Returns world position for the given color's slot.

       # Pawn storage (for promoted pawns — separate from captured-piece graveyard)
       def allocate_pawn_storage_slot(self, color: chess.Color) -> int: ...
           # Returns slot index (0–7) for pawn storage of the given color.
       def free_pawn_storage_slot(self, color: chess.Color, slot_index: int) -> None: ...
       def pawn_storage_slot_world_pos(self, color: chess.Color, slot_index: int) -> np.ndarray: ...

       # Promotion reserve
       def get_reserve_slot(self, color: chess.Color, piece_type: chess.PieceType) -> int | None: ...
           # Returns the index of the first available reserve slot of that type/color.
       def mark_reserve_used(self, color: chess.Color, piece_type: chess.PieceType, slot: int) -> None: ...
       def reserve_slot_world_pos(self, color: chess.Color, piece_type: chess.PieceType, slot: int) -> np.ndarray: ...

       def reset(self) -> None: ...
   ```

   Layout details:
   - White graveyard strip: 2×8 grid at `(board_origin_x - 3*square_size, board_origin_y, board_surface_z)`.
     Slot spacing: `board_config.graveyard_slot_spacing`.
   - Black graveyard strip: 2×8 grid at `(board_origin_x + 9*square_size, board_origin_y, board_surface_z)`.
   - White pawn-storage: 1×8 grid, further out on the white (left) side beyond the graveyard.
   - Black pawn-storage: 1×8 grid, further out on the black (right) side beyond the graveyard.
     Slot spacing: `board_config.pawn_storage_slot_spacing`.
   - Promotion reserve: placed behind the board at `board_max_y + board_config.reserve_area_offset_y`.
     4 piece types × 2 slots × 2 colors, columns separated by `reserve_slot_spacing`.
     Layout must not overlap the arm base position (validated in startup — see Milestone 14).

2. **Write tests** in `tests/unit/test_slot_manager.py`:
   - Graveyard slot allocation fills slots in order 0, 1, 2 ... for each color independently.
   - Allocating > 16 graveyard slots for one color raises an error.
   - Allocating > 8 pawn-storage slots for one color raises an error.
   - `graveyard_slot_world_pos(chess.WHITE, 0)` and `graveyard_slot_world_pos(chess.BLACK, 0)`
     return distinct world positions.
   - `pawn_storage_slot_world_pos` returns a position distinct from graveyard and reserve.
   - `free_graveyard_slot(chess.WHITE, 0)` without the color arg does not compile (type check).
   - Reserve slot lookup returns correct position for each piece type and color.
   - `reset()` clears all allocations (graveyard, pawn-storage, reserve) for both colors.

**Validation gate:**
- All tests pass.

---

### Milestone 7 — MuJoCo XML Generator (Board Only)

**Goal:** A Python script generates a valid MuJoCo XML file containing the table, board, and
64 colored squares. No pieces, no arm yet.

**Coordinate convention for XML generation (critical rule):**

MuJoCo geom `pos` attributes inside a `<body>` are **local** to that body, not world
coordinates. The implementation must use one of the following approaches — not mix them:

**Option B (chosen):** Place `board_frame` body at `(board_origin_x, board_origin_y,
board_origin_z)`. All square geom positions inside `board_frame` are **local offsets** relative
to `board_frame`'s origin:

```
local_x = file_index * square_size + square_size / 2
local_y = rank_index * square_size + square_size / 2
local_z = board_thickness / 2
```

Do NOT use `CoordinateMapper.square_to_world` for geom positions inside `board_frame`. That
method returns world coordinates and would double-apply the board origin offset. Use
`CoordinateMapper.square_to_world` only for computing end-effector targets and slot positions
in world space (arm motion control).

**Steps:**

1. **Implement `XMLGenerator`** in `src/mujoco_chess/mujoco_env/xml_generator.py`.

   **Board physics rule:** The 64 colored squares are **visual-only geoms**
   (`contype="0" conaffinity="0"`). The physical support surface for pieces is a **single
   continuous box geom** (`board_surface`). This avoids contact seams between squares that
   can cause piece jitter, drift, or unstable contacts.

   Structure of the generated XML:
   ```xml
   <mujoco model="chess">
     <compiler .../>
     <option .../>
     <asset>
       <!-- materials/textures for light/dark squares, board frame, table -->
     </asset>
     <worldbody>
       <light .../>
       <geom name="table" type="box" .../>      <!-- table surface, world coords -->
       <body name="board_frame" pos="{board_origin_x} {board_origin_y} {board_origin_z}">
         <geom name="board_base" type="box" .../>  <!-- board border/frame, local coords -->
         <geom name="board_surface" type="box"
               size="{4*square_size} {4*square_size} {board_thickness/2}"
               pos="{3.5*square_size} {3.5*square_size} 0"
               contype="1" conaffinity="1" rgba="{board_rgba}"/>
               <!-- SINGLE physical collision surface for the play area -->
         <!-- 64 visual-only square geoms, contype="0" conaffinity="0" -->
         <!-- placed at board_thickness/2 + epsilon above board_surface top -->
       </body>
     </worldbody>
   </mujoco>
   ```

   Implementation rules:
   - All positions and sizes computed from `BoardConfig` — zero hardcoded numbers.
   - One `board_surface` box geom covers the entire 8×8 play area with `contype="1"`.
   - 64 square geoms are generated in a double loop; each has `contype="0" conaffinity="0"`.
   - Square name: `f"square_{chess.FILE_NAMES[f]}{r+1}"`.
   - Square local position: `(f * square_size + square_size/2, r * square_size + square_size/2,
     board_thickness/2)` — offsets within `board_frame`, placed as a thin visual overlay.
   - Square color: alternate rgba from config based on `(f + r) % 2`.
   - Output written to `generated/chess_env.xml`.

2. Call `XMLGenerator.generate()` from `startup.py` before any MuJoCo load.

3. **Write tests** in `tests/simulation/test_xml_generation.py`:
   - `generate()` produces a file at `generated/chess_env.xml`.
   - The XML is valid (parse with `xml.etree.ElementTree`).
   - Exactly 64 square geoms are present, all with `contype="0"`.
   - A single `board_surface` geom exists with `contype="1"`.
   - Square names follow the expected pattern.
   - `board_frame` body pos matches `board_origin_x/y/z` from config.
   - Square `a1` local position is `(square_size/2, square_size/2, board_thickness/2)`.
   - Square `h8` local position is `(7.5 * square_size, 7.5 * square_size, board_thickness/2)`.

**Validation gate:**
- Tests pass.
- XML file can be loaded by MuJoCo without errors (tested in Milestone 10).

---

### Milestone 8 — Chess Pieces and Reserve Pieces in XML

**Goal:** All 32 starting pieces **and** all 16 promotion reserve pieces are added to the XML
with stable primitive collision geometry. No STL yet. All pieces spawn upright.

**Steps:**

1. **Extend `XMLGenerator`** to add the 32 active starting pieces:

   Piece bodies are placed directly under `<worldbody>` (not inside `board_frame`), using
   world coordinates from `CoordinateMapper.square_to_world`:

   ```
   x, y = square_to_world(square)[0:2]   # world X/Y of square center
   z = board_origin_z + board_thickness + pieces.base_height / 2   # world Z for body origin
   ```

   For each starting piece:
   ```xml
   <body name="piece_w_P_0" pos="{x} {y} {z}">
     <freejoint/>
     <geom name="base_w_P_0" type="cylinder"
           size="{base_radius} {base_height/2}"
           pos="0 0 0"
           mass="{mass}" friction="{friction}" solimp="{solimp}" solref="{solref}"
           rgba="{white_rgba}"/>
     <geom name="shaft_w_P_0" type="capsule"
           size="{shaft_radius} {shaft_height/2}"
           pos="0 0 {base_height/2 + shaft_height/2}"
           contype="1" conaffinity="1"
           rgba="{white_rgba}"/>
   </body>
   ```

   Body name format: `piece_{color_char}_{type_char}_{id}` where:
   - `color_char`: `w` for white, `b` for black.
   - `type_char`: `P` pawn, `R` rook, `N` knight, `B` bishop, `Q` queen, `K` king.
   - `id`: integer 0-based per (color, type) combination.

2. **Also generate 16 promotion reserve piece bodies** in the same step:

   Reserve pieces use the same cylinder+capsule geometry. They are positioned at the world
   coordinates returned by `SlotManager.reserve_slot_world_pos(color, piece_type, slot)`.

   Name format: `piece_reserve_{color_char}_{type_char}_{slot_idx}`.
   Example: `piece_reserve_w_Q_0`, `piece_reserve_w_Q_1`, `piece_reserve_b_R_0`, etc.

   Piece types in reserve: Q (2 slots), R (2 slots), B (2 slots), N (2 slots) × 2 colors = 16.

   Reserve pieces also get `<freejoint/>` so they can be physically moved.

3. Add the following static platform geoms to the XML (flat box geoms under `worldbody`,
   no freejoint, world coordinates):
   - White and black captured-piece graveyard platforms.
   - White and black pawn-storage platforms (separate from graveyard).
   - Promotion reserve area platform.
   All three types are static surfaces. Their positions come from `SlotManager` world positions.

4. **Write tests** in `tests/simulation/test_xml_generation.py` (extend):
   - All 32 active starting pieces are present.
   - All 16 reserve pieces are present (names follow `piece_reserve_*` pattern).
   - Each piece body (active and reserve) has exactly one `freejoint`.
   - Active piece spawn Z matches the formula `board_surface_z + base_height / 2`.
   - No piece body name is duplicated.
   - Reserve piece positions match `SlotManager.reserve_slot_world_pos` output.

**Validation gate:**
- Tests pass.
- XML loads in MuJoCo without errors.

---

### Milestone 9 — STL Visual Support

**Goal:** STL meshes are optionally attached as visual-only geoms. The system works whether or
not STL files exist (falls back to primitive-only visuals).

**Steps:**

1. Create placeholder STL files (empty or simple shapes) in `assets/stl/` with names:
   `pawn.stl`, `rook.stl`, `knight.stl`, `bishop.stl`, `queen.stl`, `king.stl`.

2. **Extend `XMLGenerator`**:
   - If `pieces_config.stl_scale > 0` and the STL file exists, add to each piece body
     (active and reserve):
     ```xml
     <geom name="visual_{id}" type="mesh" mesh="{piece_type}"
           contype="0" conaffinity="0" rgba="{color_rgba}" pos="0 0 0"/>
     ```
   - Add `<asset>` entries: `<mesh name="{type}" file="assets/stl/{type}.stl" scale="..."/>`.
   - If STL file missing or `stl_scale == 0`: skip silently (log a WARNING).

3. STL visual geom must have `contype="0" conaffinity="0"` — it must never participate in
   collision.

4. Extend `test_xml_generation.py`:
   - When STL files exist, mesh assets are declared in `<asset>`.
   - Visual geoms have `contype="0"`.
   - When STL file is missing, generation still succeeds (no crash, warning logged).

**Validation gate:**
- XML loads with STL visuals (if files present).
- XML loads cleanly without STL files (no crash, no collision geom generated from STL).

---

### Milestone 10 — MuJoCo Environment Loader

**Goal:** `MuJoCoEnv` can load the generated XML, step the sim, query body states, and open
the passive viewer.

**Steps:**

1. **Implement `MuJoCoEnv`** in `src/mujoco_chess/mujoco_env/env.py`:

   ```python
   class MuJoCoEnv:
       def __init__(self, xml_path: str, config: AppConfig) -> None: ...
       def load(self) -> None: ...
           # mujoco.MjModel.from_xml_path(xml_path) + mujoco.MjData(model)
       def step(self, n: int = 1) -> None: ...
           # mujoco.mj_step(model, data) × n
       def get_body_pos(self, body_name: str) -> np.ndarray: ...
           # data.body(body_name).xpos.copy()
       def get_body_quat(self, body_name: str) -> np.ndarray: ...
           # data.body(body_name).xquat.copy()
       def get_body_vel(self, body_name: str) -> tuple[np.ndarray, np.ndarray]: ...
           # Returns (lin_vel, ang_vel) from data.body(body_name).cvel
       def get_contacts(self) -> list[ContactInfo]: ...
           # See ContactInfo note below.
       def set_body_freejoint_pos(self, body_name: str, pos: np.ndarray) -> None: ...
           # DEBUG ONLY — see freejoint note below.
       def set_mocap_pos(self, mocap_name: str, pos: np.ndarray) -> None: ...
           # data.mocap_pos[mocap_id] = pos
       def set_ctrl(self, actuator_name: str, value: float) -> None: ...
           # data.ctrl[model.actuator(actuator_name).id] = value
       def get_ee_pos(self) -> np.ndarray: ...
           # data.site(arm_config.ee_site_name).xpos.copy()
       def set_ee_target(self, pos: np.ndarray) -> None: ...
           # set_mocap_pos(arm_config.mocap_body_name, pos)
       def open_viewer(self) -> None: ...
           # mujoco.viewer.launch_passive(model, data)
       def close(self) -> None: ...
   ```

   **`ContactInfo` and contact force extraction:**
   MuJoCo does not expose a simple `force` field on contact objects. Contact force must be
   computed with `mujoco.mj_contactForce(model, data, contact_idx, result)` where `result`
   is a 6-element array (3 force + 3 torque). Implement as:
   ```python
   @dataclass
   class ContactInfo:
       geom1_name: str
       geom2_name: str
       body1_name: str
       body2_name: str
       force_magnitude: float   # norm of the 3D contact force
   ```
   Populate `force_magnitude` by calling `mujoco.mj_contactForce` for each contact pair.

   **Freejoint body teleportation (debug only):**
   For a body with a `freejoint`, `body.xpos` is derived (read-only). To teleport it, find the
   body's freejoint `qpos` slice in `data.qpos` and write the new position there, then call
   `mujoco.mj_forward(model, data)` to propagate. Implementation:
   ```python
   def set_body_freejoint_pos(self, body_name: str, pos: np.ndarray) -> None:
       # Finds the freejoint for this body, updates data.qpos[joint_qpos_adr:+3],
       # then calls mujoco.mj_forward(model, data).
       # This method is DEBUG ONLY and must not be called from normal gameplay code.
   ```

2. The viewer should be optional (disabled in headless/test mode controlled by
   `DebugConfig.enabled`).

3. Ensure `XMLGenerator` never generates any site or geom named `target` or `goal` (no Fetch
   red dot).

4. **Write tests** in `tests/simulation/test_env_load.py`:
   - `MuJoCoEnv.load()` succeeds.
   - `get_body_pos("board_frame")` returns approximately expected position.
   - `step(100)` does not raise.
   - No geom or site named `target` or `goal` exists in the loaded model.
   - `get_contacts()` returns a list (may be empty).
   - `get_ee_pos()` does not raise after arm is added (tested again in Milestone 13).

5. **Create the debug CLI skeleton** in `src/mujoco_chess/debug/cli.py` using `argparse`.
   Add the first command: `run-env`. Additional commands will be added incrementally in
   later milestones as each capability is implemented.

   ```python
   # Command: run-env
   # Loads the environment, runs initial_settle_steps, keeps viewer open.
   # Prints board_frame position and confirms no unexpected contacts.
   ```

**Validation gate:**
- All tests pass.
- `python -m mujoco_chess.debug run-env` opens the viewer and shows the board.

---

### Milestone 11 — Initial Settle Phase and Piece Stability Checks

**Goal:** After environment load, all pieces (active and reserve) settle to stable positions.

**Steps:**

1. **Implement `SettlePhase`** in `src/mujoco_chess/mujoco_env/env.py` (or a helper module):
   - Step the simulation for `waypoint_config.initial_settle_steps` steps (from config, not
     hardcoded).
   - After settling, return each piece's final world position for registry initialization.

2. **Implement initial stability checks** (standalone functions at this stage — the formal
   `HealthCheckRunner` is built in Milestone 18):
   - For each active piece (32) and each reserve piece (16):
     - Z position ≥ `board_surface_z - health_config.piece_sink_threshold_m`.
     - Z position ≤ `board_surface_z + base_height + health_config.piece_float_threshold_m`.
     - Velocity magnitude ≤ `health_config.unexpected_velocity_threshold`.
     - Tilt ≤ `health_config.piece_tilt_threshold_deg` (check quaternion w component).
   - Log pass/fail per piece.
   - Raise `SettleFailureError` with detailed diagnostics if any check fails.

3. **Write tests** in `tests/simulation/test_piece_stability.py`:
   - After `initial_settle_steps` steps, all 48 pieces (32 active + 16 reserve) pass checks.
   - A piece deliberately spawned at wrong height fails the sink check.

4. **Extend debug CLI** (`debug/cli.py`): add `inspect-piece-stability` command that runs
   the settle phase, prints per-piece pass/fail, and saves a report.

**Validation gate:**
- Tests pass.
- `python -m mujoco_chess.debug run-env` prints a stability report after settle.
- `python -m mujoco_chess.debug inspect-piece-stability` outputs pass/fail for all 48 pieces.

---

### Milestone 12 — Physical Piece Registry

**Goal:** A registry maps logical chess state to concrete MuJoCo body names and expected
positions, for both active pieces and reserve pieces.

**Steps:**

1. **Implement `PhysicalPieceRegistry`** in `src/mujoco_chess/registry/piece_registry.py`:

   ```python
   class PieceStatus(Enum):
       ACTIVE = "ACTIVE"
       CAPTURED = "CAPTURED"
       RESERVE = "RESERVE"
       PROMOTED = "PROMOTED"              # reserve piece now active after promotion
       PROMOTION_PAWN_STORED = "PROMOTION_PAWN_STORED"  # original pawn stored after promotion

   @dataclass
   class PieceRecord:
       body_name: str              # MuJoCo body name, e.g. "piece_w_P_0"
       color: chess.Color
       piece_type: chess.PieceType
       status: PieceStatus
       square: chess.Square | None          # current square if ACTIVE or PROMOTED
       graveyard_slot: int | None           # slot index (per-color) if CAPTURED
       graveyard_color: chess.Color | None  # which strip the slot belongs to
       pawn_storage_slot: int | None        # slot index if PROMOTION_PAWN_STORED
       reserve_slot: int | None             # if RESERVE
       expected_pos: np.ndarray             # world position after last committed move

   class PhysicalPieceRegistry:
       def __init__(self, env: MuJoCoEnv, board: chess.Board) -> None: ...

       def initialize(self, settled_positions: dict[str, np.ndarray]) -> None: ...
           # Populate ACTIVE records for 32 starting pieces.
           # Populate RESERVE records for 16 reserve pieces.
           # Use settled_positions (from SettlePhase) as expected_pos for each body.

       def get_piece_at(self, square: chess.Square) -> PieceRecord | None: ...
       def get_piece_by_body(self, body_name: str) -> PieceRecord | None: ...

       def move_piece(self, body_name: str, to_square: chess.Square,
                      new_expected_pos: np.ndarray) -> None: ...
       def capture_piece(self, body_name: str,
                         graveyard_color: chess.Color, graveyard_slot: int,
                         slot_pos: np.ndarray) -> None: ...
       def store_promoted_pawn(self, pawn_body: str,
                               pawn_storage_slot: int, slot_pos: np.ndarray) -> None: ...
           # pawn → PROMOTION_PAWN_STORED at pawn_storage_slot.
       def activate_reserve_piece(self, reserve_body: str,
                                   to_square: chess.Square, new_pos: np.ndarray) -> None: ...
           # reserve body → PROMOTED status (treated as ACTIVE) at to_square.
       def reset(self) -> None: ...

       def get_all_active(self) -> list[PieceRecord]: ...
       def get_all_captured(self) -> list[PieceRecord]: ...
       def get_all_reserve(self) -> list[PieceRecord]: ...
       def validate_sync(self, logical_board: chess.Board) -> list[SyncError]: ...
           # Returns list of discrepancies between registry and logical board.
   ```

2. `validate_sync` must check (promotion-aware):
   - Every piece on the logical board has a corresponding ACTIVE or PROMOTED record in the
     registry at the correct square.
   - For promoted pieces: the logical board has a promoted piece type (Q/R/B/N) at the
     promotion square; the registry must have a PROMOTED record (originally a reserve body)
     at that square. The original pawn body should be PROMOTION_PAWN_STORED, not ACTIVE.
   - Every ACTIVE/PROMOTED record at a square matches the logical board (color + type).
   - Number of CAPTURED + PROMOTION_PAWN_STORED records is consistent with logical captures
     and promotions.
   - `validate_sync` does NOT require the same physical body to persist through promotion;
     it validates square occupancy and piece type, allowing body identity substitution.

3. **Write tests** in `tests/unit/test_piece_registry.py`:
   - `initialize` creates 32 ACTIVE records AND 16 RESERVE records.
   - `get_piece_at(chess.E2)` returns the white pawn at e2.
   - `move_piece` updates `square` and `expected_pos`.
   - `capture_piece` sets status CAPTURED, clears square, stores graveyard_color and slot.
   - `store_promoted_pawn` marks pawn PROMOTION_PAWN_STORED with pawn_storage_slot.
   - `activate_reserve_piece` marks reserve body PROMOTED at the target square.
   - `validate_sync` on a board where white promoted e7→e8=Q: passes when registry has
     the correct PROMOTED record and PROMOTION_PAWN_STORED pawn record.
   - `validate_sync` returns empty list for a consistent state.
   - `validate_sync` returns errors for a deliberate mismatch.
   - `get_all_reserve()` returns 16 records initially.

**Validation gate:**
- All tests pass.
- `run-env` command (extended) prints registry state after settle.

---

### Milestone 13 — Add Fetch-Style Robotic Arm to XML

**Goal:** The Fetch-style arm is present in the generated XML, positioned behind the board,
with no intersection with the board or pieces.

**Steps:**

1. **Source the Fetch arm assets:**
   - Copy the Fetch robot MJCF (or relevant subset) from MuJoCo Menagerie or Gymnasium
     Robotics into `assets/robot/fetch_arm.xml` (or inline it in the generator).
   - Record in `configs/arm.yaml` and `docs/environment.md`:
     - Robot base body name (e.g. `robot0:base`).
     - End-effector site name (e.g. `robot0:grip_site`).
     - Left gripper finger actuator name (e.g. `robot0:l_gripper_finger_joint`).
     - Right gripper finger actuator name (e.g. `robot0:r_gripper_finger_joint`).
   - Strip all Fetch goal/target visualizations from the sourced XML before use.

2. **Extend `XMLGenerator`** to include the Fetch arm:
   - Include the arm body hierarchy with the base positioned at
     `(arm_config.base_x, arm_config.base_y, arm_config.base_z)`.
   - All arm body positions are offsets relative to the arm base — no absolute hardcoding.
   - Add a `mocap` body named `ee_target` (or `arm_config.mocap_body_name`) to `worldbody`.
   - Add an `equality → weld` constraint linking `ee_target` to `arm_config.ee_weld_body_name`
     (a **body**, not a site — MuJoCo weld constraints require bodies):
     ```xml
     <equality>
       <weld body1="ee_target" body2="{arm_config.ee_weld_body_name}" relpose="0 0 0 1 0 0 0"/>
     </equality>
     ```
   - End-effector **position measurement** (`get_ee_pos()`) reads from the site named
     `arm_config.ee_site_name`, not from the weld body.
   - Do NOT include any red target site, goal marker, or site/geom named `target` or `goal`.

3. **Extend `MuJoCoEnv`** to expose:
   - `get_ee_pos() -> np.ndarray` — world position of the end-effector site.
   - `set_ee_target(pos: np.ndarray) -> None` — sets mocap body position.

4. **Write tests** in `tests/simulation/test_env_load.py` (extend):
   - Arm base body exists at the expected world position from config.
   - No body, site, or geom named `target` or `goal` exists in the model.
   - `get_ee_pos()` returns a plausible position (above table surface).
   - The mocap body `ee_target` exists in the model.

**Validation gate:**
- XML loads with arm.
- `run-env` shows arm visible and positioned correctly.
- No visual target dot.

---

### Milestone 14 — Arm–Board Placement and Layout Validation

**Goal:** Confirm arm base does not intersect the board; graveyard, pawn-storage, and reserve
areas do not intersect each other, the board, or the arm; all are reachable.

**Steps:**

1. After initial settle, run a placement check:
   - Check that no arm body is in contact with the board frame or any chess piece.
   - Check that the arm base Z is at table level (no floating, no sinking).
   - Log all contacts at startup.

2. Run a **layout-intersection check** using geometry (not MuJoCo contacts) to verify that
   the bounding boxes of the following areas do not overlap:
   - Board play area.
   - White graveyard strip.
   - Black graveyard strip.
   - White pawn-storage strip.
   - Black pawn-storage strip.
   - Promotion reserve area.
   - Arm base footprint (approximated from config).

3. If any contacts or intersections are found, log a clear `PLACEMENT_ERROR` and abort startup.

4. Add both checks to `startup.py`.

5. **Write tests** in `tests/simulation/test_env_load.py`:
   - No arm–board contacts after initial settle.
   - No arm–piece contacts after initial settle.
   - No arm–graveyard contacts after initial settle.
   - Reserve platform does not geometrically intersect the arm base bounding box.
   - Graveyard strips do not geometrically intersect the board or each other.
   - Pawn-storage strips do not geometrically intersect the graveyard or board.
   - All 48 pieces (active + reserve) are stable after initial settle (no sinking or floating).

**Validation gate:**
- Tests pass.
- Startup succeeds cleanly with no unexpected contacts logged.

---

### Milestone 15 — Reachability Validation Tool

**Goal:** Confirm the arm can reach every required position with crane-mode path safety,
not just hover-point reachability.

**Steps:**

1. **Implement `ReachabilityChecker`** in `src/mujoco_chess/debug/scenarios.py` with two
   validation levels:

   **Level 1 — Point reachability:**
   For each target position (64 squares + 32 graveyard slots + 16 reserve slots):
   - Set `ee_target` to the hover position above it.
   - Run `waypoint_config.stage_timeout_steps` settle steps.
   - Check that actual ee position reaches within `position_tolerance` of target.
   - Record pass/fail.

   **Level 2 — Crane-mode action reachability:**
   For a representative sample of positions (at minimum: all 4 board corners, center squares,
   all graveyard slot 0s, all reserve slots):
   - Execute hover → descend-to-grasp-height → complete-halt → ascend-to-hover → complete-halt.
   - Check at each sub-stage: ee reaches target, no arm–board collision, no arm–piece collision.
   - Record pass/fail per sub-stage.

2. **Add debug command** `check-reachability`:
   ```bash
   python -m mujoco_chess.debug check-reachability [--level 1|2]
   ```
   - Level 1: hover-point check for all positions.
   - Level 2: full crane-mode path check for the sample set.
   - Prints an 8×8 grid for board squares, lists graveyard/reserve results.
   - Saves a report to `logs/reachability_report.txt`.

3. If any position fails, print suggestions to adjust `arm.yaml` (shift base_y, increase
   home_position Z, etc.).

4. **Write tests** in `tests/simulation/test_reachability.py`:
   - All 64 squares pass Level 1.
   - All 32 graveyard slots pass Level 1.
   - All 16 reserve slots pass Level 1.
   - The 4 board corners pass Level 2 (crane-mode path).

5. **Extend debug CLI**: `check-reachability` command is now available.

**Validation gate:**
- All reachability tests pass.
- `check-reachability --level 1` reports 100% pass rate.
- `check-reachability --level 2` reports 100% pass rate for the sample set.

---

### Milestone 16 — Basic End-Effector Movement

**Goal:** The arm's end-effector moves smoothly from one position to another over multiple
simulation steps.

**Steps:**

1. **Implement `MotionExecutor`** skeleton in `src/mujoco_chess/motion/executor.py`:

   ```python
   class MotionExecutor:
       def __init__(self, env: MuJoCoEnv, config: WaypointConfig) -> None: ...

       def move_to(self, target_pos: np.ndarray,
                   timeout_steps: int | None = None) -> StageResult: ...
           # Interpolates toward target over multiple steps.
           # Stops when position within tolerance AND velocity below threshold.
           # Returns StageResult(success, steps_taken, final_pos, final_vel).

       def _step_toward(self, target: np.ndarray) -> None: ...
           # Computes next ee_target increment and calls env.set_ee_target.
   ```

   Movement strategy:
   - Each step, compute `delta = target - current_ee_pos`.
   - If `|delta| > transit_speed`: move by `transit_speed` in the direction of delta.
   - Else: snap to target.
   - Call `env.set_ee_target(new_target)` then `env.step(1)`.
   - Repeat until `|error| < position_tolerance` AND velocity < threshold, or timeout.

2. `StageResult` dataclass: `{ success: bool, steps_taken: int, final_pos: np.ndarray,
   final_vel: float, error_message: str | None }`.

3. All speed and tolerance values come from `WaypointConfig`.

4. **Write tests** in `tests/simulation/test_physical_moves.py`:
   - `move_to` reaches a target within tolerance in < `timeout_steps` steps.
   - `move_to` returns `success=False` for an unreachable target.
   - Velocity is below threshold when `move_to` succeeds.

5. **Extend debug CLI**: add `move-ee-to --x X --y Y --z Z` command for manual testing.

**Validation gate:**
- Tests pass.
- `move-ee-to` command visually moves the end-effector.

---

### Milestone 17 — Gripper Command Interface

**Goal:** The gripper can be commanded to open, commanded to grasp, and commanded to hold.
The `GripperController` only issues actuator commands — it does **not** step the simulation
or run settle windows. All stepping and settling is owned by `MotionExecutor`.

**Steps:**

1. **Implement `GripperController`** in `src/mujoco_chess/motion/gripper.py`:

   ```python
   class GripperController:
       def __init__(self, env: MuJoCoEnv, config: ArmConfig) -> None: ...

       def command_open(self) -> None: ...
           # Sets both finger actuators toward open_width.
           # Single command — does NOT step or settle.
       def command_grasp(self) -> None: ...
           # Sets both finger actuators toward grasp_width.
           # Single command — does NOT step or settle.
       def command_hold(self) -> None: ...
           # Re-issues the grasp_width command to maintain tension.
           # Called by MotionExecutor each step during carry settle windows.
           # Single command — does NOT step.
       def get_width(self) -> float: ...
           # Returns current measured gripper finger separation from data.qpos.
       def is_open(self) -> bool: ...
           # Returns True if width is within tolerance of open_width.
       def is_grasping(self) -> bool: ...
           # Returns True if width is within tolerance of grasp_width.
   ```

2. Actuator indices are looked up by name at load time using `arm_config.gripper_left_actuator`
   and `arm_config.gripper_right_actuator`. Never hardcode actuator indices.

3. Gradual opening/closing is achieved by `MotionExecutor` issuing `command_open()` /
   `command_grasp()` each step over `waypoint_config.gripper_settle_steps` steps, stepping
   between each call. `GripperController` itself does not loop.

4. **Write tests** (integration-level, requires env):
   - After `command_open()` + `env.step(gripper_settle_steps)`: `is_open()` returns True.
   - After `command_grasp()` + `env.step(gripper_settle_steps)`: `is_grasping()` returns True.
   - `get_width()` changes after gripper commands.

**Validation gate:**
- Tests pass.

---

### Milestone 18 — Minimal Health Check Framework

**Goal:** The `HealthCheckResult` type and `HealthCheckRunner` exist and core checks are
implemented **before** grasp/release/waypoint tests begin. These checks will be used
immediately in Milestones 19–21 to catch physics problems early.

**Steps:**

1. **Implement `HealthCheckResult`** in `src/mujoco_chess/health/checks.py`:

   ```python
   @dataclass
   class HealthCheckResult:
       check_name: str
       passed: bool
       message: str
       details: dict   # positions, velocities, contacts, threshold values, etc.
   ```

2. **Implement `HealthCheckRunner`** in `src/mujoco_chess/health/runner.py`:

   ```python
   @dataclass
   class HealthCheckContext:
       # Carries transaction-local state so checks use pending positions, not committed registry.
       transaction: MoveExecutionTransaction | None = None
       active_stage: WaypointStage | None = None
       target_piece_body: str | None = None      # body being grasped/moved
       held_piece_body: str | None = None        # body currently in gripper
       # Pending expected positions from the transaction (override registry.expected_pos).
       expected_positions_override: dict[str, np.ndarray] = field(default_factory=dict)
       # Bodies to skip for drift checks (e.g. a piece mid-transit whose old pos is stale).
       ignored_bodies_for_drift: set[str] = field(default_factory=set)

   class CheckContext(Enum):
       STARTUP = "STARTUP"
       SETTLE_WINDOW_BEFORE = "SETTLE_WINDOW_BEFORE"
       SETTLE_WINDOW_AFTER = "SETTLE_WINDOW_AFTER"
       POST_GRASP = "POST_GRASP"
       DURING_CARRY = "DURING_CARRY"
       POST_RELEASE = "POST_RELEASE"
       START_OF_TURN = "START_OF_TURN"
       END_OF_TURN = "END_OF_TURN"

   class HealthCheckRunner:
       def register(self, context: CheckContext, check_fn: Callable) -> None: ...
       def run_checks(self, context: CheckContext,
                      hc_context: HealthCheckContext | None = None,
                      **kwargs) -> list[HealthCheckResult]: ...
           # Passes hc_context to every check function so they use pending positions.
       def all_passed(self, results: list[HealthCheckResult]) -> bool: ...
       def handle_failure(self, results: list[HealthCheckResult],
                          context_info: dict) -> None: ...
           # Writes failure report JSON and raises HealthCheckFailureError.
   ```

   All health check functions that check position/drift must use
   `hc_context.expected_positions_override[body_name]` when available, falling back to
   `registry.get_piece_by_body(body_name).expected_pos` otherwise. Similarly, bodies in
   `hc_context.ignored_bodies_for_drift` are skipped for drift checks.

3. **Implement the core check functions** in `src/mujoco_chess/health/checks.py`:
   Each has signature: `def check_*(env, registry, config, **kwargs) -> HealthCheckResult`.

   | Function | What it checks |
   |---|---|
   | `check_piece_not_tilted` | Tilt angle of piece local-Z vs world-Z < `piece_tilt_threshold_deg`. Computed as `arccos(dot(rotate(local_z, body_quat), world_z))`. Do NOT use raw quaternion `w` — that does not robustly represent tilt angle. |
   | `check_piece_not_drifted` | Position distance from `hc_context.expected_positions_override` (or `registry.expected_pos` as fallback) < drift threshold. Skip bodies in `hc_context.ignored_bodies_for_drift`. |
   | `check_piece_not_sunk` | Z ≥ `board_surface_z - sink_threshold` |
   | `check_piece_not_floating` | Z ≤ `board_surface_z + base_height + float_threshold` |
   | `check_piece_not_fallen` | Z ≥ `table_z - 0.05` |
   | `check_no_unexpected_velocity` | piece velocity < unexpected_velocity_threshold |
   | `check_complete_halt` | ee linear vel, ee angular vel, all robot joint velocities, all gripper joint velocities, and (if carrying) held piece velocity — all below configured thresholds |
   | `check_robot_joints_halted` | all robot arm joint velocities < `robot_joint_velocity_threshold` |
   | `check_grasp_contact` | gripper–piece contact exists (after CLOSE_GRIPPER) |
   | `check_piece_not_slipping` | held piece velocity relative to gripper < slip threshold |
   | `check_gripper_hold_tension` | gripper width ≈ hold_width during carry |
   | `check_piece_released` | gripper–piece contact gone, piece stable, vel < `released_piece_velocity_threshold` |
   | `check_gripper_orientation_crane_mode` | gripper orientation within angular tolerance of `arm_config.crane_down_quat` |
   | `check_no_arm_board_collision` | no arm–board contacts |
   | `check_no_arm_piece_collision` | no arm–non-target piece contacts |
   | `check_no_piece_piece_collision` | no piece–piece contacts |
   | `check_registry_sync` | `registry.validate_sync` returns empty list |

4. **Implement the failure report writer:**
   ```python
   def write_failure_report(results: list[HealthCheckResult], context_info: dict,
                             log_dir: str) -> str:
       # Writes logs/failures/failure_{timestamp}.json.
       # Returns the file path.
   ```
   Report structure:
   ```json
   {
     "timestamp": "...",
     "chess_move": "...",
     "physical_action_index": 0,
     "waypoint_stage": "...",
     "failed_checks": [...],
     "piece_ids": [...],
     "positions": {...},
     "orientations": {...},
     "velocities": {...},
     "contacts": [...],
     "config_snapshot": {...}
   }
   ```

5. **Write tests** in `tests/unit/test_health_checks.py`:
   - Each check function returns PASS for valid input values.
   - Each check function returns FAIL for deliberately invalid input values.
   - `HealthCheckRunner.all_passed` returns True only when all results pass.
   - `handle_failure` writes a JSON report to `logs/failures/`.
   - `HealthCheckFailureError` is raised by `handle_failure`.

**Validation gate:**
- All health check unit tests pass.
- `HealthCheckRunner` can be instantiated and used in isolation without MuJoCo.

---

### Milestone 19 — Single-Piece Grasp Test

**Goal:** The arm can descend to a piece, close the gripper, and lift the piece without
dropping it. Health checks run during this test.

**Steps:**

1. Implement a `test_grasp` helper in `src/mujoco_chess/debug/scenarios.py`:
   - Move ee to hover above a given square.
   - Run a pre-descend settle window and call `health_runner.run_checks(SETTLE_WINDOW_BEFORE)`.
   - Descend to grasp height.
   - Call `gripper.command_grasp()` for `gripper_settle_steps` steps.
   - Run post-grasp settle window and call `health_runner.run_checks(POST_GRASP)`.
   - Ascend to safe hover height.
   - Run post-ascent settle window and call `health_runner.run_checks(DURING_CARRY)`.
   - Verify piece moved up with the arm (piece Z ≈ arm Z − grasp_offset).

2. **Extend debug CLI**: add `test-grasp --square SQUARE [--repeat N]` command.

3. **Write tests** in `tests/simulation/test_physical_moves.py`:
   - Grasp test on e2 succeeds: piece lifts off board.
   - Piece Z after ascent is within tolerance of expected lift height.
   - Piece does not fall during ascent.
   - POST_GRASP health check passes.
   - DURING_CARRY health check passes.

**Validation gate:**
- Tests pass.
- Visual test via debug command confirms lift.

---

### Milestone 20 — Single-Piece Release Test

**Goal:** The arm can descend with a piece and release it gently onto a target square without
the piece toppling or drifting. Health checks verify the release.

**Steps:**

1. Extend `test_grasp` scenario to include release:
   - After ascent (from Milestone 19), move to a target square hover position.
   - Descend to release height.
   - Call `gripper.command_open()` for `gripper_settle_steps` steps.
   - Run post-release settle window and call `health_runner.run_checks(POST_RELEASE)`.
   - Ascend.
   - Verify piece is stable on target square.

2. **Extend debug CLI**: add `test-release --square SQUARE [--repeat N]` command.

3. **Write tests**:
   - Piece released at e4 is within drift tolerance of e4 center.
   - Piece is upright (tilt < threshold).
   - Piece velocity < threshold after release settle.
   - POST_RELEASE health check passes.

**Validation gate:**
- Tests pass.
- Visual test confirms piece stays put after release.

---

### Milestone 21 — Full Waypoint Algorithm (One Piece)

**Goal:** The complete 10-stage waypoint algorithm runs for a single piece move, with
settle windows, complete-halt verification, and health checks at each stage.

**Steps:**

1. **Implement `WaypointPlan`** in `src/mujoco_chess/motion/waypoint.py`:

   ```python
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

   @dataclass
   class WaypointPlan:
       piece_body: str
       source_pos: np.ndarray
       destination_pos: np.ndarray
       hover_z: float
       grasp_z: float
       release_z: float
       crane_orientation: np.ndarray  # target gripper quaternion (from arm_config.crane_down_quat)
   ```

2. **Extend `MotionExecutor`** with `execute_plan(plan: WaypointPlan) -> MotionExecutionResult`:

   For each active stage:
   - a. Log stage start.
   - b. Run pre-stage settle window via `_run_settle_window(stage, carrying=..., is_before=True)`.
   - c. Execute the stage action (motion command or gripper command).
   - d. Wait for complete halt via `_check_complete_halt(carrying=...)`.
   - e. Run post-stage settle window via `_run_settle_window(stage, carrying=..., is_before=False)`.
   - f. Run stage-specific health checks via `health_runner.run_checks(context)`.
   - g. On any failure: call `health_runner.handle_failure(results, context_info)` and return
      `MotionExecutionResult(success=False, failed_stage=stage)`.
   - h. Log stage completion.

3. **`_run_settle_window(stage, carrying, is_before)`:**
   - Loop `waypoint_config.settle_steps` times:
     - **Maintain the current `ee_target`** — do not change it.
     - **If carrying a piece: call `gripper.command_hold()` every step** to maintain tension.
     - Call `env.step(1)`.
   - Return stability summary (velocities, contacts).

4. **`_check_complete_halt(carrying: bool)`:**
   - Polls up to `stage_timeout_steps` steps.
   - Each poll step:
     - **Issue `env.set_ee_target(current_target)` to maintain position target** (prevents drift).
     - **Issue `env.set_ee_quat(crane_orientation)` to maintain orientation target.**
     - **If carrying: issue `gripper.command_hold()`** (maintains tension).
     - Call `env.step(1)`.
   - Returns `HaltResult(reached, steps, velocities)` once ALL conditions satisfied:
     - `ee_linear_vel < velocity_threshold_linear`
     - `ee_angular_vel < velocity_threshold_angular`
     - **All robot arm joint velocities < `robot_joint_velocity_threshold`**
     - **All gripper finger joint velocities < `gripper_joint_velocity_threshold`**
     - If carrying: `piece_vel < held_piece_velocity_threshold`
     - If after release: `released_piece_vel < released_piece_velocity_threshold`
   - On timeout: returns `HaltResult(reached=False, ...)`.

5. **Crane orientation maintenance during motion:**
   - `MotionExecutor` must call `env.set_ee_quat(arm_config.crane_down_quat)` (via a new
     `set_mocap_quat` method in `MuJoCoEnv`) alongside every `set_ee_target` call.
   - This ensures the gripper remains in a downward-facing orientation throughout all stages.
   - `MuJoCoEnv` must expose: `set_ee_quat(quat: np.ndarray) -> None` which writes to
     `data.mocap_quat[mocap_id]`.

6. All safe hover, grasp, and release heights computed from config — no magic numbers.

7. **Write tests**:
   - Full waypoint execution for a pawn move (e2 → e4) succeeds.
   - Each stage is logged.
   - Moving piece ends up at e4 within tolerance.
   - No other piece drifts during the move (check all pieces post-move).
   - Timeout on a stage returns `MotionExecutionResult(success=False, ...)` with `failed_stage` populated.
   - Gripper orientation remains within `crane_orientation` tolerance throughout all stages.

8. **Extend debug CLI**: add `run-waypoint-stage --stage STAGE_NAME` and
   `run-move --move UCI` commands.

**Validation gate:**
- Tests pass.
- `python -m mujoco_chess.debug run-move --move e2e4` executes visually.

---

### Milestone 22 — Normal Chess Move Execution

**Goal:** The game loop executes a normal (non-capture, non-special) chess move end-to-end:
validate logically, translate to physical actions, execute atomically, update registry, commit.

**Steps:**

1. **Implement `MoveTranslator`** in `src/mujoco_chess/move_translation/translator.py`:

   ```python
   @dataclass
   class PhysicalAction:
       action_type: ActionType   # MOVE_PIECE (additional types added later)
       piece_body: str
       source_pos: np.ndarray
       destination_pos: np.ndarray
       hover_z: float
       grasp_z: float
       release_z: float

   class MoveTranslator:
       def __init__(self, registry: PhysicalPieceRegistry,
                    mapper: CoordinateMapper,
                    slot_manager: SlotManager,
                    waypoint_config: WaypointConfig) -> None: ...

       def translate(self, move: chess.Move, board: chess.Board) -> list[PhysicalAction]: ...
   ```

   **Move detection priority order** (must be applied strictly in this sequence):
   ```python
   if board.is_castling(move):
       return self._translate_castling(move, board)
   elif board.is_en_passant(move):
       return self._translate_en_passant(move, board)
   elif move.promotion is not None:
       return self._translate_promotion(move, board)   # handles promotion-capture too
   elif board.is_capture(move):
       return self._translate_capture(move, board)
   else:
       return self._translate_normal(move, board)
   ```

   This order is necessary because:
   - En passant satisfies `is_capture()` but requires special physical handling.
   - Promotion-captures satisfy both `move.promotion is not None` and `is_capture()`.
   - Checking castling/en-passant/promotion first prevents misclassification.

   For a normal move: return a single `PhysicalAction(MOVE_PIECE, body, source, dest, ...)`.

2. **Implement `MoveExecutionTransaction`** in `src/mujoco_chess/move_translation/translator.py`:

   Multi-action moves (captures, castling, promotions) must be tracked as a transaction.

   **Critical design requirement:** Health checks like `check_piece_not_drifted` compare a
   piece's physical position against its `expected_pos` in the registry. But because registry
   updates are deferred until all physical actions complete, `expected_pos` still reflects the
   old position during execution. To avoid false health-check failures, the transaction
   maintains a **pending expected position table** that health checks must use instead of the
   committed registry during the transaction:

   ```python
   @dataclass
   class MoveExecutionTransaction:
       chess_move: chess.Move
       actions: list[PhysicalAction]
       completed_action_indices: list[int]   # indices of successfully completed actions
       registry_snapshot: dict               # copy of relevant registry state before start
       # Pending positions: updated after each physical action completes.
       # Maps body_name → expected world position after that action.
       # Health checks must use this during execution instead of registry.expected_pos.
       pending_expected_positions: dict[str, np.ndarray] = field(default_factory=dict)
       pending_statuses: dict[str, PieceStatus] = field(default_factory=dict)
       committed: bool = False               # True only after commit_move() called
   ```

   After each physical action completes, update `pending_expected_positions` for the moved
   body with its new destination. Health checks must receive this mapping so that:
   - A piece mid-transit is checked against its current waypoint target, not its old square.
   - A captured piece is checked against its graveyard slot once action 1 completes.
   - A pawn is checked against pawn-storage once it is stored.

   Registry updates (the committed `PhysicalPieceRegistry` itself) are applied **only after
   all physical actions complete**. The transaction's pending state bridges the gap.

   On partial failure (action N succeeds, action N+1 fails):
   - Halt execution.
   - Record in the failure report: "Physical action N succeeded. Action N+1 failed. Logical
     board not committed. Registry not committed. Physical world is in partial state."
   - Raise `PartialMoveFailureError` with full context.
   - Never teleport pieces to clean up a partial state. Leave physical world as-is.
   - Recovery is done explicitly with `python -m mujoco_chess.debug reset-piece`.

3. **Implement `GameLoop`** skeleton in `src/mujoco_chess/app/game_loop.py`:

   ```python
   class GameLoop:
       def execute_chess_move(self, move: chess.Move) -> bool:
           # 1. Validate with chess_engine.is_move_legal(move).
           # 2. Get ChessMoveAnalysis from chess_engine.apply_move(move) (does NOT commit).
           # 3. Translate to list[PhysicalAction] via translator.translate(move, board).
           # 4. Build MoveExecutionTransaction (with empty pending_expected_positions).
           # 5. For each physical action:
           #    a. Pass transaction to executor so health checks use pending_expected_positions.
           #    b. Execute the action.
           #    c. On success: update transaction.pending_expected_positions for the moved piece.
           #    d. On failure: raise PartialMoveFailureError, do NOT update registry or commit.
           # 6. If all succeed: apply all registry updates atomically, call commit_move(move).
           # 7. Return True if committed, False if failed.
   ```

4. **Write tests** in `tests/integration/test_move_pipeline.py`:
   - `execute_chess_move(e2e4)` returns True.
   - After the move, `chess_engine.get_board_state()` reflects e2→e4.
   - Registry shows piece at e4, not e2.
   - `validate_sync` returns no errors.

**Validation gate:**
- Tests pass.
- `run-move --move e2e4` succeeds, shows piece moving, board updates.

---

### Milestone 23 — Captures and Graveyard

**Goal:** Capture moves are physically executed: captured piece goes to graveyard first, then
the attacking piece moves to the destination square. The transaction model is used.

**Steps:**

1. **Implement `_translate_capture`** in `MoveTranslator`:
   - Identify the captured piece body from registry at `move.to_square`.
   - Allocate a graveyard slot via `slot_manager.allocate_graveyard_slot(captured_color)`.
   - Return two `PhysicalAction`s:
     1. `MOVE_PIECE`: captured piece → graveyard slot.
     2. `MOVE_PIECE`: attacking piece → destination square.

2. **Execute using the transaction model** (from Milestone 22):
   - All registry updates (`capture_piece`, `move_piece`) applied only after both physical
     actions complete.
   - If action 1 succeeds but action 2 fails: report partial state, do not commit, do not
     update registry.

3. **Write tests** in `tests/integration/test_capture_pipeline.py`:
   - Execute a capture move from a test position.
   - Both physical actions succeed.
   - Captured piece is in graveyard physically and in registry.
   - Attacking piece is on destination square.
   - `validate_sync` returns no errors.
   - Partial failure test: simulate action 2 failing; confirm registry unchanged and logical
     board not committed.

4. **Write simulation test**:
   - Piece released at graveyard slot 0 (white) remains stable (drift/tilt check).

5. **Extend debug CLI**: add `run-scenario capture` command.

**Validation gate:**
- Tests pass.
- `run-scenario capture` executes visually.

---

### Milestone 24 — Castling

**Goal:** Castling (kingside and queenside, both colors) is physically executed correctly.

**Steps:**

1. **Implement `_translate_castling`** in `MoveTranslator`:
   - Detect via `board.is_castling(move)`.
   - Determine king and rook source/destination squares via `python-chess` helpers
     (`board.is_kingside_castling(move)` etc.).
   - Return two `PhysicalAction`s: **king first, then rook**.

   **Castling order rationale:** King-first is the default because:
   - The king moves to a square adjacent to the rook's destination. Moving the rook first
     may place it physically close to where the king must transit, increasing the risk of
     the gripper clipping the rook during the king's descent.
   - King-first should be validated experimentally with the `run-scenario castling` debug
     command for both kingside and queenside. If collision risk is observed, rook-first must
     be tested and the order made configurable via `configs/waypoint.yaml` →
     `castling_king_first: bool`.

2. **Write tests** in `tests/integration/test_special_moves.py`:
   - Kingside castling (white): king e1→g1, rook h1→f1.
   - Queenside castling (white): king e1→c1, rook a1→d1.
   - Same for black.
   - Registry correct after castling.
   - `validate_sync` returns no errors.

3. **Extend debug CLI**: add `run-scenario castling` command.

**Validation gate:**
- Tests pass.
- `run-scenario castling` executes visually (both kingside and queenside tested).
- No gripper–piece collision observed during the scenario.

---

### Milestone 25 — En Passant

**Goal:** En passant is physically executed correctly: the captured pawn is removed from its
actual square (not the destination square).

**Steps:**

1. **Implement `_translate_en_passant`** in `MoveTranslator`:
   - Captured pawn's actual square: same file as `move.to_square`, same rank as
     `move.from_square`:
     ```python
     captured_pawn_sq = chess.square(
         chess.square_file(move.to_square),
         chess.square_rank(move.from_square)
     )
     ```
   - Allocate graveyard slot for the captured pawn.
   - Return two `PhysicalAction`s:
     1. Move captured pawn from `captured_pawn_sq` to graveyard.
     2. Move capturing pawn from `move.from_square` to `move.to_square`.

2. **Write tests**:
   - Set up a board in en passant position (use `chess.Board` FEN).
   - Execute the en passant move.
   - Captured pawn's body was at `captured_pawn_sq`, not `move.to_square` — verify the
     correct body was used.
   - Captured pawn is in graveyard.
   - Capturing pawn is at destination.
   - `validate_sync` returns no errors.

3. **Extend debug CLI**: add `run-scenario en-passant` command.

**Validation gate:**
- Tests pass.
- `run-scenario en-passant` executes visually.

---

### Milestone 26 — Promotion

**Goal:** Pawn promotion (including promotion-capture) is physically executed using the
reserve area strategy.

**Steps:**

1. **Implement `_translate_promotion`** in `MoveTranslator`:

   Detect via `move.promotion is not None`. Build action list as follows:

   **Case A: Promotion without capture** (`not board.is_capture(move)`):
   - 2 physical actions:
     1. Move promoting pawn from `move.from_square` to pawn-storage graveyard slot.
     2. Move selected reserve piece from reserve slot to `move.to_square`.

   **Case B: Promotion with capture** (`board.is_capture(move)` is True, since en passant
   is already handled before reaching this branch):
   - 3 physical actions:
     1. Move the enemy piece at `move.to_square` to graveyard (capture it first, so the
        square is clear before the reserve piece is placed).
     2. Move promoting pawn from `move.from_square` to pawn-storage graveyard slot.
     3. Move selected reserve piece from reserve slot to `move.to_square`.

   Reserve slot identified via `slot_manager.get_reserve_slot(color, move.promotion)`.
   If no reserve slot available: raise `PromotionReserveExhaustedError` before any physical
   action starts.

2. **Registry updates** (applied only after all physical actions complete):
   - Case A: pawn → CAPTURED, reserve piece → PROMOTED/ACTIVE at `move.to_square`.
   - Case B: enemy piece → CAPTURED, pawn → CAPTURED, reserve piece → PROMOTED/ACTIVE.
   - Call `registry.promote_piece(pawn_body, reserve_body, move.to_square, new_pos)`.
   - Call `registry.capture_piece` for the enemy piece in Case B.

3. **Write tests**:
   - Promotion to queen (Case A) succeeds.
   - Promotion-capture to queen (Case B): enemy piece goes to graveyard, destination is clear
     before reserve queen is placed.
   - After promotion, reserve slot is marked used.
   - After two same-type promotions, third raises `PromotionReserveExhaustedError`.
   - Registry is correct (`validate_sync` passes) for both cases.

4. **Extend debug CLI**: add `run-scenario promotion` and `run-scenario promotion-capture`.

**Validation gate:**
- Tests pass for both promotion cases.
- `run-scenario promotion` and `run-scenario promotion-capture` execute visually.

---

### Milestone 27 — UI (Pygame Chess Board)

**Goal:** A Pygame window shows the chess board, highlights selected squares, accepts user
input, and communicates moves to the game loop.

**Steps:**

1. **Implement `BoardUI`** in `src/mujoco_chess/ui/board_ui.py`:

   Visual layout:
   - 8×8 grid of 64 px × 64 px squares (configurable).
   - Light/dark square coloring matching `BoardConfig`.
   - Piece icons: filled circles with piece-type letter (text is acceptable for v1).
   - Sidebar: current turn, game status (Playing / Check / Checkmate / Stalemate / Draw),
     last move, optional move history.
   - Buttons: "AI plays for me", optional "Reset".

   Interaction:
   - Click a piece of the current player: highlight it (selected).
   - Click a destination: if legal, post `MoveRequest`.
   - If promotion required: show a 4-button popup; wait for selection.
   - Illegal move: flash red highlight, show message.

2. **Implement `EventHandler`** in `src/mujoco_chess/ui/event_handler.py`:
   - `GameEvent` types: `MoveRequest(from_sq, to_sq, promotion)`, `AIPlayRequest`,
     `ResetRequest`, `QuitRequest`.

3. **Thread model:**
   - **Primary model:** MuJoCo viewer in main thread, Pygame in secondary thread.
   - **Fallback model:** If Pygame has platform-specific issues on a secondary thread,
     switch to running Pygame in the main thread with `mujoco.viewer.launch_passive` driven
     from the game loop's tick. Document which model is active in `docs/development_guide.md`.
   - Communication in both models: two `queue.Queue` instances (UI→game, game→UI).
   - Game loop sends immutable state snapshots (dataclass copies, not live `chess.Board`) to UI.

4. **Write tests** (unit-level):
   - `EventHandler` converts click at pixel `(x, y)` to correct chess square.
   - `MoveRequest` is produced after two valid clicks (piece → destination).
   - Promotion popup returns correct piece type for each button.

**Validation gate:**
- Manual test: UI opens, board displays correctly, clicking squares sends events.
- `run-env` command shows both the MuJoCo viewer and the Pygame window.

---

### Milestone 28 — "AI Plays For Me" and Full Game Loop

**Goal:** The complete human-vs-computer game loop works end-to-end.

**Steps:**

1. **Complete `GameLoop`** in `src/mujoco_chess/app/game_loop.py`:

   ```python
   def run(self) -> None:
       while not self.chess_engine.is_game_over():
           self._process_ui_events()
           if self._is_human_turn():
               pass   # wait for MoveRequest from UI queue
           else:
               move = self.move_selector.choose_move(self.chess_engine.get_board_state())
               self.execute_chess_move(move)
           self._send_state_to_ui()
           self.env.step(1)
       self._handle_game_over()
   ```

   Human color is determined by `game_config.human_color` (`"white"` or `"black"`).

2. **"AI plays for me":** On `AIPlayRequest`, call `move_selector.choose_move(board)` and
   execute it as a human-side move.

3. **Game-over:** Detect via `is_game_over()`, display result in UI, log result.

4. **Reset:** On `ResetRequest`: reset `ChessEngine`, `PhysicalPieceRegistry`, `SlotManager`,
   regenerate XML and reload MuJoCo, run initial settle, re-initialize registry.

5. **Write tests** in `tests/integration/test_game_reset.py`:
   - After reset, board is starting position.
   - All 32 active pieces on starting squares.
   - All 16 reserve pieces back at reserve slots (RESERVE status).
   - Graveyard is empty.

**Validation gate:**
- `python -m mujoco_chess.debug run-random-game --max-moves 20` completes without crashing.
- **Extend debug CLI**: `run-random-game --max-moves N` command added here.

---

### Milestone 29 — Full Health Check Integration

**Goal:** All health checks from section 13.1 of `plan.md` are wired into the motion executor,
startup, and game loop at the correct points. (The framework was built in Milestone 18; this
milestone completes integration.)

**Steps:**

1. Register all check functions with `HealthCheckRunner` for their `CheckContext`s
   (see the table in Milestone 18).

2. Wire `health_runner.run_checks(STARTUP)` into `startup.py` after initial settle.

3. Wire `health_runner.run_checks(START_OF_TURN)` and `END_OF_TURN` into `GameLoop`.

4. Confirm wiring in `MotionExecutor` for `SETTLE_WINDOW_BEFORE/AFTER`, `POST_GRASP`,
   `DURING_CARRY`, `POST_RELEASE` (partially done in Milestones 19–26; complete here).

5. Run all existing tests — they must still pass with full health checks active.

6. Add an integration test: deliberately introduce a piece drift, confirm health check
   catches it, failure report is written to `logs/failures/`.

**Validation gate:**
- All existing tests pass.
- Deliberate drift triggers a failure report.

---

### Milestone 30 — Full Debug Runner Commands

**Goal:** All debug commands listed in section 17 of `plan.md` are available. (Many were
added incrementally in earlier milestones; this milestone fills in any gaps and adds recovery
commands.)

**Steps:**

1. Verify all commands exist in `src/mujoco_chess/debug/cli.py`:

   | Command | Added at |
   |---|---|
   | `run-env` | Milestone 10 |
   | `inspect-piece-stability` | Milestone 11 |
   | `check-reachability [--level 1\|2]` | Milestone 15 |
   | `move-ee-to --x X --y Y --z Z` | Milestone 16 |
   | `test-grasp --square SQUARE [--repeat N]` | Milestone 19 |
   | `test-release --square SQUARE [--repeat N]` | Milestone 20 |
   | `run-waypoint-stage --stage STAGE_NAME` | Milestone 21 |
   | `run-move --move UCI` | Milestone 21 |
   | `run-scenario SCENARIO` | Milestones 23–26 |
   | `run-random-game [--max-moves N]` | Milestone 28 |

2. **Implement recovery commands** in `src/mujoco_chess/debug/recovery.py`:
   - `reset-piece --body BODY_NAME`: uses `env.set_body_freejoint_pos` to teleport piece
     to its registry `expected_pos`, then calls `mujoco.mj_forward`. Logs at WARNING level:
     `"DEBUG RECOVERY: teleporting {body_name} to {expected_pos}"`.
   - `reset-arm`: moves arm to home position via normal motion (no teleport).
   - `clear-velocities`: zeros `data.qvel`, calls `mujoco.mj_forward`, runs fresh settle.

3. Stress test scaffold in `run-random-game` and `test-grasp --repeat 100`:
   - Reports: N/total succeeded, failures per iteration, mean steps.

4. Ensure `set_body_freejoint_pos` in `MuJoCoEnv` correctly:
   - Finds the freejoint for the given body by name.
   - Writes the new position to `data.qpos[joint.qposadr : joint.qposadr + 3]`.
   - Resets the orientation (quaternion) to identity if only position is changed.
   - Calls `mujoco.mj_forward(model, data)`.
   - Is never called from anywhere except `debug/recovery.py`.

**Validation gate:**
- All commands run without errors.
- `run-random-game --max-moves 10` completes with no crashes.
- `reset-piece` teleports and logs correctly.

---

### Milestone 31 — Visual Debugging Tools

**Goal:** Optional MuJoCo visual markers for square centers, waypoint paths, and active pieces
are available in debug mode.

**Steps:**

1. **Implement `VisualMarkers`** in `src/mujoco_chess/debug/visual_markers.py`:

   If `debug_config.show_square_centers`:
   - `XMLGenerator` adds small sphere geoms (visual-only, `contype="0"`) at each square
     center in local `board_frame` coordinates.

   If `debug_config.show_waypoint_path`:
   - `MotionExecutor` updates sphere marker positions at hover, grasp, release points.

   If `debug_config.highlight_target_piece`:
   - `MotionExecutor` temporarily changes target piece geom rgba.

2. If `debug_config.pause_before_stage`:
   - `MotionExecutor` pauses before each active stage and waits for ENTER.
   - Log: `"[DEBUG] Pausing before stage {stage}. Press ENTER to continue."`.

3. If `debug_config.step_mode`: execute one simulation step per key press.

4. All visual marker geoms: `contype="0" conaffinity="0"`.

**Validation gate:**
- `run-env` with `debug.enabled: true` and `show_square_centers: true` shows markers.
- `pause_before_stage: true` causes pauses.

---

### Milestone 32 — Stress Tests

**Goal:** Automated stress tests run many repetitions and produce summary reports.

**Steps:**

1. **Implement stress test suite** in `tests/simulation/` (runnable via `pytest -m stress`):

   | Test | What it does |
   |---|---|
   | `stress_grasp_release` | Grasp and release a piece at e2 100 times |
   | `stress_move_all_squares` | Move a piece to every square and back to e2 |
   | `stress_random_game` | 5 full random games (up to 200 moves each) |
   | `stress_starting_pawns` | Execute all 16 starting pawn moves |
   | `stress_graveyard_slots` | Fill all 32 graveyard slots (simulate 32 captures) |
   | `stress_captures` | Set up and execute 10 consecutive captures |
   | `stress_promotions` | Promote all 8 pawns of one color |

2. Each test produces `logs/stress_{test_name}_{timestamp}.json`:
   ```json
   {
     "test": "stress_grasp_release",
     "total": 100,
     "passed": 99,
     "failed": 1,
     "failures": [{"iteration": 47, "details": "..."}],
     "mean_steps_per_action": 234.5
   }
   ```

**Validation gate:**
- `stress_grasp_release` achieves ≥ 95% pass rate.
- `stress_random_game` completes all 5 games without crash.
- `stress_graveyard_slots` passes (all 32 slots, no collision).

---

### Milestone 33 — Documentation

**Goal:** All required documentation files in `docs/` are complete and accurate.

**Steps:**

1. **`docs/architecture.md`**: Components, responsibilities, data flow diagram, separation of
   concerns rationale.

2. **`docs/environment.md`**: Board geometry formula, orientation convention, local vs world
   coordinate rule for XML generation, piece physics, STL setup, graveyard/reserve layouts,
   arm placement, Fetch arm asset source and actuator names, MuJoCo physics attributes.

3. **`docs/waypoint_algorithm.md`**: Crane mode, all 10 stages, hold-aware settle windows,
   complete-halt requirement (including that polling must maintain ee_target and gripper
   tension), failure handling.

4. **`docs/chess_move_translation.md`**: Detection priority order (castling → en-passant →
   promotion → capture → normal), all move types, promotion-capture sequence, transaction
   model for partial failures.

5. **`docs/health_checks.md`**: Framework architecture, full check table, contexts, failure
   behavior, report format, how to add a check. Note: contact force uses
   `mujoco.mj_contactForce`, not a direct field.

6. **`docs/debugging.md`**: All debug commands (with the milestone they were added), recovery
   commands, visual flags, failure report location, common failure modes.

7. **`docs/testing.md`**: Test categories, how to run, what each suite verifies, how to add
   regression tests.

8. **`docs/development_guide.md`**: Setup, running the project, regenerating XML, UI thread
   model and fallback, known v1 limitations (max 2 promotions per type/color, etc.), future
   improvements.

**Validation gate:**
- All 8 docs files exist, are non-empty, contain no "TODO" or "TBD" placeholders.

---

### Milestone 34 — End-to-End Validation

**Goal:** All acceptance criteria from section 20 of `plan.md` are verified.

**Steps:**

1. **Run all test suites:**
   ```bash
   pytest tests/unit/ tests/integration/ tests/simulation/ -v --tb=short
   ```
   All must pass.

2. **Run stress tests** and confirm they meet pass-rate thresholds.

3. **Manual acceptance checklist:**

   Functional:
   - [ ] User can play a chess game against the computer.
   - [ ] UI displays board and game status.
   - [ ] Legal moves accepted; illegal moves rejected.
   - [ ] "AI plays for me" works.
   - [ ] Computer plays legal moves.
   - [ ] Physical execution for all move types.
   - [ ] Captures, castling, en passant, promotion all work.
   - [ ] Promotion-captures work.
   - [ ] Captured pieces go to graveyard.
   - [ ] Promoted pieces handled correctly (reserve → active).
   - [ ] Logical and physical boards remain synchronized.

   Simulation:
   - [ ] Environment loads.
   - [ ] Board: 64 squares, each 7 × 7 cm.
   - [ ] Board is color-coded.
   - [ ] All 48 pieces (32 active + 16 reserve) spawn correctly.
   - [ ] Pieces do not sink or float.
   - [ ] Pieces stable after settle.
   - [ ] No Fetch target dot.
   - [ ] Arm in crane mode.
   - [ ] Arm reaches every square and slot (Level 1 and Level 2 reachability).
   - [ ] No low horizontal sweeping.

   Robustness:
   - [ ] Health checks run at meaningful stages.
   - [ ] Health check failures halt execution.
   - [ ] Failure reports saved to `logs/failures/`.
   - [ ] Partial physical execution failure reported clearly without committing logical move.
   - [ ] Debug mode provides visibility.
   - [ ] Tests cover core logic and all special moves.
   - [ ] Stress tests available and pass.
   - [ ] Documentation complete.

4. If any item fails: create a regression test, fix the issue, re-run.

**Validation gate:**
- All automated tests pass.
- All manual checklist items checked.

---

## Cross-Cutting Requirements

These apply throughout all milestones.

### No Magic Numbers

Every numeric value that affects geometry, physics, motion, or thresholds must come from a
config file loaded by `load_all_configs()`. If a number appears directly in source code, it
must be a mathematical constant (e.g. `0.5` for midpoint), not a domain value.

### No Hidden Teleport

`env.set_body_freejoint_pos()` must only be called from `src/mujoco_chess/debug/recovery.py`.
Any call outside that module is a bug. All calls in `recovery.py` log at WARNING level with
prefix `"DEBUG RECOVERY:"`.

### Logical Board Is Committed Last

`chess_engine.commit_move()` is called **only** after all `PhysicalAction`s in the action list
have been executed and all post-move health checks pass. If any physical action fails, the
logical board is not updated and the registry is not updated.

### Transaction Model for Multi-Action Moves

Registry updates for multi-action moves (captures, castling, en passant, promotion) are
applied together only after **all** physical actions in the `MoveExecutionTransaction` succeed.
Partial physical states (action N succeeded, action N+1 failed) are reported clearly via the
failure report and left for manual recovery — never silently fixed.

### Move Translation Priority

Inside `MoveTranslator.translate`, detection order must be:
`castling → en_passant → promotion → capture → normal`.
Violating this order causes misclassification of promotion-captures and en passant.

### XML Coordinate Rule

Geom `pos` attributes inside a `<body>` are local to that body. Use local offsets relative to
the parent body's origin for all geoms inside `board_frame`. Use world coordinates (from
`CoordinateMapper.square_to_world`) only for end-effector targets and slot positions used in
arm motion control.

### Orientation Convention

White plays from rank 1, file a at minimum X. The UI, MuJoCo scene, coordinate mapper, debug
tools, and docs must all use this same convention. Any deviation is a bug.

### Arm Does Not Know Chess

`MotionExecutor`, `GripperController`, and `WaypointPlan` have no imports from
`src/mujoco_chess/chess_logic/`.

### Chess Engine Does Not Know MuJoCo

`ChessEngine` and `MoveSelector` have no imports from `src/mujoco_chess/mujoco_env/` or
`src/mujoco_chess/motion/`.

### Gripper Stepping Owned by Executor

`GripperController` methods (`command_open`, `command_grasp`, `command_hold`) only issue
actuator commands. They do not call `env.step()` or run settle windows. Stepping is owned
exclusively by `MotionExecutor`.

### Complete-Halt Polling Maintains Commands

During `_check_complete_halt` polling, every iteration must:
- Re-issue `env.set_ee_target(current_target)` to prevent target drift.
- If carrying a piece: issue `gripper.command_hold()` to maintain tension.

### Logging at Key Points

Every module logs at start and end of meaningful actions using
`logging.getLogger(__name__)`. All log messages identify the module clearly.

---

## Dependency Graph

```
1 (skeleton)
└─ 2 (config)
   ├─ 3 (chess logic)
   │  └─ 4 (move selector)
   ├─ 5 (coord mapper)
   │  └─ 6 (slot manager)
   │     └─ 7 (XML: board only)
   │        └─ 8 (XML: pieces + reserve)
   │           └─ 9 (XML: STL)
   │              └─ 10 (env loader + CLI skeleton)
   │                 └─ 11 (settle phase)
   │                    └─ 12 (registry: active + reserve)
   │                       └─ 13 (arm XML)
   │                          └─ 14 (arm placement)
   │                             └─ 15 (reachability: L1 + L2)
   │                                └─ 16 (ee movement)
   │                                   └─ 17 (gripper commands)
   │                                      └─ 18 (health check framework)
   │                                         └─ 19 (grasp test)
   │                                            └─ 20 (release test)
   │                                               └─ 21 (full waypoint)
   │                                                  └─ 22 (normal move + transaction)
   │                                                     ├─ 23 (captures)
   │                                                     ├─ 24 (castling)
   │                                                     ├─ 25 (en passant)
   │                                                     └─ 26 (promotion + promo-capture)
   │                                                        └─ 27 (UI)
   │                                                           └─ 28 (full game loop)
   │                                                              └─ 29 (full health integration)
   │                                                                 └─ 30 (debug commands)
   │                                                                    └─ 31 (visual debug)
   │                                                                       └─ 32 (stress tests)
   │                                                                          └─ 33 (docs)
   │                                                                             └─ 34 (e2e)
```

Milestones 23–26 can be implemented in parallel once Milestone 22 is done.
Milestones 30–32 can be implemented in parallel once Milestone 29 is done.

---

## Quick Reference: Key Files

| File | Purpose |
|---|---|
| `main.py` | Entry point |
| `src/mujoco_chess/app/startup.py` | Boot: config → XML → env → settle → registry → game loop |
| `src/mujoco_chess/app/game_loop.py` | Turn management, move execution orchestration |
| `src/mujoco_chess/utils/config_loader.py` | YAML load + Pydantic validation, `load_all_configs()` |
| `src/mujoco_chess/chess_logic/engine.py` | `ChessEngine` wrapping python-chess |
| `src/mujoco_chess/board/coordinate_mapper.py` | Square ↔ world coordinate conversion |
| `src/mujoco_chess/board/slot_manager.py` | Graveyard (color-aware) and reserve slot allocation |
| `src/mujoco_chess/move_translation/translator.py` | Chess move → `list[PhysicalAction]` with priority order |
| `src/mujoco_chess/registry/piece_registry.py` | `PhysicalPieceRegistry` (active + reserve) |
| `src/mujoco_chess/mujoco_env/xml_generator.py` | Generates `generated/chess_env.xml` |
| `src/mujoco_chess/mujoco_env/env.py` | `MuJoCoEnv`: load/step/query/viewer; contact force via `mj_contactForce`; freejoint teleport via qpos |
| `src/mujoco_chess/motion/executor.py` | `MotionExecutor`: waypoint plans, settle windows, halt checks |
| `src/mujoco_chess/motion/waypoint.py` | `WaypointStage` enum, `WaypointPlan` dataclass |
| `src/mujoco_chess/motion/gripper.py` | `GripperController`: command-only, no stepping |
| `src/mujoco_chess/health/checks.py` | Individual health check functions |
| `src/mujoco_chess/health/runner.py` | `HealthCheckRunner` + failure report writer |
| `src/mujoco_chess/ui/board_ui.py` | Pygame chess board display |
| `src/mujoco_chess/debug/cli.py` | Debug command CLI (built incrementally, Milestones 10–28) |
| `src/mujoco_chess/debug/recovery.py` | Developer-only recovery tools (only caller of `set_body_freejoint_pos`) |
| `generated/chess_env.xml` | Auto-generated MuJoCo XML (never edit manually) |
| `configs/game.yaml` | Human color, move selector type |
| `assets/robot/` | Fetch arm MJCF assets (copied from Menagerie) |
