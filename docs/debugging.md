# Debugging

## Overview

The debug CLI (`mujoco-chess-debug`) is the primary tool for inspecting and validating the
simulation at every layer — from raw physics to full game moves. Every command that touches
MuJoCo loads the full environment (XML generation → MuJoCo load → arm initialisation →
settle phase) before running the requested operation.

**Entry points:**

```bash
# Full game (MuJoCo viewer + Pygame board UI, configured via configs/game.yaml)
python main.py

# Debug CLI
mujoco-chess-debug <command> [options]
python -m mujoco_chess.debug <command> [options]
```

**Visual mode:** Pass `--viewer` to any command to open the MuJoCo passive 3D viewer.
The simulation steps automatically update the viewer. Press **Esc** or close the window
to stop. `run-env --viewer` is the canonical "see the board" command.

---

## Command Reference

### `run-env`

Load the environment, settle all pieces, run a stability check, and keep the viewer open.

```bash
mujoco-chess-debug run-env [--viewer] [--no-viewer]
```

- `--viewer` (default) — open the MuJoCo 3D viewer and loop until closed
- `--no-viewer` — just print the stability report and exit

**Use this to:** visually verify the board layout, piece positions, arm home position,
and that no pieces have sunk, floated, or tilted after the settle phase.

**Output:** per-piece PASS/FAIL table and `logs/piece_stability_report.txt`.

---

### `inspect-piece-stability`

After the settle phase, print a per-piece stability table without keeping the viewer running.

```bash
mujoco-chess-debug inspect-piece-stability [--viewer]
```

Checks each piece body (`piece_*`) for:
- Z position within `[board_z - sink_threshold, board_z + piece_height + float_threshold]`
- Tilt (degrees from vertical) below `piece_tilt_threshold_deg`
- Linear velocity below `unexpected_velocity_threshold`

**Output:** table of PASS/FAIL per piece and `logs/piece_stability_report.txt`.

---

### `check-reachability`

Verify the Fetch arm can reach all 64 board squares.

```bash
mujoco-chess-debug check-reachability [--level {1,2}] [--viewer]
```

- `--level 1` (default) — hover height above every square in a snake scan
- `--level 2` — additionally check grasp height for corner and centre squares

Off-board slots (graveyard, pawn-storage, reserve) are intentionally excluded because
those moves use teleportation, not arm movement.

**Output:** pass count, first failures, `logs/reachability_report.txt`.

---

### `move-ee-to`

Command the arm to a specific world coordinate and report the outcome.

```bash
mujoco-chess-debug move-ee-to --x X --y Y --z Z [--viewer]
```

Uses `MotionExecutor.move_to()`. Reports success/failure, step count, final position,
and final linear velocity.

**Example:**
```bash
mujoco-chess-debug move-ee-to --x 0.1 --y 0.5 --z 0.6 --viewer
```

---

### `test-grasp`

Attempt to grasp the piece on a given square and verify contact.

```bash
mujoco-chess-debug test-grasp --square SQUARE [--repeat N] [--viewer]
```

Full sequence: hover → descend to grasp height → close gripper → POST_GRASP health check
→ lift to hover → DURING_CARRY health check. Reports success/failure with contact count
and lift delta.

**Example:**
```bash
mujoco-chess-debug test-grasp --square e2 --repeat 3 --viewer
```

---

### `test-release`

Full grasp-carry-release cycle for a piece.

```bash
mujoco-chess-debug test-release --square DEST [--source SRC] [--repeat N] [--viewer]
```

- `--square` — destination square (where to place the piece)
- `--source` — source square to pick from (default: `e2`)

Reports drift, tilt, and linear velocity of the placed piece.

**Example:**
```bash
mujoco-chess-debug test-release --square e4 --source e2 --viewer
```

---

### `run-waypoint-stage`

Execute a single named waypoint stage (arm motion primitive).

```bash
mujoco-chess-debug run-waypoint-stage --stage STAGE [--square SQ] [--dest DSQ] [--viewer]
```

Valid stages:

| Stage | Description | Needs `--square` | Needs `--dest` |
|---|---|---|---|
| `MOVE_TO_HOME` | Move arm to home position | No | No |
| `MOVE_TO_SOURCE_HOVER` | Move to hover above `--square` | Yes | No |
| `DESCEND_TO_GRASP` | Descend to grasp height above `--square` | Yes | No |
| `CLOSE_GRIPPER` | Close gripper (grasp command) | No | No |
| `ASCEND_WITH_PIECE` | Ascend to hover above `--square` | Yes | No |
| `MOVE_TO_DESTINATION_HOVER` | Move to hover above `--dest` | Yes | Yes |
| `DESCEND_TO_RELEASE` | Descend to release height above `--square` | Yes | No |
| `OPEN_GRIPPER` | Open gripper (release command) | No | No |
| `ASCEND_AFTER_RELEASE` | Ascend to hover after release | Yes | No |
| `RETURN_HOME` | Move arm to home position | No | No |

**Examples:**
```bash
mujoco-chess-debug run-waypoint-stage --stage MOVE_TO_HOME --viewer
mujoco-chess-debug run-waypoint-stage --stage MOVE_TO_SOURCE_HOVER --square e2 --viewer
mujoco-chess-debug run-waypoint-stage --stage MOVE_TO_DESTINATION_HOVER --square e2 --dest e4 --viewer
```

---

### `run-move`

Execute a normal board-to-board move through the full game pipeline.

```bash
mujoco-chess-debug run-move --move UCI [--viewer]
```

Runs: move validation → translation → physical execution (arm motion) → registry commit
→ board commit. Starting position is the standard chess starting position.

**Examples:**
```bash
mujoco-chess-debug run-move --move e2e4 --viewer
mujoco-chess-debug run-move --move g1f3 --viewer
```

---

### `run-scenario`

Run a named multi-move scenario from a fixed starting FEN.

```bash
mujoco-chess-debug run-scenario {castling|en-passant|promotion} [--viewer]
```

| Scenario | FEN | Moves |
|---|---|---|
| `castling` | `r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1` | e1g1, e8g8, e1c1 |
| `en-passant` | `4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1` | e5d6 |
| `promotion` | `3k4/4P3/8/8/8/8/8/4K3 w - - 0 1` | e7e8q |

**Example:**
```bash
mujoco-chess-debug run-scenario castling --viewer
```

---

### `run-random-game`

Run a full AI-vs-AI game using `HeuristicMoveSelector`.

```bash
mujoco-chess-debug run-random-game [--max-moves N] [--viewer]
```

Both sides are controlled by the AI (`human_color=None`). Prints the result
(`1-0`, `0-1`, `1/2-1/2`) or `None` if `max_moves` was reached (default: 20).

**Example:**
```bash
mujoco-chess-debug run-random-game --max-moves 40 --viewer
```

---

### `reset-piece`

Teleport a piece body back to its last recorded expected position.

```bash
mujoco-chess-debug reset-piece --body BODY_NAME
```

Uses `recovery.py:reset_piece()` (debug-only teleportation). Logs with `DEBUG RECOVERY:`
prefix. Steps 100 frames after teleport.

**Example:**
```bash
mujoco-chess-debug reset-piece --body piece_w_P_0
```

---

### `reset-arm`

Move the end-effector to the home position.

```bash
mujoco-chess-debug reset-arm
```

Uses `recovery.py:reset_arm()`. Steps 100 frames.

---

### `clear-velocities`

Zero all joint velocities in the simulation.

```bash
mujoco-chess-debug clear-velocities
```

Uses `recovery.py:clear_velocities()`. Use after a stuck or high-velocity state.
Steps 100 frames after clearing.

---

## Failure Reports

Failure reports are written to `logs/failures/`. Each report is a text file named with
a timestamp and move UCI (e.g. `2024-01-15T10-30-00_e2e4.txt`). Reports include the
failed health check name, context, and simulation state snapshot.

---

## Quick Start — Visual Debugging

```bash
# 1. See the board and arm in 3D
mujoco-chess-debug run-env --viewer

# 2. Check arm can reach all squares
mujoco-chess-debug check-reachability --viewer

# 3. Run a single move and watch it
mujoco-chess-debug run-move --move e2e4 --viewer

# 4. Watch a full AI game (20 moves)
mujoco-chess-debug run-random-game --max-moves 20 --viewer

# 5. Test grasp at e2 three times
mujoco-chess-debug test-grasp --square e2 --repeat 3 --viewer

# 6. Play the full game interactively (Pygame board + MuJoCo viewer)
python main.py
```

---

## Configuration

Debug behaviour is controlled by `configs/debug.yaml`:

- `enabled` — master debug flag
- `show_square_centers` — add red sphere markers at the centre of each board square
- `show_waypoint_path` — (reserved for future use)
- `highlight_target_piece` — (reserved for future use)
- `pause_before_stage` — (reserved for future use)
- `step_mode` — (reserved for future use)

To enable square centre markers: set `show_square_centers: true` in `configs/debug.yaml`
and re-run any command (XML is regenerated automatically).
