# Debugging

The debug CLI currently includes `run-env`, `inspect-piece-stability`, `check-reachability`, `move-ee-to`, `test-grasp`, `test-release`, `run-waypoint-stage`, `run-move`, `run-scenario`, and `run-random-game`. `check-reachability`, `move-ee-to`, `test-grasp`, `test-release`, and normal `run-move` commands execute against the real Fetch simulation. Commands for scenarios, random games, and special moves remain pending later milestones.

`move-ee-to` uses `MotionExecutor.move_to()` and reports success/failure, step count, final position, and final linear velocity. `run-move --move UCI` currently executes normal board-to-board moves such as `e2e4` through the game loop, move translator, physical executor, and registry commit path. `run-scenario castling` executes kingside castling (white), kingside castling (black), and queenside castling (white) sequentially against the Fetch simulation. `run-scenario en-passant` executes the en passant capture e5xd6 from a minimal starting FEN. `run-scenario promotion` executes a queen push-promotion (e7e8q) from a minimal starting FEN; all promotion actions are teleports (pawn to storage, reserve queen to board).

Recovery helpers live in `src/mujoco_chess/debug/recovery.py`. The only normal caller of debug teleportation is `reset_piece`, which logs with the `DEBUG RECOVERY:` prefix before using `set_body_freejoint_pos`.

Failure reports are written to `logs/failures/`.

`check-reachability` writes `logs/reachability_report.txt` and checks the 64 board squares only. Graveyard, pawn-storage, and reserve slots are intentionally excluded because those off-board actions use teleportation.
