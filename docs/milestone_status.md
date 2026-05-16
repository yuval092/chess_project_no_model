# Milestone Status

Current implementation state is tracked here as milestones are implemented and validated.

| Milestone | State | Validation |
|---|---|---|
| 1 - Project Skeleton | Complete | Editable install succeeds in `.venv`; imports pass |
| 2 - Config Loading and Validation | Complete | `tests/unit/test_config.py` passes |
| 3 - Chess Logic Wrapper | Complete | `tests/unit/test_chess_engine.py` passes |
| 4 - Move Selector Interface | Complete | Selector-specific tests pass |
| 5 - Board Coordinate Mapper | Complete | `tests/unit/test_coordinate_mapper.py` passes |
| 6 - Slot Manager | Complete | `tests/unit/test_slot_manager.py` passes |
| 7 - MuJoCo XML Generator (Board Only) | Complete | XML generation tests pass |
| 8 - Pieces and Reserve Pieces in XML | Complete | XML generation tests verify 48 freejoint bodies |
| 9 - STL Visual Support | Partial | Optional mesh support implemented; placeholder STL files not generated because `stl_scale` is disabled |
| 10 - MuJoCo Environment Loader | Partial | Load/step/contact smoke test passes; viewer not launched in headless validation |
| 11 - Initial Settle Phase | Partial | Settle helper and position stability test pass; strict velocity stability pending physics tuning |
| 12 - Physical Piece Registry | Complete | Dedicated registry tests pass |
| 13 - Fetch Arm XML | Complete | Gymnasium Robotics Fetch XML/STL/texture assets are copied and generated XML loads |
| 14 - Arm/Layout Validation | Complete | Geometry layout validation and real Fetch startup arm-contact tests pass |
| 15 - Reachability Validation Tool | Complete | All 64 board squares pass arm reachability; off-board slots (graveyard, pawn storage, reserve) use teleportation and are excluded from arm reach checks |
| 16 - Basic End-Effector Movement | Complete | `MotionExecutor.move_to()` reaches board targets within position and velocity tolerance; unreachable target test returns failure |
| 17 - Gripper Command Interface | Partial | Real Fetch gripper open/grasp integration test passes |
| 18 - Minimal Health Check Framework | Partial | Core result/runner/failure report and several checks implemented, including grasp contact, slipping, and release checks; full check table pending |
| 19 - Single-Piece Grasp Test | Partial | e2 pawn grasp/lift passes focused simulation test and debug CLI; POST_GRASP and DURING_CARRY health gates run, but broader settle-window checks remain pending |
| 20 - Single-Piece Release Test | Partial | e2-to-e4 grasp/carry/release passes focused simulation test and debug CLI; POST_RELEASE health gate runs, but broader settle-window checks remain pending |
| 21 - Full Waypoint Algorithm (One Piece) | Partial | `MotionExecutor.execute_plan()` performs home, source hover, grasp, lift, hold-aware carry, release, ascent, and return home for e2-to-e4; complete health-check table and halt polling remain pending |
| 22 - Normal Chess Move Execution | Complete | `GameLoop.execute_chess_move(e2e4)` validates, translates, executes physically, commits board/registry transactionally; `run-move --move e2e4` passes |
| 23 - Captures and Graveyard | Complete | e4xd5 executes transactionally after e2e4/d7d5; captured piece teleports to graveyard, attacker moves physically, board/registry sync passes |
| 24 - Castling | Complete | `tests/integration/test_special_moves.py` castling tests pass (kingside/queenside, both colors, registry sync); `run-scenario castling` CLI handler implemented |
| 25 - En Passant | Complete | `tests/integration/test_special_moves.py` en passant tests pass (correct source square d5 not d6, graveyard slot, attacker destination, registry sync); `run-scenario en-passant` CLI handler implemented |
| 26 - Promotion | Complete | `tests/integration/test_special_moves.py` promotion tests pass (normal push-promotion, capture-promotion, pawn→storage teleport, reserve piece→board teleport, registry sync); `run-scenario promotion` CLI handler implemented |
| 27 - BoardUI (Pygame Chess Board) | Complete | Full Pygame board with 8×8 grid, piece circles with letters, sidebar, AI/Reset buttons, promotion popup, state/event queues; `tests/unit/test_event_handler.py` passes (6 tests) |
| 28 - Full Game Loop | Complete | `GameLoop.run()` and `reset()` implemented; `run-random-game` CLI wired; `tests/integration/test_game_reset.py` passes (4 tests) |
| 29 - Full Health Check Integration | Complete | `build_health_runner()` in `startup.py` registers STARTUP checks; `health_runner` parameter in `GameLoop` calls START_OF_TURN and END_OF_TURN around `execute_chess_move()` |
| 30 - Full Debug Runner Commands | Complete | `reset-piece --body`, `reset-arm`, and `clear-velocities` sub-commands added to debug CLI; `run-random-game` implemented |
| 31 - Visual Debugging (Minimal) | Complete | `xml_generator.py` adds debug sphere geoms when `debug.show_square_centers=True`; `VisualMarkers.highlight_square()` implemented |
| 32 - Stress Test Scaffold | Complete | `tests/simulation/test_stress.py` with `@pytest.mark.stress`; excluded from normal runs, 2 scaffold tests pass |
| 33 - Documentation | Complete | `milestone_status.md`, `testing.md`, `debugging.md`, `development_guide.md`, and `deviations.md` updated for M27–M32 |
| 34 - End-to-End Validation | Complete | All 79 automated tests pass; `main.py` launches the full game (MuJoCo viewer + Pygame UI); debug CLI has full `--viewer` support, `inspect-piece-stability`, `run-waypoint-stage`, and comprehensive help text; `debugging.md` updated with complete command reference |
