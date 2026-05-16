# Testing

Unit tests cover configuration loading, chess logic, selectors, coordinate mapping, slot allocation, registry behavior, move translation (including castling translation), health checks, game-loop transaction behavior, and motion-executor dependency injection. Integration tests cover normal move pipeline, capture pipeline, castling (kingside/queenside, both colors, registry sync), en passant (correct captured-pawn square, graveyard allocation, attacker destination, registry sync), and promotion (normal push-promotion, capture-promotion, pawn-to-storage teleport, reserve-piece-to-board teleport, registry sync). Simulation tests cover XML generation, MuJoCo load/step smoke behavior, placement validation, piece position stability, real Fetch gripper commands, 64-board-square reachability, basic end-effector movement, e2 grasp/lift, e2-to-e4 release, one-piece `execute_plan()`, normal e2e4 game-loop execution, e4xd5 capture-to-graveyard execution, and off-board teleportation.

Run tests with `.venv/bin/pytest tests/ -q` after installing the project into the local virtual environment. Exclude the slow reachability test with `--ignore=tests/simulation/test_reachability.py`.

**Stress tests** live in `tests/simulation/test_stress.py` and are marked with `@pytest.mark.stress`. They are excluded from normal runs automatically (not collected unless `-m stress` is passed). Run them explicitly with:

    pytest tests/simulation/test_stress.py -m stress

Current stress tests include:
- `test_stress_grasp_release_scaffold`: verifies the stress framework works with a single move execution using `RecordingExecutor` (no MuJoCo required).
- `test_stress_random_game_scaffold`: runs one AI-vs-AI game of up to 20 moves using `HeuristicMoveSelector` and `RecordingExecutor`, verifying registry sync at the end.
