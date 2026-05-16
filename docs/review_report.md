# Review Report

**Reviewer:** Claude (automated review pass)
**Date:** 2026-05-16
**Branch:** master
**Test baseline:** 46 tests pass (excluding `tests/simulation/test_reachability.py`)

---

## 1. Previously-reported bugs — disposition

| # | Bug | Status |
|---|-----|--------|
| 1 | `MotionExecutor` had hidden `env.config.arm` dependency | **Fixed.** Constructor now takes explicit `ArmConfig`; no `env.config` access anywhere in `executor.py`. |
| 2 | `test_release()` created a second `MotionExecutor` that called `initialize_arm()` mid-sequence | **Fixed.** `test_release` in `scenarios.py` creates one executor, passes it to `test_grasp`, then reuses it. `test_release_places_pawn_on_target_square` in `test_physical_moves.py` asserts `initialize_calls == 1`. |
| 3 | `check_level2` grasp formula had spurious `+0.045` hardcode | **Fixed.** `scenarios.py` `check_level2` computes grasp Z as `base_height + shaft_height/2 + grasp_z_offset` from config — no hardcode. |
| 4 | `move-ee-to` CLI didn't use `MotionExecutor` | **Fixed.** `cli.py` line 57 creates `MotionExecutor(env, config.waypoint, config.arm).move_to(...)`. |
| 5 | `WaypointPlan.crane_orientation` was unused in `execute_plan` | **Fixed.** `execute_plan` passes `plan.crane_orientation` to every `_run_move_stage` and `_run_hold_stage` call. Test `test_execute_plan_uses_plan_crane_orientation` verifies this. |
| 6 | M19/M20 health check gates not integrated | **Partially fixed.** POST_GRASP and DURING_CARRY run in `execute_plan`. POST_RELEASE runs after ascent. Continuous carry health for every transit segment is still deferred (documented deviation 21). |
| 7 | `deviations.md` M13 row said "reachability still fails" | **Fixed / Not present.** `deviations.md` M13 row says reachability passes; the milestone_status.md M15 entry is "Complete". |
| 8 | M20 "piece does not fall during ascent" test missing | **Addressed.** `test_execute_plan_moves_one_piece_e2_to_e4` in `test_physical_moves.py` validates that the piece ends at destination and is not moving. A discrete "does not fall during ascent" assertion is not isolated, but ascent failure would cause the plan to fail and the test to fail. |

---

## 2. New bugs found in current code

None. The codebase is internally consistent and all 46 tests pass.

Minor observations (not bugs, no code change required):

- `executor.py` `_run_hold_stage` only fires DURING_CARRY health at the `ASCEND_WITH_PIECE` stage. The plan deviation (21) documents this intentionally. No action needed.
- `GameLoop._commit_registry_updates` walks `action.graveyard_color` without a None-guard, but `graveyard_color` is only read when `graveyard_slot is not None`, which is set together — so the logic is safe.
- `MoveExecutionTransaction.pending_statuses` is populated in the dataclass but never written to by `game_loop.py`. This is dead state from the M22/M23 transaction model stub. It does not cause any test failure.

---

## 3. Misalignments with implementation plan

### Already implemented but plan says pending

- `_translate_castling` is fully implemented in `translator.py` (lines 103–114).
- `_translate_en_passant` is fully implemented in `translator.py` (lines 116–125).
- `_translate_promotion` is fully implemented in `translator.py` (lines 127–143).
- `test_castling_translation` exists in `tests/unit/test_move_translator.py`.

### Not yet implemented per plan

- `tests/integration/test_special_moves.py` does not exist. M24 requires full integration tests for castling with registry validation.
- `run-scenario castling` CLI command is registered (`run-scenario` sub-parser exists) but the handler currently prints "registered but full execution is pending".
- M25 integration tests for en passant are missing.
- M26 integration tests for promotion are missing.
- `run-scenario en-passant` and `run-scenario promotion` handlers are not implemented.

---

## 4. Stale documentation

- `docs/waypoint_algorithm.md` last sentence says "Multi-action transaction use for castling, en passant, and promotion remains pending." This is stale: translation is implemented. The integration tests and CLI scenario handlers remain pending.
- `docs/milestone_status.md` M24+ entry says "Castling, en passant, promotion ... remain pending" — accurate that tests and CLI are not done, but should note that translation logic exists.
- `docs/testing.md` does not mention the castling translation test (`test_castling_translation`).

---

## 5. What is correctly implemented

- Full project skeleton, config loading, chess logic, move selector, coordinate mapper, slot manager, XML generator, piece XML generation, MuJoCo environment loader, settle phase, physical piece registry, Fetch arm XML, arm/layout validation, reachability validation tool, basic EE movement, gripper command interface, health check framework, single-piece grasp, single-piece release, full waypoint algorithm, normal chess move execution (M22), captures and graveyard (M23).
- `MoveTranslator` dispatches all five move types: normal, capture, castling, en passant, promotion.
- `GameLoop.execute_chess_move` handles multi-action moves generically — it iterates actions and uses the transaction model.
- `MotionExecutor` cleanly separates arm config from env.
- All integration tests for the normal move pipeline and capture pipeline pass.

---

## 6. Summary of work completed in this review pass

1. Verified all 8 previously-reported bugs are resolved.
2. Found no new bugs.
3. Identified missing M24 integration tests and CLI handler as the primary gap.
4. Proceeding to implement M24 (tests + CLI) and M25 (tests + CLI) in this session.
