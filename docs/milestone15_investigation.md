# Milestone 15 — Reachability Investigation

## Status
**RESOLVED FOR BOARD TARGETS** — all 64 board squares pass Level 1 reachability. Off-board graveyard, pawn-storage, and reserve slots now use teleportation by project decision and are intentionally excluded from arm reachability.

---

## Root Causes (5 distinct bugs)

### Bug 1 — Wrong `home_position` (arm.yaml)

`home_position: [0.0, 0.2, 0.95]`

The arm at default joint positions (all zeros) has `robot0:gripper_link` at world
`[0, -0.278, 1.036]`. The mocap body `ee_target` starts at `home_position = [0, 0.2, 0.95]`.
These are **0.49 m apart**. The weld constraint immediately generates a massive force trying
to pull the arm 0.49 m from its natural rest, which the high joint damping cannot satisfy.
The arm is permanently stuck near joints=0 regardless of what target is set.

**Fix:** `home_position` must be set to where the arm's gripper_link actually rests at the
keyframe joint configuration (see Bug 5).

---

### Bug 2 — Missing Gymnasium-style weld initialization

In `MuJoCoEnv.load()`, the simulation starts with:
- `model.eq_data[i, :7]` = whatever the XML specified (not necessarily identity relpose)
- `data.mocap_pos[0]` = `home_position` from the XML body
- `data.mocap_quat[0]` = `crane_down_quat` from the XML body quat

Gymnasium does three additional steps before using the arm:
1. `reset_mocap_welds`: sets `model.eq_data[i, :7] = [0,0,0,0,0,0,1]` (identity relpose)
2. `reset_mocap2body_xpos`: sets `mocap_pos = gripper_link.xpos` and
   `mocap_quat = gripper_link.xquat`
3. Runs `mj_forward` once

Without step 1+2, the weld starts with nonzero position AND orientation error, generating
competing forces that prevent the arm from ever converging to any target.

**Verified:** After adding steps 1+2, the arm converges to board center `[0, 0, 0.86]`
with error 0.007 m (passes 0.01 m tolerance) within 2000 steps.

**Fix:** Add `MuJoCoEnv.initialize_arm()` that performs the Gymnasium init sequence plus
sets the arm joints to the keyframe (see Bug 5).

---

### Bug 3 — Wrong `crane_down_quat` (arm.yaml)

`crane_down_quat: [1.0, 0.0, 0.0, 0.0]`

`[1, 0, 0, 0]` is the identity quaternion. The weld forces `robot0:gripper_link` to have
**identity world orientation**, meaning the gripper's local +X axis points in world +X.
But the arm's kinematics (with base yaw = -π/2) have the gripper pointing in world -Y at
rest. The orientation constraint fights the kinematics, making the arm unable to lower its
Z to board height.

For **crane mode** (gripper pointing straight down for picking pieces from above), the
gripper's local +X axis must point in world -Z. The correct quaternion for this is:

```
crane_down_quat: [0.7071, 0.0, 0.7071, 0.0]
```

Verified: R([0.7071,0,0.7071,0]) maps local +X → world (0,0,-1) = world -Z = downward. ✓

**Fix:** Update arm.yaml `crane_down_quat: [0.7071, 0.0, 0.7071, 0.0]`.

---

### Bug 4 — `_check_point` steps only 180 times from cold start

`ReachabilityChecker._check_point` calls `env.step(min(stage_timeout_steps, 180))`.

Even with proper initialization, the arm needs ~2000 steps to converge from a resting
position to a board target position. 180 steps is insufficient. The method also does not
call `initialize_arm()` before the sweep, so it starts from an incorrect state.

**Fix:**
- Call `env.initialize_arm()` once before running all checks.
- Increase step count to 1000 per check (checks are sequential on the same env, so the arm
  moves incrementally from target to target — 1000 steps is sufficient for incremental moves).
- For the first check after `initialize_arm()`, increase to 2000 steps.

---

### Bug 5 — No arm joint keyframe; arm starts at joints = 0

All joints default to 0. This puts the gripper_link at `[0, -0.278, 1.036]` — far from the
board workspace (hover height z=0.86 m, board center y=0). The arm must travel
0.18 m in Z and up to 0.28 m in both X/Y to reach any board position.

Without a pre-set keyframe, the arm starts outside the working envelope and the weld
constraint (even when correctly initialized) cannot reliably pull the arm into position for
cold-start checks.

**Required initial joint configuration** (places arm at board center hover in crane mode):
```yaml
initial_qpos:
  robot0:torso_lift_joint: 0.20
  robot0:shoulder_pan_joint: 0.0
  robot0:shoulder_lift_joint: -0.40
  robot0:upperarm_roll_joint: 0.0
  robot0:elbow_flex_joint: 1.80
  robot0:forearm_roll_joint: 0.0
  robot0:wrist_flex_joint: -1.40
  robot0:wrist_roll_joint: 0.0
```
(Approximate values; must be verified by running `initialize_arm()` and checking EE pos.)

**Fix:** Add `initial_qpos` to `ArmConfig`, set in arm.yaml, and apply in `initialize_arm()`.

---

### Bug 6 — Board layout: pawn storage and reserve outside arm workspace

The arm's reachable workspace at hover height z=0.86 m spans approximately:
- **X:** −0.40 m to +0.40 m  (board uses ±0.245 m → OK)
- **Y:** −0.30 m to +0.10 m  (board uses −0.245 to +0.245 m → upper ranks FAIL)
- **Z:** 0.86 m hover is reachable once arm is in proper configuration

Layout problems:
- **Pawn storage** at `x = board_origin_x ± 6 * square_size = ±(0.28+0.42) = ±0.70 m`
  → Exceeds ±0.40 m lateral reach. **Must move closer** (e.g., ±3–4 squares from board edge).
- **Reserve area** at `y = board_max_y + 0.16 = 0.44 m`
  → Exceeds +0.10 m Y reach. **Must move closer** (e.g., behind the board in negative Y,
  or reduce offset).
- **Board near ranks (7–8)** at y ≈ +0.175 to +0.245 m → at the edge of Y workspace;
  reachable only after proper arm initialization and joint pre-positioning.

**Resolution:** off-board slots are no longer arm targets. Graveyard, pawn storage, and
reserve movement is handled through `env.teleport_piece()`, while arm reachability is
validated only for the public 64-square board workspace.

---

## Fix Plan (ordered)

1. Update `arm.yaml`: `crane_down_quat`, `home_position`, add `initial_qpos`
2. Update `ArmConfig` in `config_loader.py`: add `initial_qpos: dict[str, float] = {}`
3. Add `MuJoCoEnv.initialize_arm()` in `env.py`
4. Update `ReachabilityChecker._check_point` in `scenarios.py` to call `initialize_arm()`
   and use 1000 steps
5. Keep off-board slots in layout, but exclude them from arm reachability because they teleport.
6. Run reachability check to verify; tune `initial_qpos` if needed.
7. Update `test_reachability.py` to assert 100% pass rate for the 64 board squares.

---

## Key Empirical Data

| Target | steps | error (m) | status |
|--------|-------|-----------|--------|
| Board center [0,0,0.86] (cold start, proper init) | 2000 | 0.007 | PASS |
| Board center (sequential, proper init) | 1000 | ~0.005 | PASS |
| a1 hover cold start, improper init | 5000 | 0.165 | FAIL |
| h8 hover from board center | 1000 | 0.010 | ~PASS |
| Reserve [0, 0.44, 0.86] | 5000 | 0.505 | FAIL (outside workspace) |
| Pawn storage [±0.70, −0.28, 0.86] | 5000 | 0.45 | FAIL (outside workspace) |
