# Waypoint Algorithm

The intended full waypoint sequence is: move home, move to source hover, descend to grasp, close gripper, ascend with piece, move to destination hover, descend to release, open gripper, ascend after release, and return home.

The current implementation includes the `WaypointStage` enum, `WaypointPlan` dataclass, board-target end-effector interpolation, real mocap target updates, real Fetch gripper commands, and focused grasp/release debug scenarios. `MotionExecutor.move_to()` now validates both position and end-effector velocity before succeeding. `MotionExecutor` receives arm configuration explicitly, and `WaypointPlan.crane_orientation` is used when executing a plan.

Off-board graveyard, pawn-storage, and reserve actions are intentionally handled by teleportation, so the full physical waypoint algorithm only needs to cover board-to-board movement.

The focused grasp/release scenarios run POST_GRASP, DURING_CARRY, and POST_RELEASE health gates. `MotionExecutor.execute_plan()` now executes the full one-piece e2-to-e4 waypoint path with gripper open/close commands, hold-aware carry, release-center compensation, optional health-runner gates, ascent after release, and return home.

Pending work: add complete-halt polling to all settle windows, make continuous carry health robust across long horizontal transits, register the full health-check table for every stage, and validate complete chess moves transactionally.

Normal e2e4 chess moves, e4xd5 captures, castling (both sides, both colors), en passant, and promotion (push and capture-promotion) are now validated transactionally through `GameLoop`, `MoveTranslator`, and `MotionExecutor`. Captures and en passant use the project-approved off-board teleportation path for the captured piece and arm motion for the attacking board-to-board move. Promotion is entirely off-board: the pawn teleports to pawn storage, the captured piece (if any) teleports to graveyard, and the reserve piece teleports to the destination square.
