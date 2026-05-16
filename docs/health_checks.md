# Health Checks

Health checks return `HealthCheckResult` values and are scheduled through `HealthCheckRunner` contexts such as startup, settle windows, post-grasp, during-carry, post-release, start-of-turn, and end-of-turn.

Position checks must use transaction pending expected positions when available. Failure handling writes JSON reports under `logs/failures/` and raises `HealthCheckFailureError`.

MuJoCo contact force extraction uses `mujoco.mj_contactForce`; contact objects do not expose a direct force field.
