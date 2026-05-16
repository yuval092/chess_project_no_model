from __future__ import annotations

import numpy as np

from mujoco_chess.motion.executor import MotionExecutor
from mujoco_chess.motion.waypoint import WaypointPlan
from mujoco_chess.utils.config_loader import load_all_configs


class EnvWithoutConfig:
    def __init__(self, body_name: str) -> None:
        self.body_name = body_name
        self.pos = np.zeros(3)
        self.body_positions = {"piece_w_P_0": np.zeros(3)}
        self.vel = (np.zeros(3), np.zeros(3))
        self.target = np.zeros(3)
        self.quat = None

    def get_body_pos(self, body_name: str) -> np.ndarray:
        if body_name in self.body_positions:
            return self.body_positions[body_name].copy()
        assert body_name == self.body_name
        return self.pos.copy()

    def get_body_vel(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        assert body_name == self.body_name
        return self.vel

    def set_ee_target(self, target: np.ndarray) -> None:
        self.target = target.copy()

    def set_ee_quat(self, quat: np.ndarray) -> None:
        self.quat = quat.copy()

    def set_ctrl(self, actuator_name: str, value: float) -> None:
        del actuator_name, value

    def step(self, n: int = 1) -> None:
        del n
        self.pos = self.target.copy()


def test_motion_executor_uses_injected_arm_config_not_env_config() -> None:
    config = load_all_configs()
    env = EnvWithoutConfig(config.arm.ee_weld_body_name)

    result = MotionExecutor(env, config.waypoint, config.arm).move_to(np.array([0.1, 0.0, 0.2]))

    assert result.success
    assert env.quat is not None
    np.testing.assert_allclose(env.quat, np.array(config.arm.crane_down_quat))


def test_execute_plan_uses_plan_crane_orientation() -> None:
    config = load_all_configs()
    env = EnvWithoutConfig(config.arm.ee_weld_body_name)
    orientation = np.array([0.5, 0.5, 0.5, 0.5])
    plan = WaypointPlan(
        piece_body="piece_w_P_0",
        source_pos=np.array([0.0, 0.0, 0.0]),
        destination_pos=np.array([0.1, 0.0, 0.0]),
        hover_z=0.2,
        grasp_z=0.1,
        release_z=0.1,
        crane_orientation=orientation,
    )

    result = MotionExecutor(env, config.waypoint, config.arm).execute_plan(plan)

    assert result.success
    np.testing.assert_allclose(env.quat, orientation)
