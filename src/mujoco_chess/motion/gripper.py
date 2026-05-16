from __future__ import annotations

from mujoco_chess.utils.config_loader import ArmConfig


class GripperController:
    def __init__(self, env, config: ArmConfig) -> None:
        self.env = env
        self.config = config

    def command_open(self) -> None:
        self._set_width(self.config.gripper_open_width)

    def command_grasp(self) -> None:
        self._set_width(self.config.gripper_grasp_width)

    def command_hold(self) -> None:
        self._set_width(self.config.gripper_hold_width)

    def get_width(self) -> float:
        left = self.env.get_joint_qpos("robot0:l_gripper_finger_joint")
        right = self.env.get_joint_qpos("robot0:r_gripper_finger_joint")
        return self.config.gripper_grasp_width + left + right

    def is_open(self) -> bool:
        return abs(self.get_width() - self.config.gripper_open_width) < 0.005

    def is_grasping(self) -> bool:
        return abs(self.get_width() - self.config.gripper_grasp_width) < 0.005

    def _set_width(self, width: float) -> None:
        finger_target = max(0.0, (width - self.config.gripper_grasp_width) / 2)
        self.env.set_ctrl(self.config.gripper_left_actuator, finger_target)
        self.env.set_ctrl(self.config.gripper_right_actuator, finger_target)
