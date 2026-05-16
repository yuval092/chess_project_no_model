from __future__ import annotations

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.motion.gripper import GripperController
from mujoco_chess.mujoco_env.env import MuJoCoEnv


def test_gripper_open_and_grasp_commands_change_width() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    gripper = GripperController(env, config.arm)
    initial = gripper.get_width()
    gripper.command_open()
    env.step(config.waypoint.gripper_settle_steps)
    opened = gripper.get_width()
    assert opened > initial
    assert gripper.is_open()
    gripper.command_grasp()
    env.step(config.waypoint.gripper_settle_steps)
    assert gripper.get_width() < opened
    assert gripper.is_grasping()
