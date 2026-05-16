from __future__ import annotations

import numpy as np

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.mujoco_env.env import MuJoCoEnv


def test_env_loads_and_steps() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    assert np.allclose(env.get_body_pos("board_frame"), [-0.28, -0.28, 0.5])
    env.step(10)
    assert isinstance(env.get_contacts(), list)
    assert np.allclose(env.get_body_pos(config.arm.base_body_name), [config.arm.base_x, config.arm.base_y, config.arm.base_z], atol=1e-6)
    assert env.get_ee_pos()[2] > config.board.table_height
    env.set_ee_target(np.array(config.arm.home_position))
    env.set_ee_quat(np.array(config.arm.crane_down_quat))
    names = set()
    import mujoco

    for obj_type in (mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE):
        count = env.model.ngeom if obj_type == mujoco.mjtObj.mjOBJ_GEOM else env.model.nsite
        for idx in range(count):
            name = mujoco.mj_id2name(env.model, obj_type, idx)
            if name:
                names.add(name)
    assert "target" not in names
    assert "goal" not in names


def test_real_fetch_has_no_startup_arm_board_or_piece_contacts() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    env.step(config.waypoint.initial_settle_steps)
    for contact in env.get_contacts():
        names = {contact.body1_name, contact.body2_name, contact.geom1_name, contact.geom2_name}
        has_arm = any(name.startswith("robot0:") for name in names)
        has_board_or_piece = any(name.startswith("board") or name.startswith("piece") for name in names)
        assert not (has_arm and has_board_or_piece), contact
