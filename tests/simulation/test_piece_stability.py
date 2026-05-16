from __future__ import annotations

import mujoco

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.mujoco_env.env import MuJoCoEnv, SettlePhase


def test_all_pieces_settle_without_sinking_or_floating() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    piece_names = [
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, idx)
        for idx in range(env.model.nbody)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, idx) or "").startswith("piece")
    ]
    positions = SettlePhase(env, config).run(piece_names)
    assert len(positions) == 48
    board_min_z = config.board.board_surface_z - config.health.piece_sink_threshold_m
    board_max_z = config.board.board_surface_z + config.pieces.base_height + config.health.piece_float_threshold_m
    floor_max_z = config.board.table_height - 0.05  # reserve/storage pieces sit on floor, not table
    for body_name, pos in positions.items():
        if "reserve" in body_name:
            # Reserve pieces are stored on the floor before any promotion occurs.
            assert pos[2] < floor_max_z, (body_name, pos[2], "expected on floor")
        else:
            assert board_min_z <= pos[2] <= board_max_z, (body_name, pos[2], board_min_z, board_max_z)
