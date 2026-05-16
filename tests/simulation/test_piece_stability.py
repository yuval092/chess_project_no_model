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
    min_z = config.board.board_surface_z - config.health.piece_sink_threshold_m
    max_z = config.board.board_surface_z + config.pieces.base_height + config.health.piece_float_threshold_m
    for body_name, pos in positions.items():
        assert min_z <= pos[2] <= max_z, (body_name, pos[2], min_z, max_z)
