from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def reset_piece(env, registry, body_name: str) -> None:
    record = registry.get_piece_by_body(body_name)
    if record is None:
        raise KeyError(body_name)
    LOGGER.warning("DEBUG RECOVERY: teleporting %s to %s", body_name, record.expected_pos)
    env.set_body_freejoint_pos(body_name, record.expected_pos)


def reset_arm(env, home_position) -> None:
    env.set_ee_target(home_position)


def clear_velocities(env) -> None:
    env.data.qvel[:] = 0
