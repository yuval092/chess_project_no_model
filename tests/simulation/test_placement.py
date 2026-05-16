from __future__ import annotations

from mujoco_chess.mujoco_env.placement import PlacementValidationError, assert_valid_layout, validate_layout
from mujoco_chess.utils.config_loader import AppConfig, load_all_configs


def test_default_layout_does_not_intersect() -> None:
    config = load_all_configs()
    result = validate_layout(config)
    assert result.passed, result.errors
    assert_valid_layout(config)


def test_arm_board_intersection_fails() -> None:
    config = load_all_configs()
    # Place the arm base directly on the board center — must be rejected.
    board_center_x = (config.board.board_origin_x + config.board.board_max_x) / 2
    board_center_y = (config.board.board_origin_y + config.board.board_max_y) / 2
    bad_arm = config.arm.model_copy(update={"base_x": board_center_x, "base_y": board_center_y})
    bad = AppConfig(**{**config.__dict__, "arm": bad_arm})
    result = validate_layout(bad)
    assert not result.passed
    assert any("arm_base intersects board" == error for error in result.errors)
    try:
        assert_valid_layout(bad)
    except PlacementValidationError:
        pass
    else:
        raise AssertionError("expected placement validation failure")
