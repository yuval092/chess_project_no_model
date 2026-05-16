from __future__ import annotations

import pytest

from mujoco_chess.utils.config_loader import AppConfig, ArmConfig, ConfigValidationError, load_all_configs


def test_valid_yaml_loads() -> None:
    config = load_all_configs()
    assert isinstance(config, AppConfig)
    assert config.board.square_size == 0.07


def test_gripper_open_width_cross_validation() -> None:
    config = load_all_configs()
    bad_arm = config.arm.model_copy(update={"gripper_open_width": 0.08})
    bad = AppConfig(**{**config.__dict__, "arm": bad_arm})
    from mujoco_chess.utils.config_loader import _validate_cross_config

    with pytest.raises(ConfigValidationError):
        _validate_cross_config(bad)


def test_missing_required_waypoint_value_raises(tmp_path) -> None:
    # Loading from an empty temp project should fail on missing config files.
    with pytest.raises(ConfigValidationError):
        load_all_configs(tmp_path)
