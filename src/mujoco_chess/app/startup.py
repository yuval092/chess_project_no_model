from __future__ import annotations

import logging

from mujoco_chess.logging_setup.setup import setup_logging
from mujoco_chess.mujoco_env.placement import assert_valid_layout
from mujoco_chess.mujoco_env.xml_generator import XMLGenerator
from mujoco_chess.utils.config_loader import load_all_configs

LOGGER = logging.getLogger(__name__)


def bootstrap(generate_xml: bool = True):
    config = load_all_configs()
    setup_logging(config.logging)
    assert_valid_layout(config)
    xml_path = XMLGenerator(config).generate() if generate_xml else config.root_dir / "generated" / "chess_env.xml"
    LOGGER.info("Bootstrap complete: %s", xml_path)
    return config, xml_path


def main() -> None:
    config, xml_path = bootstrap(generate_xml=True)
    print(f"Generated MuJoCo XML: {xml_path}")
    print("Full viewer/game loop wiring is available through debug commands as milestones progress.")
