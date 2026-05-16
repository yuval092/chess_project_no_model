from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_chess.mujoco_env.xml_generator import XMLGenerator
from mujoco_chess.utils.config_loader import load_all_configs


def test_xml_generation_board_and_pieces() -> None:
    config = load_all_configs()
    path = XMLGenerator(config).generate()
    root = ET.parse(path).getroot()
    squares = [geom for geom in root.findall(".//geom") if geom.attrib.get("name", "").startswith("square_")]
    assert len(squares) == 64
    assert all(geom.attrib.get("contype") == "0" for geom in squares)
    assert len(root.findall(".//geom[@name='board_surface']")) == 1
    assert root.find(".//body[@name='board_frame']").attrib["pos"] == "-0.28 -0.28 0.5"
    piece_bodies = [body for body in root.findall(".//body[freejoint]") if body.attrib["name"].startswith("piece")]
    assert len(piece_bodies) == 48
    names = [body.attrib["name"] for body in root.findall(".//body")]
    assert len(names) == len(set(names))


def test_xml_generation_fetch_arm() -> None:
    config = load_all_configs()
    path = XMLGenerator(config).generate()
    root = ET.parse(path).getroot()
    assert root.find(f".//body[@name='{config.arm.base_body_name}']") is not None
    assert root.find(f".//body[@name='{config.arm.ee_weld_body_name}']") is not None
    assert root.find(f".//site[@name='{config.arm.ee_site_name}']") is not None
    assert root.find(f".//body[@name='{config.arm.mocap_body_name}']") is not None
    weld = root.find(".//equality/weld")
    assert weld is not None
    assert weld.attrib["body2"] == config.arm.ee_weld_body_name
    assert weld.attrib["body1"] == config.arm.mocap_body_name
