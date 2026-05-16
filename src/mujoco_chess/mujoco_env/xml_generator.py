from __future__ import annotations

import logging
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import chess

from mujoco_chess.board.coordinate_mapper import CoordinateMapper
from mujoco_chess.board.slot_manager import SlotManager
from mujoco_chess.utils.config_loader import AppConfig

LOGGER = logging.getLogger(__name__)


class XMLGenerator:
    def __init__(self, config: AppConfig, output_path: str | Path | None = None) -> None:
        self.config = config
        self.output_path = Path(output_path) if output_path else config.root_dir / "generated" / "chess_env.xml"
        self.mapper = CoordinateMapper(config.board)
        self.slots = SlotManager(config.board, config.pieces)

    def generate(self) -> Path:
        LOGGER.info("Generating MuJoCo XML at %s", self.output_path)
        root = ET.Element("mujoco", {"model": "chess"})
        ET.SubElement(
            root,
            "compiler",
            {
                "angle": "radian",
                "coordinate": "local",
                "meshdir": str(self.config.root_dir / "assets" / "robot" / "stls" / "fetch"),
                "texturedir": str(self.config.root_dir / "assets" / "robot" / "textures"),
            },
        )
        ET.SubElement(root, "option", {"gravity": "0 0 -9.81", "timestep": "0.002"})
        asset = ET.SubElement(root, "asset")
        self._add_materials(asset)
        self._add_mesh_assets(asset)
        self._merge_fetch_shared(root, asset)
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", {"name": "key_light", "pos": "0 0 3", "dir": "0 0 -1"})
        ET.SubElement(world, "geom", {
            "name": "floor",
            "type": "plane",
            "size": "5 5 0.1",
            "pos": "0 0 0",
            "rgba": "0.45 0.45 0.45 1",
            "condim": "3",
        })
        self._add_table(world)
        self._add_board(world)
        self._add_storage_platforms(world)
        self._add_starting_pieces(world)
        self._add_reserve_pieces(world)
        self._add_fetch_arm(world)
        self._add_fetch_actuators(root)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(self.output_path, encoding="utf-8", xml_declaration=True)
        return self.output_path

    def _add_materials(self, asset: ET.Element) -> None:
        for name, rgba in {
            "light_square": self.config.board.light_square_rgba,
            "dark_square": self.config.board.dark_square_rgba,
            "board": self.config.board.board_rgba,
            "white_piece": self.config.pieces.white_rgba,
            "black_piece": self.config.pieces.black_rgba,
        }.items():
            ET.SubElement(asset, "material", {"name": name, "rgba": _fmt(rgba)})

    def _add_mesh_assets(self, asset: ET.Element) -> None:
        if self.config.pieces.stl_scale <= 0:
            return
        for piece_name in ("pawn", "rook", "knight", "bishop", "queen", "king"):
            path = self.config.root_dir / "assets" / "stl" / f"{piece_name}.stl"
            if path.exists():
                ET.SubElement(asset, "mesh", {"name": piece_name, "file": str(path), "scale": _fmt([self.config.pieces.stl_scale] * 3)})
            else:
                LOGGER.warning("Skipping missing STL asset: %s", path)

    def _add_table(self, world: ET.Element) -> None:
        b = self.config.board
        # Table surface: a thick slab whose TOP face sits at table_height.
        # The slab is 5 cm thick; its center is at table_height - 0.025.
        slab_half_z = 0.025
        slab_center_z = b.table_height - slab_half_z
        # Half-width large enough to hold the board (with frame) plus 8 cm clearance.
        board_half_with_frame = b.files * b.square_size / 2 + b.square_size / 2  # squares + 1 border
        slab_half_xy = board_half_with_frame + 0.08
        ET.SubElement(world, "geom", {
            "name": "table_surface",
            "type": "box",
            "size": _fmt([slab_half_xy, slab_half_xy, slab_half_z]),
            "pos": _fmt([0.0, 0.0, slab_center_z]),
            "rgba": "0.45 0.36 0.28 1",
        })
        # Four legs: square cross-section, from floor (z=0) to slab bottom.
        leg_half = 0.04
        leg_half_z = (b.table_height - slab_half_z * 2) / 2
        leg_center_z = leg_half_z
        leg_offset = slab_half_xy - leg_half
        for lx, ly in [(leg_offset, leg_offset), (leg_offset, -leg_offset),
                       (-leg_offset, leg_offset), (-leg_offset, -leg_offset)]:
            ET.SubElement(world, "geom", {
                "name": f"table_leg_{int(lx*100)}_{int(ly*100)}",
                "type": "box",
                "size": _fmt([leg_half, leg_half, leg_half_z]),
                "pos": _fmt([lx, ly, leg_center_z]),
                "rgba": "0.38 0.30 0.22 1",
            })

    def _add_board(self, world: ET.Element) -> None:
        b = self.config.board
        body = ET.SubElement(world, "body", {"name": "board_frame", "pos": _fmt([b.board_origin_x, b.board_origin_y, b.board_origin_z])})
        # Squares span local x/y: [0, files*sq] = [0, 0.56].
        # Equal 3.5 cm (half-square) border on all four sides:
        #   board spans [-border, files*sq+border] = [-0.035, 0.595]
        #   center = 0.28, half-extent = 0.315
        border = b.square_size / 2
        board_half = (b.files * b.square_size + 2.0 * border) / 2.0
        board_center = b.files * b.square_size / 2.0
        # board_origin_z is the BOTTOM of the board; geom center is half-thickness above that.
        # board top = board_origin_z + board_thickness = board_surface_z ✓
        board_geom_z = b.board_thickness / 2
        ET.SubElement(body, "geom", {
            "name": "board_base",
            "type": "box",
            "size": _fmt([board_half, board_half, b.board_thickness / 2]),
            "pos": _fmt([board_center, board_center, board_geom_z]),
            "contype": "0",
            "conaffinity": "0",
            "material": "board",
        })
        ET.SubElement(body, "geom", {
            "name": "board_surface",
            "type": "box",
            "size": _fmt([board_half, board_half, b.board_thickness / 2]),
            "pos": _fmt([board_center, board_center, board_geom_z]),
            "contype": "1",
            "conaffinity": "1",
            "material": "board",
        })
        # Square visuals sit flush on top of the board surface.
        visual_z = b.board_thickness + 0.0005
        for file_idx in range(8):
            for rank_idx in range(8):
                name = f"square_{chess.FILE_NAMES[file_idx]}{rank_idx + 1}"
                material = "light_square" if (file_idx + rank_idx) % 2 == 0 else "dark_square"
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "size": _fmt([b.square_size / 2, b.square_size / 2, 0.0005]),
                        "pos": _fmt([file_idx * b.square_size + b.square_size / 2, rank_idx * b.square_size + b.square_size / 2, visual_z]),
                        "contype": "0",
                        "conaffinity": "0",
                        "material": material,
                    },
                )
        # Debug square-center markers (only when debug.show_square_centers is True)
        if self.config.debug.show_square_centers:
            debug_z = b.board_thickness + 0.001
            for file_idx in range(8):
                for rank_idx in range(8):
                    file_name = chess.FILE_NAMES[file_idx]
                    rank_name = str(rank_idx + 1)
                    ET.SubElement(
                        body,
                        "geom",
                        {
                            "name": f"debug_center_{file_name}{rank_name}",
                            "type": "sphere",
                            "size": "0.005",
                            "pos": _fmt([file_idx * b.square_size + b.square_size / 2, rank_idx * b.square_size + b.square_size / 2, debug_z]),
                            "contype": "0",
                            "conaffinity": "0",
                            "rgba": "1 0 0 1",
                        },
                    )

    def _add_storage_platforms(self, world: ET.Element) -> None:
        for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            gy_positions = [self.slots.graveyard_slot_world_pos(color, idx) for idx in range(16)]
            ps_positions = [self.slots.pawn_storage_slot_world_pos(color, idx) for idx in range(8)]
            self._platform(world, f"{label}_graveyard_platform", gy_positions)
            self._platform(world, f"{label}_pawn_storage_platform", ps_positions)
        reserve_positions = [
            self.slots.reserve_slot_world_pos(color, piece_type, slot)
            for color in (chess.WHITE, chess.BLACK)
            for piece_type in SlotManager.RESERVE_TYPES
            for slot in range(2)
        ]
        self._platform(world, "promotion_reserve_platform", reserve_positions)

    def _platform(self, world: ET.Element, name: str, positions: list) -> None:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        z = float(positions[0][2]) - 0.002
        size_x = max(0.04, (max(xs) - min(xs)) / 2 + 0.04)
        size_y = max(0.04, (max(ys) - min(ys)) / 2 + 0.04)
        ET.SubElement(world, "geom", {"name": name, "type": "box", "size": _fmt([size_x, size_y, 0.002]), "pos": _fmt([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, z]), "rgba": "0.22 0.22 0.22 1"})

    def _add_starting_pieces(self, world: ET.Element) -> None:
        counters: dict[tuple[chess.Color, chess.PieceType], int] = {}
        board = chess.Board()
        for square, piece in board.piece_map().items():
            key = (piece.color, piece.piece_type)
            piece_id = counters.get(key, 0)
            counters[key] = piece_id + 1
            color_char = "w" if piece.color == chess.WHITE else "b"
            type_char = piece.symbol().upper()
            self._piece_body(world, f"piece_{color_char}_{type_char}_{piece_id}", piece.color, piece.piece_type, self.mapper.square_to_world(square))

    def _add_reserve_pieces(self, world: ET.Element) -> None:
        for color in (chess.WHITE, chess.BLACK):
            color_char = "w" if color == chess.WHITE else "b"
            for piece_type in SlotManager.RESERVE_TYPES:
                type_char = chess.piece_symbol(piece_type).upper()
                for slot in range(2):
                    pos = self.slots.reserve_slot_world_pos(color, piece_type, slot)
                    self._piece_body(world, f"piece_reserve_{color_char}_{type_char}_{slot}", color, piece_type, pos)

    def _piece_body(self, world: ET.Element, body_name: str, color: chess.Color, piece_type: chess.PieceType, surface_pos) -> None:
        p = self.config.pieces
        body_z = float(surface_pos[2]) + p.base_height / 2
        body = ET.SubElement(world, "body", {"name": body_name, "pos": _fmt([surface_pos[0], surface_pos[1], body_z])})
        ET.SubElement(body, "freejoint")
        material = "white_piece" if color == chess.WHITE else "black_piece"
        ET.SubElement(body, "geom", {"name": f"base_{body_name}", "type": "cylinder", "size": _fmt([p.base_radius, p.base_height / 2]), "pos": "0 0 0", "mass": str(p.mass), "friction": _fmt(p.friction), "solimp": _fmt(p.solimp), "solref": _fmt(p.solref), "material": material})
        ET.SubElement(body, "geom", {"name": f"shaft_{body_name}", "type": "capsule", "size": _fmt([p.shaft_radius, p.shaft_height / 2]), "pos": _fmt([0, 0, p.base_height / 2 + p.shaft_height / 2]), "contype": "1", "conaffinity": "1", "material": material})
        mesh_name = chess.piece_name(piece_type)
        if p.stl_scale > 0 and (self.config.root_dir / "assets" / "stl" / f"{mesh_name}.stl").exists():
            ET.SubElement(body, "geom", {"name": f"visual_{body_name}", "type": "mesh", "mesh": mesh_name, "contype": "0", "conaffinity": "0", "material": material})

    def _merge_fetch_shared(self, root: ET.Element, project_asset: ET.Element) -> None:
        shared_path = self.config.root_dir / "assets" / "robot" / "fetch" / "shared.xml"
        if not shared_path.exists():
            raise FileNotFoundError(f"Missing Fetch shared XML: {shared_path}")
        shared = ET.parse(shared_path).getroot()
        shared_asset = shared.find("asset")
        if shared_asset is not None:
            for child in list(shared_asset):
                project_asset.append(copy.deepcopy(child))
        for tag in ("default", "contact"):
            child = shared.find(tag)
            if child is not None:
                root.append(copy.deepcopy(child))
        equality = shared.find("equality")
        if equality is not None:
            new_equality = copy.deepcopy(equality)
            for weld in new_equality.findall("weld"):
                if weld.attrib.get("body1") == "robot0:mocap":
                    weld.attrib["body1"] = self.config.arm.mocap_body_name
                if weld.attrib.get("body2") == "robot0:gripper_link":
                    weld.attrib["body2"] = self.config.arm.ee_weld_body_name
            root.append(new_equality)

    def _add_fetch_arm(self, world: ET.Element) -> None:
        robot_path = self.config.root_dir / "assets" / "robot" / "fetch" / "robot.xml"
        if not robot_path.exists():
            raise FileNotFoundError(f"Missing Fetch robot XML: {robot_path}")
        robot = ET.parse(robot_path).getroot()
        for body in robot.findall("body"):
            body_copy = copy.deepcopy(body)
            if body_copy.attrib.get("name") == "robot0:mocap":
                body_copy.attrib["name"] = self.config.arm.mocap_body_name
                body_copy.attrib["pos"] = _fmt(self.config.arm.home_position)
                body_copy.attrib["quat"] = _fmt(self.config.arm.crane_down_quat)
                for geom in list(body_copy.findall("geom")):
                    body_copy.remove(geom)
                ET.SubElement(body_copy, "geom", {"name": "ee_mocap_marker", "type": "sphere", "size": "0.003", "rgba": "0 0 0 0", "contype": "0", "conaffinity": "0"})
            elif body_copy.attrib.get("name") == self.config.arm.base_body_name:
                body_copy.attrib["pos"] = _fmt([self.config.arm.base_x, self.config.arm.base_y, self.config.arm.base_z])
                body_copy.attrib["euler"] = _fmt([0, 0, self.config.arm.base_yaw])
            self._remove_fetch_goal_artifacts(body_copy)
            world.append(body_copy)

    def _add_fetch_actuators(self, root: ET.Element) -> None:
        actuator = ET.SubElement(root, "actuator")
        ET.SubElement(actuator, "position", {"name": self.config.arm.gripper_left_actuator, "joint": "robot0:l_gripper_finger_joint", "kp": "30000", "ctrlrange": "0 0.2", "ctrllimited": "true", "user": "1"})
        ET.SubElement(actuator, "position", {"name": self.config.arm.gripper_right_actuator, "joint": "robot0:r_gripper_finger_joint", "kp": "30000", "ctrlrange": "0 0.2", "ctrllimited": "true", "user": "1"})

    def _remove_fetch_goal_artifacts(self, element: ET.Element) -> None:
        for child in list(element):
            name = child.attrib.get("name", "").lower()
            if name in {"target", "goal", "target0", "goal0"}:
                element.remove(child)
            else:
                self._remove_fetch_goal_artifacts(child)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)
