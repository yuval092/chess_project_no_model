from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from mujoco_chess.utils.config_loader import AppConfig

LOGGER = logging.getLogger(__name__)


class MuJoCoUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContactInfo:
    geom1_name: str
    geom2_name: str
    body1_name: str
    body2_name: str
    force_magnitude: float


class MuJoCoEnv:
    def __init__(self, xml_path: str | Path, config: AppConfig) -> None:
        self.xml_path = str(xml_path)
        self.config = config
        self.model = None
        self.data = None
        self.viewer = None

    def load(self) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise MuJoCoUnavailableError("MuJoCo is not installed in this environment") from exc
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        self._require_loaded()
        import mujoco
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and i % 10 == 0:
                self.viewer.sync()

    def sync_viewer(self) -> None:
        """Push current simulation state to the passive viewer (no-op if no viewer)."""
        if self.viewer is not None:
            self.viewer.sync()

    def get_body_pos(self, body_name: str) -> np.ndarray:
        self._require_loaded()
        return self.data.body(body_name).xpos.copy()

    def get_body_quat(self, body_name: str) -> np.ndarray:
        self._require_loaded()
        return self.data.body(body_name).xquat.copy()

    def get_body_vel(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        self._require_loaded()
        cvel = self.data.body(body_name).cvel.copy()
        return cvel[3:].copy(), cvel[:3].copy()

    def get_contacts(self) -> list[ContactInfo]:
        self._require_loaded()
        import mujoco
        contacts: list[ContactInfo] = []
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, idx, force)
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ""
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ""
            contacts.append(ContactInfo(geom1_name, geom2_name, body1_name, body2_name, float(np.linalg.norm(force[:3]))))
        return contacts

    def teleport_piece(self, body_name: str, world_pos: np.ndarray) -> None:
        """Instantly move a freejoint body to world_pos and zero its velocity."""
        self._require_loaded()
        import mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for joint_id in range(self.model.njnt):
            if self.model.jnt_bodyid[joint_id] == body_id and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                qposadr = self.model.jnt_qposadr[joint_id]
                dofadr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qposadr : qposadr + 3] = world_pos
                self.data.qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[dofadr : dofadr + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)
                return
        raise ValueError(f"No freejoint found for body {body_name}")

    def set_body_freejoint_pos(self, body_name: str, pos: np.ndarray) -> None:
        self._require_loaded()
        import mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for joint_id in range(self.model.njnt):
            if self.model.jnt_bodyid[joint_id] == body_id and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[adr : adr + 3] = pos
                self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
                mujoco.mj_forward(self.model, self.data)
                return
        raise ValueError(f"No freejoint found for body {body_name}")

    def set_mocap_pos(self, mocap_name: str, pos: np.ndarray) -> None:
        self._require_loaded()
        idx = self._mocap_index(mocap_name)
        self.data.mocap_pos[idx] = pos

    def set_mocap_quat(self, mocap_name: str, quat: np.ndarray) -> None:
        self._require_loaded()
        idx = self._mocap_index(mocap_name)
        self.data.mocap_quat[idx] = quat

    def set_ctrl(self, actuator_name: str, value: float) -> None:
        self._require_loaded()
        actuator = self.model.actuator(actuator_name)
        self.data.ctrl[actuator.id] = value

    def get_joint_qpos(self, joint_name: str) -> float:
        self._require_loaded()
        joint = self.model.joint(joint_name)
        return float(self.data.qpos[joint.qposadr[0]])

    def get_ee_pos(self) -> np.ndarray:
        self._require_loaded()
        return self.data.site(self.config.arm.ee_site_name).xpos.copy()

    def set_ee_target(self, pos: np.ndarray) -> None:
        self.set_mocap_pos(self.config.arm.mocap_body_name, pos)

    def set_ee_quat(self, quat: np.ndarray) -> None:
        self.set_mocap_quat(self.config.arm.mocap_body_name, quat)

    def initialize_arm(self) -> None:
        self._require_loaded()
        import mujoco
        for joint_name, value in self.config.arm.initial_qpos.items():
            j = self.model.joint(joint_name)
            self.data.qpos[j.qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.model.eq_data[i, :7] = np.array([0., 0., 0., 0., 0., 0., 1.])
        gl_pos = self.data.body(self.config.arm.ee_weld_body_name).xpos.copy()
        gl_quat = self.data.body(self.config.arm.ee_weld_body_name).xquat.copy()
        self.set_mocap_pos(self.config.arm.mocap_body_name, gl_pos)
        self.set_mocap_quat(self.config.arm.mocap_body_name, gl_quat)
        mujoco.mj_forward(self.model, self.data)
        self.set_ee_target(np.array(self.config.arm.home_position))
        self.set_ee_quat(np.array(self.config.arm.crane_down_quat))
        self.step(200)

    def open_viewer(self) -> None:
        self._require_loaded()
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _mocap_index(self, body_name: str) -> int:
        import mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Unknown mocap body: {body_name}")
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            raise ValueError(f"Body is not mocap-controlled: {body_name}")
        return int(mocap_id)

    def _require_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCoEnv.load() must be called first")


class SettleFailureError(Exception):
    pass


class SettlePhase:
    def __init__(self, env: MuJoCoEnv, config: AppConfig) -> None:
        self.env = env
        self.config = config

    def run(self, piece_body_names: list[str]) -> dict[str, np.ndarray]:
        self.env.step(self.config.waypoint.initial_settle_steps)
        return {body: self.env.get_body_pos(body) for body in piece_body_names}
