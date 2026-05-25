from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from prosthetic_grasp.common.types import (
    Phase5HandPrediction,
    Phase5ManoResult,
    Phase6ProstheticActionResult,
)


_MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
_FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

_ROBOT_PROFILES: dict[str, dict[str, Any]] = {
    "shadow_hand": {
        "urdf_path": "hand/shadow_hand/shadowhand.urdf",
        "wrist_link": "robot0:palm",
        "fingertip_links": [
            "robot0:thdistal",
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
        ],
        "tip_point": "mesh_tip",
    },
    "folding_hand": {
        "urdf_path": "hand/folding_hand/urdf/folding.urdf",
        "xml_path": "hand/folding_hand/folding.xml",
        "model_format": "xml",
        "wrist_link": "base_link",
        "fingertip_links": [
            "th_distal",
            "ff_distal",
            "mf_distal",
            "rf_distal",
            "lf_distal",
        ],
        "tip_point": "mesh_tip",
        "joint_limits": {
            "th_base": (0.0, 1.5708),
            "th_pip": (0.0, 1.22173),
            "th_mip": (0.0, float(np.pi)),
            "th_dip": (0.0, float(np.pi)),
            "ff_base": (0.0, 1.5708),
            "ff_pip": (0.0, float(np.pi)),
            "ff_dip": (0.0, float(np.pi)),
            "mf_pip": (0.0, float(np.pi)),
            "mf_dip": (0.0, float(np.pi)),
            "rf_base": (0.0, 1.5708),
            "rf_pip": (0.0, float(np.pi)),
            "rf_dip": (0.0, float(np.pi)),
            "lf_base": (0.0, 1.5708),
            "lf_pip": (0.0, float(np.pi)),
            "lf_dip": (0.0, float(np.pi)),
        },
    },
    "inspire_hand": {
        "urdf_path": "hand/inspire_hand_ftp/urdf/inspire_right.urdf",
        "wrist_link": "right_base_link",
        "fingertip_links": [
            "R_thumb_tip",
            "R_index_tip",
            "R_middle_tip",
            "R_ring_tip",
            "R_little_tip",
        ],
        "tip_point": "link_origin",
    },
}


@dataclass
class Phase6ProstheticActionConfig:
    """Retarget MANO fingertips to a configurable fixed-wrist robot hand.

    MANO input uses 21-keypoint fingertip indices [4, 8, 12, 16, 20].
    Robot output uses five configured fingertip links in thumb/index/middle/
    ring/little order. The retargeter keeps the robot wrist/base fixed and
    optimizes only actuated joints from the robot URDF.

    To add another hand under ``hand/``, set ``robot_profile = "custom"`` and
    provide ``robot_urdf_path`` plus ``fingertip_links``. Optionally provide
    ``joint_names`` to restrict or reorder optimized joints.
    """

    robot_profile: str = "shadow_hand"
    robot_urdf_path: str = ""
    robot_xml_path: str = ""
    model_format: str = ""
    wrist_link: str = ""
    fingertip_links: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    tip_point: str = ""
    hand_preference: str = "right"
    optimization_restarts: int = 8
    max_nfev: int = 250
    regularization_weight: float = 0.002
    preserve_scale: bool = False
    min_scale: float = 0.25
    max_scale: float = 4.0
    random_seed: int = 7
    action_names: list[str] = field(default_factory=list)

    # Backward-compatible aliases from the previous Shadow-only config.
    shadow_urdf_path: str = ""
    shadow_point: str = ""

    # Backward-compatible, unused fields from the older folding MJCF config.
    mjcf_path: str = "hand/folding_hand/folding.xml"
    mesh_dir: str = "hand/folding_hand/meshes"
    grid_samples: int = 21
    refine_iterations: int = 2

    def __post_init__(self) -> None:
        self.robot_profile = self.robot_profile.strip()
        self.robot_urdf_path = self.robot_urdf_path.strip()
        self.robot_xml_path = self.robot_xml_path.strip()
        self.model_format = self.model_format.strip().lower()
        self.wrist_link = self.wrist_link.strip()
        self.hand_preference = self.hand_preference.strip().lower()
        self.tip_point = self.tip_point.strip()
        self.shadow_urdf_path = self.shadow_urdf_path.strip()
        self.shadow_point = self.shadow_point.strip()
        self.mjcf_path = self.mjcf_path.strip()
        self.mesh_dir = self.mesh_dir.strip()

        if self.robot_urdf_path == "" and self.shadow_urdf_path:
            self.robot_urdf_path = self.shadow_urdf_path
        if self.robot_xml_path == "" and self.model_format in {"xml", "mjcf"} and self.mjcf_path:
            self.robot_xml_path = self.mjcf_path
        if self.tip_point == "" and self.shadow_point:
            self.tip_point = "mesh_tip" if self.shadow_point == "distal_mesh_tip" else "link_origin"

        if self.hand_preference not in {"right", "left", "any"}:
            raise ValueError(
                "hand_preference must be 'right', 'left', or 'any', "
                f"got {self.hand_preference!r}."
            )
        if self.tip_point and self.tip_point not in {"mesh_tip", "link_origin"}:
            raise ValueError(
                "tip_point must be 'mesh_tip' or 'link_origin', "
                f"got {self.tip_point!r}."
            )
        if self.model_format and self.model_format not in {"urdf", "xml", "mjcf"}:
            raise ValueError(
                "model_format must be 'urdf', 'xml', or 'mjcf', "
                f"got {self.model_format!r}."
            )
        if self.optimization_restarts <= 0:
            raise ValueError(
                f"optimization_restarts must be positive, got {self.optimization_restarts}."
            )
        if self.max_nfev <= 0:
            raise ValueError(f"max_nfev must be positive, got {self.max_nfev}.")
        if self.regularization_weight < 0:
            raise ValueError(
                "regularization_weight must be non-negative, "
                f"got {self.regularization_weight}."
            )
        if self.min_scale <= 0 or self.max_scale <= 0 or self.min_scale > self.max_scale:
            raise ValueError(
                "min_scale and max_scale must be positive and ordered, "
                f"got ({self.min_scale}, {self.max_scale})."
            )

    def resolved_robot(
        self,
    ) -> tuple[Path, str, str, list[str], list[str] | None, str, str, dict[str, tuple[float, float]]]:
        profile = self.robot_profile
        if profile not in _ROBOT_PROFILES and profile != "custom":
            raise ValueError(
                f"Unknown robot_profile {profile!r}. "
                f"Known profiles: {sorted(_ROBOT_PROFILES)} or 'custom'."
            )

        preset = _ROBOT_PROFILES.get(profile, {})
        model_format = self.model_format or str(preset.get("model_format", "urdf"))
        if model_format == "mjcf":
            model_format = "xml"
        model_path = (
            self.robot_xml_path
            if model_format == "xml"
            else self.robot_urdf_path
        )
        model_path = model_path or str(preset.get("xml_path" if model_format == "xml" else "urdf_path", ""))
        wrist_link = self.wrist_link or str(preset.get("wrist_link", ""))
        fingertip_links = self.fingertip_links or list(preset.get("fingertip_links", []))
        tip_point = self.tip_point or str(preset.get("tip_point", "mesh_tip"))
        joint_names = self.joint_names or None
        joint_limits = dict(preset.get("joint_limits", {}))
        if not model_path:
            raise ValueError("robot_urdf_path or robot_xml_path is required for custom Phase6 retargeting.")
        if not wrist_link:
            raise ValueError("wrist_link is required for custom Phase6 retargeting.")
        if len(fingertip_links) != 5:
            raise ValueError(
                "fingertip_links must contain exactly five links in thumb/index/"
                f"middle/ring/little order, got {fingertip_links!r}."
            )
        return Path(model_path), model_format, wrist_link, fingertip_links, joint_names, tip_point, profile, joint_limits


class _RobotHandKinematics:
    def __init__(
        self,
        urdf_path: str | Path,
        wrist_link: str,
        fingertip_links: list[str],
        joint_names: list[str] | None,
        tip_point: str,
        joint_limit_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        from yourdfpy import URDF

        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Robot hand URDF does not exist: {self.urdf_path}")
        self.robot = URDF.load(str(self.urdf_path))
        self.wrist_link = wrist_link
        self.fingertip_links = list(fingertip_links)
        self.joint_names = list(joint_names) if joint_names is not None else list(self.robot.actuated_joint_names)
        self.tip_point = tip_point
        self.joint_limit_overrides = joint_limit_overrides or {}
        if self.wrist_link not in self.robot.link_map:
            raise ValueError(f"Robot hand URDF has no wrist link {self.wrist_link!r}.")
        for link_name in self.fingertip_links:
            if link_name not in self.robot.link_map:
                raise ValueError(f"Robot hand URDF has no fingertip link {link_name!r}.")
        for joint_name in self.joint_names:
            if joint_name not in self.robot.joint_map:
                raise ValueError(f"Robot hand URDF has no actuated joint {joint_name!r}.")

        self.zero_action = np.zeros(len(self.joint_names), dtype=np.float64)
        self.tip_offsets = self._mesh_tip_offsets() if tip_point == "mesh_tip" else None

    def forward(self, action: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(action, dtype=np.float64)
        self.robot.update_cfg(dict(zip(self.joint_names, q)))
        world_to_wrist = np.linalg.inv(self.robot.get_transform(self.wrist_link))
        tips = []
        for link_name in self.fingertip_links:
            transform = self.robot.get_transform(link_name)
            if self.tip_offsets is None:
                tip_world = transform[:3, 3].copy()
            else:
                tip_world = transform[:3, :3] @ self.tip_offsets[link_name] + transform[:3, 3]
            tips.append((world_to_wrist @ np.append(tip_world, 1.0))[:3])
        return {
            "wrist": np.zeros(3, dtype=np.float64),
            "fingertips": np.stack(tips, axis=0),
        }

    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in self.joint_names:
            if name in self.joint_limit_overrides:
                lo, hi = self.joint_limit_overrides[name]
            else:
                limit = self.robot.joint_map[name].limit
                lo = float(limit.lower) if limit is not None and limit.lower is not None else -np.pi
                hi = float(limit.upper) if limit is not None and limit.upper is not None else np.pi
            if hi <= lo:
                hi = lo + 1e-9
            lower.append(lo)
            upper.append(hi)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def _mesh_tip_offsets(self) -> dict[str, np.ndarray]:
        import trimesh

        offsets: dict[str, np.ndarray] = {}
        for link_name in self.fingertip_links:
            link = self.robot.link_map[link_name]
            candidates = []
            for visual in link.visuals:
                mesh = visual.geometry.mesh
                if mesh is None:
                    continue
                mesh_path = self.urdf_path.parent / mesh.filename
                if not mesh_path.exists():
                    mesh_path = self.urdf_path.parent / Path(mesh.filename).name
                loaded = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(loaded.vertices, dtype=np.float64)
                if mesh.scale is not None:
                    vertices = vertices * np.asarray(mesh.scale, dtype=np.float64)
                vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
                vertices_link = (visual.origin @ vertices_h.T).T[:, :3]
                candidates.append(vertices_link)
            if not candidates:
                offsets[link_name] = np.zeros(3, dtype=np.float64)
                continue
            vertices = np.concatenate(candidates, axis=0)
            offsets[link_name] = vertices[np.argmax(np.linalg.norm(vertices, axis=1))]
        return offsets


def _parse_vec(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None or value.strip() == "":
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(part) for part in value.split()], dtype=np.float64)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    # MuJoCo quaternions are stored as w x y z.
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


@dataclass
class _MjcfBody:
    name: str
    parent: str | None
    pos: np.ndarray
    quat: np.ndarray
    joint_names: list[str]
    geoms: list[str]


@dataclass
class _MjcfJoint:
    name: str
    body: str
    axis: np.ndarray
    range: tuple[float, float]


@dataclass
class _MjcfActuator:
    name: str
    range: tuple[float, float]
    joint: str | None = None
    tendon: str | None = None


class _MjcfHandKinematics:
    def __init__(
        self,
        xml_path: str | Path,
        wrist_link: str,
        fingertip_links: list[str],
        joint_names: list[str] | None,
        tip_point: str,
    ) -> None:
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Robot hand XML does not exist: {self.xml_path}")
        self.wrist_link = wrist_link
        self.fingertip_links = list(fingertip_links)
        self.tip_point = tip_point
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.mesh_dir = self.xml_path.parent / self.root.find("compiler").get("meshdir", ".") if self.root.find("compiler") is not None else self.xml_path.parent

        self.default_ranges = self._parse_default_ranges()
        self.mesh_assets = self._parse_mesh_assets()
        self.bodies: dict[str, _MjcfBody] = {}
        self.joints: dict[str, _MjcfJoint] = {}
        self._parse_bodies()
        self.tendons = self._parse_fixed_tendons()
        self.actuators = self._parse_actuators()

        if self.wrist_link not in self.bodies:
            raise ValueError(f"Robot XML has no wrist body {self.wrist_link!r}.")
        for link_name in self.fingertip_links:
            if link_name not in self.bodies:
                raise ValueError(f"Robot XML has no fingertip body {link_name!r}.")

        if self.actuators and joint_names is None:
            self.joint_names = [actuator.name for actuator in self.actuators]
        else:
            self.joint_names = list(joint_names) if joint_names is not None else list(self.joints)
        self.zero_action = np.zeros(len(self.joint_names), dtype=np.float64)
        self.tip_offsets = self._mesh_tip_offsets() if tip_point == "mesh_tip" else None

    @property
    def urdf_path(self) -> Path:
        return self.xml_path

    def forward(self, action: np.ndarray) -> dict[str, np.ndarray]:
        joint_values = self._action_to_joint_values(np.asarray(action, dtype=np.float64))
        transforms = self._body_transforms(joint_values)
        world_to_wrist = np.linalg.inv(transforms[self.wrist_link])
        tips = []
        for link_name in self.fingertip_links:
            transform = transforms[link_name]
            if self.tip_offsets is None:
                tip_world = transform[:3, 3].copy()
            else:
                tip_world = transform[:3, :3] @ self.tip_offsets[link_name] + transform[:3, 3]
            tips.append((world_to_wrist @ np.append(tip_world, 1.0))[:3])
        return {"wrist": np.zeros(3, dtype=np.float64), "fingertips": np.stack(tips, axis=0)}

    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.actuators:
            ranges = {actuator.name: actuator.range for actuator in self.actuators}
        else:
            ranges = {name: joint.range for name, joint in self.joints.items()}
        lower, upper = zip(*(ranges[name] for name in self.joint_names))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def mesh_vertices(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        joint_values = self._action_to_joint_values(np.asarray(action, dtype=np.float64))
        transforms = self._body_transforms(joint_values)
        world_to_wrist = np.linalg.inv(transforms[self.wrist_link])
        meshes = []
        for body_name, body in self.bodies.items():
            for mesh_name in body.geoms:
                mesh_path = self.mesh_assets.get(mesh_name)
                if mesh_path is None:
                    continue
                mesh = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
                vertices_wrist = (world_to_wrist @ transforms[body_name] @ vertices_h.T).T[:, :3]
                transformed = trimesh.Trimesh(vertices=vertices_wrist, faces=np.asarray(mesh.faces), process=False)
                meshes.append(transformed)
        if not meshes:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        combined = trimesh.util.concatenate(meshes)
        return np.asarray(combined.vertices, dtype=np.float64), np.asarray(combined.faces, dtype=np.int64)

    def _parse_default_ranges(self) -> dict[str, tuple[float, float]]:
        ranges = {}
        for default in self.root.findall("./default/default"):
            class_name = default.get("class")
            if not class_name:
                continue
            joint = default.find("joint")
            position = default.find("position")
            if joint is not None and joint.get("range"):
                value = _parse_vec(joint.get("range"), (0.0, 0.0))
                ranges[f"joint:{class_name}"] = (float(value[0]), float(value[1]))
            if position is not None and position.get("ctrlrange"):
                value = _parse_vec(position.get("ctrlrange"), (0.0, 0.0))
                ranges[f"actuator:{class_name}"] = (float(value[0]), float(value[1]))
        return ranges

    def _parse_mesh_assets(self) -> dict[str, Path]:
        assets = {}
        for mesh in self.root.findall("./asset/mesh"):
            name = mesh.get("name")
            filename = mesh.get("file")
            if name and filename:
                assets[name] = self.mesh_dir / filename
        return assets

    def _parse_bodies(self) -> None:
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"Robot XML has no worldbody: {self.xml_path}")
        for body in worldbody.findall("body"):
            self._parse_body(body, None)

    def _parse_body(self, elem: ET.Element, parent: str | None) -> None:
        name = elem.get("name")
        if not name:
            return
        joint_names = []
        geoms = []
        for joint in elem.findall("joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            range_value = _parse_vec(joint.get("range"), self.default_ranges.get(f"joint:{joint.get('class', '')}", (0.0, 0.0)))
            axis = _parse_vec(joint.get("axis"), (0.0, 0.0, 1.0))
            self.joints[joint_name] = _MjcfJoint(
                name=joint_name,
                body=name,
                axis=axis,
                range=(float(range_value[0]), float(range_value[1])),
            )
            joint_names.append(joint_name)
        for geom in elem.findall("geom"):
            if geom.get("type") == "mesh" and geom.get("mesh"):
                geoms.append(str(geom.get("mesh")))
        self.bodies[name] = _MjcfBody(
            name=name,
            parent=parent,
            pos=_parse_vec(elem.get("pos"), (0.0, 0.0, 0.0)),
            quat=_parse_vec(elem.get("quat"), (1.0, 0.0, 0.0, 0.0)),
            joint_names=joint_names,
            geoms=geoms,
        )
        for child in elem.findall("body"):
            self._parse_body(child, name)

    def _parse_fixed_tendons(self) -> dict[str, list[tuple[str, float]]]:
        tendons = {}
        for fixed in self.root.findall("./tendon/fixed"):
            name = fixed.get("name")
            if not name:
                continue
            joints = []
            for joint in fixed.findall("joint"):
                joint_name = joint.get("joint")
                if joint_name:
                    joints.append((joint_name, float(joint.get("coef", "1.0"))))
            tendons[name] = joints
        return tendons

    def _parse_actuators(self) -> list[_MjcfActuator]:
        actuators = []
        for position in self.root.findall("./actuator/position"):
            name = position.get("name")
            if not name:
                continue
            if position.get("ctrlrange"):
                range_value = _parse_vec(position.get("ctrlrange"), (0.0, 0.0))
            else:
                range_value = np.asarray(
                    self.default_ranges.get(f"actuator:{position.get('class', '')}", (0.0, 0.0)),
                    dtype=np.float64,
                )
            actuators.append(
                _MjcfActuator(
                    name=name,
                    range=(float(range_value[0]), float(range_value[1])),
                    joint=position.get("joint"),
                    tendon=position.get("tendon"),
                )
            )
        return actuators

    def _action_to_joint_values(self, action: np.ndarray) -> dict[str, float]:
        joint_values = {name: 0.0 for name in self.joints}
        if self.actuators:
            actuator_map = {actuator.name: actuator for actuator in self.actuators}
            for name, value in zip(self.joint_names, action):
                actuator = actuator_map[name]
                if actuator.joint:
                    joint_values[actuator.joint] = float(value)
                elif actuator.tendon:
                    for joint_name, coef in self.tendons.get(actuator.tendon, []):
                        joint_values[joint_name] = float(value) * coef
        else:
            for name, value in zip(self.joint_names, action):
                joint_values[name] = float(value)
        return joint_values

    def _body_transforms(self, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {}
        pending = list(self.bodies)
        while pending:
            progressed = False
            for name in pending[:]:
                body = self.bodies[name]
                if body.parent is not None and body.parent not in transforms:
                    continue
                parent_transform = transforms[body.parent] if body.parent is not None else np.eye(4)
                transform = parent_transform @ _make_transform(_quat_to_matrix(body.quat), body.pos)
                for joint_name in body.joint_names:
                    joint = self.joints[joint_name]
                    transform = transform @ _make_transform(
                        _axis_angle_matrix(joint.axis, joint_values.get(joint_name, 0.0)),
                        np.zeros(3, dtype=np.float64),
                    )
                transforms[name] = transform
                pending.remove(name)
                progressed = True
            if not progressed:
                raise ValueError(f"Robot XML has cyclic or invalid body hierarchy: {self.xml_path}")
        return transforms

    def _mesh_tip_offsets(self) -> dict[str, np.ndarray]:
        import trimesh

        offsets = {}
        for body_name in self.fingertip_links:
            vertices_list = []
            for mesh_name in self.bodies[body_name].geoms:
                mesh_path = self.mesh_assets.get(mesh_name)
                if mesh_path is None:
                    continue
                mesh = trimesh.load_mesh(str(mesh_path), process=False)
                vertices_list.append(np.asarray(mesh.vertices, dtype=np.float64))
            if not vertices_list:
                offsets[body_name] = np.zeros(3, dtype=np.float64)
                continue
            vertices = np.concatenate(vertices_list, axis=0)
            offsets[body_name] = vertices[np.argmax(np.linalg.norm(vertices, axis=1))]
        return offsets


class Phase6ProstheticAction:
    """Map MANO fingertips to a configurable fixed-wrist robot hand action."""

    def __init__(self, config: Phase6ProstheticActionConfig | None = None) -> None:
        self.config = config or Phase6ProstheticActionConfig()
        self._model: _RobotHandKinematics | _MjcfHandKinematics | None = None
        self._last_scale = 1.0
        self._resolved_profile = self.config.robot_profile

    def run(self, phase5_result: Phase5ManoResult | Any) -> Phase6ProstheticActionResult:
        if not isinstance(phase5_result, Phase5ManoResult):
            return self._empty_result("invalid_input", "Phase6 expected a Phase5ManoResult.")
        if phase5_result.status != "ok" or not phase5_result.hands:
            return self._empty_result("no_mano_hand", "No MANO hand prediction is available.")

        hand = self._select_hand(phase5_result.hands)
        mano_wrist, mano_fingertips = self._mano_wrist_and_fingertips(hand)
        target_wrist = self._map_mano_to_robot_wrist_frame(hand, mano_wrist, mano_fingertips)
        action, prosthetic_wrist = self._solve_action(target_wrist)
        error = np.linalg.norm(prosthetic_wrist - target_wrist, axis=1)
        model = self._ensure_model()

        return Phase6ProstheticActionResult(
            status="ok",
            message=(
                "Computed fixed-wrist robot hand action from MANO wrist-relative "
                "fingertip targets."
            ),
            selected_hand_index=hand.hand_index,
            action_names=list(self.config.action_names),
            action=action.astype(np.float32),
            mano_wrist=mano_wrist.astype(np.float32),
            mano_fingertips=mano_fingertips.astype(np.float32),
            target_fingertips_wrist=target_wrist.astype(np.float32),
            prosthetic_fingertips_wrist=prosthetic_wrist.astype(np.float32),
            fingertip_error=error.astype(np.float32),
            metadata={
                "finger_names": list(_FINGER_NAMES),
                "mano_fingertip_indices": _MANO_FINGERTIP_INDICES.astype(int).tolist(),
                "target_space": "robot_fixed_wrist_frame",
                "robot_profile": self._resolved_profile,
                "model_format": "xml" if isinstance(model, _MjcfHandKinematics) else "urdf",
                "robot_model_path": str(model.urdf_path),
                "robot_urdf_path": str(model.urdf_path),
                "wrist_link": model.wrist_link,
                "tip_point": model.tip_point,
                "fingertip_links": list(model.fingertip_links),
                "scale": float(self._last_scale),
                "preserve_scale": bool(self.config.preserve_scale),
                "optimization_restarts": int(self.config.optimization_restarts),
                "max_nfev": int(self.config.max_nfev),
                "regularization_weight": float(self.config.regularization_weight),
            },
        )

    def _empty_result(self, status: str, message: str) -> Phase6ProstheticActionResult:
        return Phase6ProstheticActionResult(
            status=status,
            message=message,
            selected_hand_index=None,
            action_names=list(self.config.action_names),
            action=np.zeros(len(self.config.action_names), dtype=np.float32),
            mano_wrist=None,
            mano_fingertips=None,
            target_fingertips_wrist=None,
            prosthetic_fingertips_wrist=None,
            fingertip_error=None,
            metadata={},
        )

    def _ensure_model(self) -> _RobotHandKinematics | _MjcfHandKinematics:
        if self._model is None:
            model_path, model_format, wrist_link, fingertip_links, joint_names, tip_point, profile, joint_limits = (
                self.config.resolved_robot()
            )
            if model_format == "xml":
                self._model = _MjcfHandKinematics(
                    model_path,
                    wrist_link,
                    fingertip_links,
                    joint_names,
                    tip_point,
                )
            else:
                self._model = _RobotHandKinematics(
                    model_path,
                    wrist_link,
                    fingertip_links,
                    joint_names,
                    tip_point,
                    joint_limits,
                )
            self._resolved_profile = profile
            self.config.action_names = list(self._model.joint_names)
        return self._model

    def _select_hand(self, hands: list[Phase5HandPrediction]) -> Phase5HandPrediction:
        preferred = hands
        if self.config.hand_preference != "any":
            want_right = self.config.hand_preference == "right"
            filtered = [hand for hand in hands if hand.is_right == want_right]
            if filtered:
                preferred = filtered
        return max(preferred, key=lambda hand: hand.keypoint_score_mean)

    def _mano_wrist_and_fingertips(
        self, hand: Phase5HandPrediction
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.asarray(hand.keypoints_3d, dtype=np.float64)
        if keypoints.shape[0] <= int(_MANO_FINGERTIP_INDICES.max()) or keypoints.shape[1] != 3:
            raise ValueError(f"Expected MANO keypoints with shape (>=21, 3), got {keypoints.shape}.")
        if not hand.is_right:
            keypoints = keypoints.copy()
            keypoints[:, 0] *= -1.0
        return keypoints[0].copy(), keypoints[_MANO_FINGERTIP_INDICES].copy()

    def _map_mano_to_robot_wrist_frame(
        self,
        hand: Phase5HandPrediction,
        mano_wrist: np.ndarray,
        mano_fingertips: np.ndarray,
    ) -> np.ndarray:
        model = self._ensure_model()
        robot_open = model.forward(model.zero_action)["fingertips"]
        mano_keypoints = np.asarray(hand.keypoints_3d, dtype=np.float64)
        if not hand.is_right:
            mano_keypoints = mano_keypoints.copy()
            mano_keypoints[:, 0] *= -1.0

        mano_frame = _build_hand_frame(
            wrist=mano_keypoints[0],
            index_mcp=mano_keypoints[5],
            middle_mcp=mano_keypoints[9],
            little_mcp=mano_keypoints[17],
        )
        robot_frame = _build_hand_frame(
            wrist=np.zeros(3),
            index_mcp=robot_open[1],
            middle_mcp=robot_open[2],
            little_mcp=robot_open[4],
        )

        mano_rel = mano_fingertips - mano_wrist
        mano_local = mano_rel @ mano_frame
        if self.config.preserve_scale:
            scale = 1.0
        else:
            mano_mid = max(float(np.linalg.norm(mano_rel[2])), 1e-8)
            robot_mid = max(float(np.linalg.norm(robot_open[2])), 1e-8)
            scale = float(np.clip(robot_mid / mano_mid, self.config.min_scale, self.config.max_scale))
        self._last_scale = scale
        return (mano_local * scale) @ robot_frame.T

    def _solve_action(self, target_fingertips_wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.optimize import least_squares

        model = self._ensure_model()
        lower, upper = model.joint_bounds()
        q0 = np.clip(model.zero_action.copy(), lower, upper)
        rng = np.random.default_rng(self.config.random_seed)
        starts = [q0, np.clip(0.5 * (lower + upper), lower, upper)]
        for _ in range(max(self.config.optimization_restarts - len(starts), 0)):
            starts.append(rng.uniform(lower, upper))

        def residual(q: np.ndarray) -> np.ndarray:
            tips = model.forward(q)["fingertips"]
            fingertip_residual = (tips - target_fingertips_wrist).reshape(-1) * 100.0
            regularizer = self.config.regularization_weight * q
            return np.concatenate([fingertip_residual, regularizer])

        best_result = None
        for start in starts:
            result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=self.config.max_nfev,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                x_scale="jac",
                diff_step=1e-4,
            )
            if best_result is None or result.cost < best_result.cost:
                best_result = result
        action = best_result.x
        tips = model.forward(action)["fingertips"]
        return action, tips


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value
    return value / norm


def _build_hand_frame(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    little_mcp: np.ndarray,
) -> np.ndarray:
    forward = _normalize(middle_mcp - wrist)
    lateral = _normalize(index_mcp - little_mcp)
    normal = _normalize(np.cross(lateral, forward))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = _normalize(np.cross(forward, normal))
    return np.stack([lateral, forward, normal], axis=1)
