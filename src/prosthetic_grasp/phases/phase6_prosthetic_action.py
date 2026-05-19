from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from prosthetic_grasp.common.types import (
    Phase5HandPrediction,
    Phase5ManoResult,
    Phase6ProstheticActionResult,
)


_MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
_FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


@dataclass
class Phase6ProstheticActionConfig:
    """Simplified MANO-to-folding-hand retargeting.

    The target only preserves five MANO fingertip positions relative to the
    wrist. MANO fingertips follow OmniDexGrasp's joint indices [4, 8, 12, 16,
    20]. Folding-hand fingertips are virtual points attached to distal links.
    """

    mjcf_path: str = "hand/folding_hand/folding.xml"
    mesh_dir: str = "hand/folding_hand/meshes"
    hand_preference: str = "right"
    grid_samples: int = 21
    refine_iterations: int = 2
    preserve_scale: bool = False
    min_scale: float = 0.25
    max_scale: float = 4.0
    action_names: list[str] = field(
        default_factory=lambda: [
            "th_base",
            "th",
            "ff_base",
            "ff",
            "mf",
            "rf_base",
            "rf",
            "lf_base",
            "lf",
        ]
    )

    def __post_init__(self) -> None:
        self.mjcf_path = self.mjcf_path.strip()
        self.mesh_dir = self.mesh_dir.strip()
        self.hand_preference = self.hand_preference.strip().lower()
        if self.hand_preference not in {"right", "left", "any"}:
            raise ValueError(
                "hand_preference must be 'right', 'left', or 'any', "
                f"got {self.hand_preference!r}."
            )
        if self.grid_samples < 3:
            raise ValueError(f"grid_samples must be at least 3, got {self.grid_samples}.")
        if self.refine_iterations < 0:
            raise ValueError(
                f"refine_iterations must be non-negative, got {self.refine_iterations}."
            )
        if self.min_scale <= 0 or self.max_scale <= 0 or self.min_scale > self.max_scale:
            raise ValueError(
                "min_scale and max_scale must be positive and ordered, "
                f"got ({self.min_scale}, {self.max_scale})."
            )


@dataclass
class _BodySpec:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    joint_name: str | None
    joint_axis: np.ndarray | None
    children: list["_BodySpec"]


class _FoldingHandKinematics:
    distal_links = {
        "thumb": "th_distal",
        "index": "ff_distal",
        "middle": "mf_distal",
        "ring": "rf_distal",
        "little": "lf_distal",
    }

    fallback_tip_offsets = {
        "th_distal": np.array([-0.020, 0.030, 0.0], dtype=np.float64),
        "ff_distal": np.array([0.0, 0.040, 0.0], dtype=np.float64),
        "mf_distal": np.array([0.0, 0.044, 0.0], dtype=np.float64),
        "rf_distal": np.array([0.0, 0.040, 0.0], dtype=np.float64),
        "lf_distal": np.array([0.0, 0.034, 0.0], dtype=np.float64),
    }

    action_ranges = {
        "th_base": (0.0, 1.0),
        "th": (0.0, 1.0),
        "ff_base": (0.0, 1.0),
        "ff": (0.0, float(np.pi)),
        "mf": (0.0, float(np.pi)),
        "rf_base": (0.0, 1.0),
        "rf": (0.0, float(np.pi)),
        "lf_base": (0.0, 1.0),
        "lf": (0.0, float(np.pi)),
    }

    def __init__(self, mjcf_path: str | Path, mesh_dir: str | Path) -> None:
        self.mjcf_path = Path(mjcf_path)
        self.mesh_dir = Path(mesh_dir)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"Folding-hand MJCF does not exist: {self.mjcf_path}")
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Folding-hand mesh directory does not exist: {self.mesh_dir}")

        self.root = self._parse_mjcf(self.mjcf_path)
        self.tip_offsets = self._load_tip_offsets()
        self.joint_names = self._joint_names()
        self.zero_action = np.zeros(9, dtype=np.float64)

    def forward(self, action: np.ndarray) -> dict[str, np.ndarray]:
        joint_pose = self._expand_action(action)
        transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._forward_body(self.root, np.eye(3), np.zeros(3), joint_pose, transforms)

        tips = []
        for finger in _FINGER_NAMES:
            link_name = self.distal_links[finger]
            rotation, translation = transforms[link_name]
            tips.append(translation + rotation @ self.tip_offsets[link_name])

        return {
            "wrist": np.zeros(3, dtype=np.float64),
            "fingertips": np.stack(tips, axis=0),
        }

    def _parse_mjcf(self, path: Path) -> _BodySpec:
        root_xml = ET.parse(str(path)).getroot()
        worldbody = root_xml.find("worldbody")
        if worldbody is None:
            raise ValueError(f"MJCF has no worldbody: {path}")
        body_xml = worldbody.find("body")
        if body_xml is None:
            raise ValueError(f"MJCF worldbody has no root body: {path}")
        return self._parse_body(body_xml)

    def _parse_body(self, body_xml: ET.Element) -> _BodySpec:
        joint_xml = body_xml.find("joint")
        joint_name = None
        joint_axis = None
        if joint_xml is not None:
            joint_name = joint_xml.get("name")
            joint_axis = _normalize(_parse_vec(joint_xml.get("axis"), default=(0.0, 0.0, 1.0)))

        children = [self._parse_body(child) for child in body_xml.findall("body")]
        return _BodySpec(
            name=str(body_xml.get("name")),
            pos=_parse_vec(body_xml.get("pos"), default=(0.0, 0.0, 0.0)),
            quat=_parse_vec(body_xml.get("quat"), default=(1.0, 0.0, 0.0, 0.0), size=4),
            joint_name=joint_name,
            joint_axis=joint_axis,
            children=children,
        )

    def _forward_body(
        self,
        body: _BodySpec,
        parent_rotation: np.ndarray,
        parent_translation: np.ndarray,
        joint_pose: dict[str, float],
        transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        rotation = parent_rotation @ _quat_wxyz_to_matrix(body.quat)
        translation = parent_translation + parent_rotation @ body.pos
        if body.joint_name is not None and body.joint_axis is not None:
            angle = joint_pose.get(body.joint_name, 0.0)
            rotation = rotation @ _axis_angle_to_matrix(body.joint_axis, angle)

        transforms[body.name] = (rotation, translation)
        for child in body.children:
            self._forward_body(child, rotation, translation, joint_pose, transforms)

    def _expand_action(self, action: np.ndarray) -> dict[str, float]:
        q = {name: 0.0 for name in self.joint_names}
        q["th_base"] = float(action[0])
        q["th_pip"] = 0.5 * float(action[1])
        q["th_mip"] = 0.8 * float(action[1])
        q["th_dip"] = 0.8 * float(action[1])
        q["ff_base"] = float(action[2])
        q["ff_pip"] = float(action[3])
        q["ff_dip"] = float(action[3])
        q["mf_pip"] = float(action[4])
        q["mf_dip"] = float(action[4])
        q["rf_base"] = float(action[5])
        q["rf_pip"] = float(action[6])
        q["rf_dip"] = float(action[6])
        q["lf_base"] = float(action[7])
        q["lf_pip"] = float(action[8])
        q["lf_dip"] = float(action[8])
        return q

    def _joint_names(self) -> list[str]:
        names: list[str] = []

        def visit(body: _BodySpec) -> None:
            if body.joint_name is not None:
                names.append(body.joint_name)
            for child in body.children:
                visit(child)

        visit(self.root)
        return names

    def _load_tip_offsets(self) -> dict[str, np.ndarray]:
        offsets = {}
        for link_name in self.distal_links.values():
            mesh_path = self.mesh_dir / f"{link_name}.STL"
            try:
                vertices = _read_stl_vertices(mesh_path)
                if vertices.size == 0:
                    raise ValueError("empty STL")
                offsets[link_name] = vertices[np.argmax(np.linalg.norm(vertices, axis=1))]
            except Exception:
                offsets[link_name] = self.fallback_tip_offsets[link_name].copy()
        return offsets


class Phase6ProstheticAction:
    """Map MANO fingertip geometry to a simplified folding-hand action."""

    def __init__(self, config: Phase6ProstheticActionConfig | None = None) -> None:
        self.config = config or Phase6ProstheticActionConfig()
        self._model: _FoldingHandKinematics | None = None

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

        return Phase6ProstheticActionResult(
            status="ok",
            message=(
                "Computed simplified folding-hand action from MANO wrist-relative "
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
                "target_space": "folding_hand_base_link_wrist_relative",
                "scale": float(self._last_scale),
                "preserve_scale": bool(self.config.preserve_scale),
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

    def _ensure_model(self) -> _FoldingHandKinematics:
        if self._model is None:
            self._model = _FoldingHandKinematics(self.config.mjcf_path, self.config.mesh_dir)
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
        model = self._ensure_model()
        action = model.zero_action.copy()
        solve_specs = [
            (0, [0, 1]),
            (1, [2, 3]),
            (2, [4]),
            (3, [5, 6]),
            (4, [7, 8]),
        ]
        for finger_index, action_indices in solve_specs:
            action = self._solve_finger(model, action, finger_index, action_indices, target_fingertips_wrist[finger_index])
        tips = model.forward(action)["fingertips"]
        return action, tips

    def _solve_finger(
        self,
        model: _FoldingHandKinematics,
        action: np.ndarray,
        finger_index: int,
        action_indices: list[int],
        target: np.ndarray,
    ) -> np.ndarray:
        best = action.copy()
        best_loss = self._finger_loss(model, best, finger_index, target)
        ranges = [model.action_ranges[self.config.action_names[i]] for i in action_indices]

        centers = [0.5 * (lo + hi) for lo, hi in ranges]
        widths = [hi - lo for lo, hi in ranges]
        for iteration in range(self.config.refine_iterations + 1):
            candidate_values = []
            for (lo, hi), center, width in zip(ranges, centers, widths):
                if iteration == 0:
                    values = np.linspace(lo, hi, self.config.grid_samples)
                else:
                    half = 0.5 * width
                    values = np.linspace(max(lo, center - half), min(hi, center + half), self.config.grid_samples)
                candidate_values.append(values)

            for values in _cartesian_product(candidate_values):
                candidate = best.copy()
                for idx, value in zip(action_indices, values):
                    candidate[idx] = value
                loss = self._finger_loss(model, candidate, finger_index, target)
                if loss < best_loss:
                    best_loss = loss
                    best = candidate
                    centers = list(values)

            widths = [max(width / max(self.config.grid_samples - 1, 1), 1e-6) for width in widths]

        return best

    @staticmethod
    def _finger_loss(
        model: _FoldingHandKinematics,
        action: np.ndarray,
        finger_index: int,
        target: np.ndarray,
    ) -> float:
        tip = model.forward(action)["fingertips"][finger_index]
        diff = tip - target
        return float(diff @ diff)


def _cartesian_product(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0][:, None]
    mesh = np.meshgrid(*arrays, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=-1)


def _parse_vec(value: str | None, default: tuple[float, ...], size: int = 3) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float64)
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.size != size:
        raise ValueError(f"Expected vector of length {size}, got {value!r}.")
    return parsed


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value
    return value / norm


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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


def _read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack("<I", data[80:84])[0]
        expected = 84 + triangle_count * 50
        if expected == len(data):
            vertices = []
            offset = 84
            for _ in range(triangle_count):
                tri = struct.unpack("<12fH", data[offset : offset + 50])
                vertices.extend((tri[3:6], tri[6:9], tri[9:12]))
                offset += 50
            return np.asarray(vertices, dtype=np.float64)

    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)
