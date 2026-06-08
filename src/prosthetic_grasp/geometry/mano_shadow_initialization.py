from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ManoShadowInitialization:
    """Shadow action initialized from a MANO hand pose."""

    action: np.ndarray
    raw_action: np.ndarray
    clipped_action: np.ndarray
    action_names: tuple[str, ...]
    clipped_joint_names: tuple[str, ...]

    @property
    def num_clipped(self) -> int:
        return len(self.clipped_joint_names)


def mano_pose_to_shadow_action(
    mano_pose: np.ndarray,
    action_names: list[str] | tuple[str, ...],
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    scale: float = 1.0,
) -> ManoShadowInitialization:
    """Map MANO axis-angle finger pose to a Shadow Hand joint action.

    This follows OmniDexGrasp's hand-written MANO-to-Shadow initialization:
    MANO joint axis-angles are converted to XYZ Euler angles, then selected
    flexion/abduction components are copied to Shadow joints. The returned
    action is ordered by ``action_names`` and clipped if joint bounds are given.
    """

    names = tuple(str(name) for name in action_names)
    if not names:
        raise ValueError("action_names must not be empty.")
    pose = np.asarray(mano_pose, dtype=np.float64)
    if pose.shape == (1, 45):
        pose = pose[0]
    if pose.shape != (45,):
        raise ValueError(f"mano_pose must have shape (45,) or (1,45), got {pose.shape}.")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")

    euler = matrix_to_euler_xyz(axis_angle_to_matrix(pose.reshape(15, 3)))
    values = _omnidexgrasp_shadow_values(euler)
    raw = np.zeros(len(names), dtype=np.float64)
    unknown = []
    for index, name in enumerate(names):
        key = normalize_shadow_joint_name(name)
        if key not in values:
            unknown.append(name)
            continue
        raw[index] = values[key] * scale
    if unknown:
        raise ValueError(f"Unsupported Shadow action names: {unknown[:5]}.")

    clipped = raw.copy()
    clipped_names: tuple[str, ...] = ()
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("lower and upper must be provided together.")
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != raw.shape or upper.shape != raw.shape:
            raise ValueError(
                f"Bounds must match action shape {raw.shape}, got lower={lower.shape}, upper={upper.shape}."
            )
        clipped = np.clip(raw, lower, upper)
        changed = np.flatnonzero(np.abs(clipped - raw) > 1e-9)
        clipped_names = tuple(names[int(i)] for i in changed)

    return ManoShadowInitialization(
        action=clipped.copy(),
        raw_action=raw,
        clipped_action=clipped,
        action_names=names,
        clipped_joint_names=clipped_names,
    )


def action_summary(action: np.ndarray, action_names: list[str] | tuple[str, ...], *, threshold: float = 1e-6) -> dict:
    """Compact JSON-serializable action statistics for experiment summaries."""

    action = np.asarray(action, dtype=np.float64)
    names = tuple(str(name) for name in action_names)
    nonzero = np.flatnonzero(np.abs(action) > threshold)
    return {
        "num_dofs": int(len(action)),
        "num_nonzero": int(len(nonzero)),
        "l2_norm": float(np.linalg.norm(action)),
        "max_abs": float(np.max(np.abs(action))) if len(action) else 0.0,
        "nonzero_joints": [
            {"name": names[int(index)], "value": float(action[int(index)])}
            for index in nonzero
        ],
    }


def normalize_shadow_joint_name(name: str) -> str:
    """Normalize URDF/MJCF Shadow joint names to bare names such as ``FFJ2``."""

    key = str(name).split(":")[-1]
    if key.startswith(("rh_", "lh_")):
        key = key[3:]
    return key


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion from axis-angle to rotation matrix."""

    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle last dimension must be 3, got {axis_angle.shape}.")
    flat = axis_angle.reshape(-1, 3)
    angle = np.linalg.norm(flat, axis=1)
    axis = np.zeros_like(flat)
    valid = angle > 1e-12
    axis[valid] = flat[valid] / angle[valid, None]

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), k.shape).copy()
    sin = np.sin(angle)[:, None, None]
    one_minus_cos = (1.0 - np.cos(angle))[:, None, None]
    matrix = eye + sin * k + one_minus_cos * (k @ k)
    matrix[~valid] = np.eye(3, dtype=np.float64)
    return matrix.reshape(axis_angle.shape[:-1] + (3, 3))


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """PyTorch3D-compatible ``matrix_to_euler_angles(..., 'XYZ')`` for numpy."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end with shape (3,3), got {matrix.shape}.")
    central = np.arcsin(np.clip(matrix[..., 0, 2], -1.0, 1.0))
    first = np.arctan2(-matrix[..., 2, 1], matrix[..., 2, 2])
    third = np.arctan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    return np.stack([first, central, third], axis=-1)


def _omnidexgrasp_shadow_values(euler: np.ndarray) -> dict[str, float]:
    values = {name: 0.0 for name in _SHADOW_JOINT_ORDER}

    values["FFJ3"] = euler[0, 1]
    values["FFJ2"] = euler[0, 2]
    values["FFJ1"] = euler[1, 2]
    values["FFJ0"] = euler[2, 2]

    values["MFJ3"] = euler[3, 1]
    values["MFJ2"] = euler[3, 2]
    values["MFJ1"] = euler[4, 2]
    values["MFJ0"] = euler[5, 2]

    values["RFJ3"] = euler[9, 1]
    values["RFJ2"] = euler[9, 2]
    values["RFJ1"] = euler[10, 2]
    values["RFJ0"] = euler[11, 2]

    values["LFJ4"] = euler[6, 1] * 0.5 + euler[6, 2] * 0.3
    values["LFJ3"] = euler[6, 1]
    values["LFJ2"] = euler[6, 2]
    values["LFJ1"] = euler[7, 2]
    values["LFJ0"] = euler[8, 2]

    values["THJ4"] = 0.0
    values["THJ3"] = euler[12, 1]
    values["THJ2"] = euler[12, 2]
    values["THJ1"] = euler[13, 2] * 0.5
    values["THJ0"] = euler[13, 2] * 0.3
    return values


_SHADOW_JOINT_ORDER = (
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "FFJ0",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "MFJ0",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "RFJ0",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "LFJ0",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "THJ0",
)
