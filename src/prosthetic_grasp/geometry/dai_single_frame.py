from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ContactStage = Literal["in_hand", "rest"]


@dataclass(frozen=True)
class ContactStageResult:
    stage: ContactStage
    min_distance: float


@dataclass(frozen=True)
class SingleFrameReference:
    """Single-frame analogue of Do as I Do's hand-object reference state."""

    object_points: np.ndarray
    object_normals: np.ndarray
    hand_points: np.ndarray
    fingertip_targets: np.ndarray
    palm_position: np.ndarray
    palm_rotation: np.ndarray
    q_reference: np.ndarray
    contact_stage: ContactStage
    contact_distance_threshold: float = 0.03


@dataclass(frozen=True)
class DaiSingleFrameWeights:
    """Loss scales mapped from Do as I Do trajectory rewards to one frame."""

    hand_position: float = 1.0
    hand_orientation: float = 0.3
    fingertip_position: float = 1.0
    finger_joint: float = 0.01
    object_relative: float = 1.0
    missing_contact: float = 1.0
    rest_contact: float = 0.5
    penetration: float = 3000.0
    normal_alignment: float = 0.03
    control_regularization: float = 0.01
    warmup_regularization: float = 0.01
    perturbation: float = 1.0
    penetration_margin: float = 0.005


@dataclass(frozen=True)
class PerturbationRobustnessScore:
    nominal_loss: float
    mean_loss: float
    worst_loss: float
    success_rate: float
    score: float


def classify_contact_stage(
    hand_points: np.ndarray,
    object_points: np.ndarray,
    *,
    threshold: float,
) -> ContactStageResult:
    hand = _as_points("hand_points", hand_points)
    obj = _as_points("object_points", object_points)
    if len(hand) == 0 or len(obj) == 0:
        return ContactStageResult(stage="rest", min_distance=float("inf"))
    min_distance = float(np.sqrt(_pairwise_sq_dists(hand, obj).min()))
    return ContactStageResult(
        stage="in_hand" if min_distance < float(threshold) else "rest",
        min_distance=min_distance,
    )


def warmup_loss_terms(
    *,
    robot_points: np.ndarray,
    robot_normals: np.ndarray,
    fingertip_points: np.ndarray,
    palm_position: np.ndarray,
    palm_rotation: np.ndarray,
    q: np.ndarray,
    reference: SingleFrameReference,
    q_kinematic: np.ndarray,
    weights: DaiSingleFrameWeights | None = None,
) -> dict[str, float]:
    """Static warmup objective: fixed object, hand moves toward stable contact."""

    weights = weights or DaiSingleFrameWeights()
    geom = _geometry_terms(
        robot_points=robot_points,
        robot_normals=robot_normals,
        fingertip_points=fingertip_points,
        palm_position=palm_position,
        palm_rotation=palm_rotation,
        q=q,
        reference=reference,
        q_target=q_kinematic,
        weights=weights,
    )
    total = (
        weights.hand_position * geom["hand_position"]
        + weights.hand_orientation * geom["hand_orientation"]
        + weights.fingertip_position * geom["fingertip_position"]
        + weights.finger_joint * geom["finger_joint"]
        + weights.missing_contact * geom["missing_contact"]
        + weights.rest_contact * geom["rest_contact"]
        + weights.penetration * geom["penetration"]
        + weights.normal_alignment * geom["normal_alignment"]
        + weights.control_regularization * geom["control_regularization"]
    )
    return {**geom, "total": float(total)}


def final_loss_terms(
    *,
    robot_points: np.ndarray,
    robot_normals: np.ndarray,
    fingertip_points: np.ndarray,
    palm_position: np.ndarray,
    palm_rotation: np.ndarray,
    q: np.ndarray,
    reference: SingleFrameReference,
    q_kinematic: np.ndarray,
    q_warmup: np.ndarray,
    weights: DaiSingleFrameWeights | None = None,
    perturbation_expected_loss: float = 0.0,
) -> dict[str, float]:
    """Final one-frame tracking objective following Do as I Do reward terms."""

    weights = weights or DaiSingleFrameWeights()
    geom = _geometry_terms(
        robot_points=robot_points,
        robot_normals=robot_normals,
        fingertip_points=fingertip_points,
        palm_position=palm_position,
        palm_rotation=palm_rotation,
        q=q,
        reference=reference,
        q_target=q_kinematic,
        weights=weights,
    )
    object_relative = _object_relative_loss(fingertip_points, reference)
    warmup_regularization = _normalized_mse(q, q_warmup)
    total = (
        weights.hand_position * geom["hand_position"]
        + weights.hand_orientation * geom["hand_orientation"]
        + weights.fingertip_position * geom["fingertip_position"]
        + weights.finger_joint * geom["finger_joint"]
        + weights.object_relative * object_relative
        + weights.missing_contact * geom["missing_contact"]
        + weights.rest_contact * geom["rest_contact"]
        + weights.penetration * geom["penetration"]
        + weights.normal_alignment * geom["normal_alignment"]
        + weights.warmup_regularization * warmup_regularization
        + weights.perturbation * float(perturbation_expected_loss)
    )
    return {
        **geom,
        "object_relative": float(object_relative),
        "warmup_regularization": float(warmup_regularization),
        "perturbation_expected": float(perturbation_expected_loss),
        "total": float(total),
    }


def perturbation_robustness_score(
    *,
    nominal_loss: float,
    perturbed_losses: np.ndarray,
    success_threshold: float,
    mean_weight: float = 1.0,
    worst_weight: float = 0.25,
    success_weight: float = 1.0,
) -> PerturbationRobustnessScore:
    losses = np.asarray(perturbed_losses, dtype=np.float64).reshape(-1)
    if losses.size == 0:
        mean_loss = float(nominal_loss)
        worst_loss = float(nominal_loss)
        success_rate = 1.0 if nominal_loss <= success_threshold else 0.0
    else:
        mean_loss = float(np.mean(losses))
        worst_loss = float(np.max(losses))
        success_rate = float(np.mean(losses <= float(success_threshold)))
    score = (
        -float(nominal_loss)
        - float(mean_weight) * mean_loss
        - float(worst_weight) * worst_loss
        + float(success_weight) * success_rate
    )
    return PerturbationRobustnessScore(
        nominal_loss=float(nominal_loss),
        mean_loss=mean_loss,
        worst_loss=worst_loss,
        success_rate=success_rate,
        score=float(score),
    )


def _geometry_terms(
    *,
    robot_points: np.ndarray,
    robot_normals: np.ndarray,
    fingertip_points: np.ndarray,
    palm_position: np.ndarray,
    palm_rotation: np.ndarray,
    q: np.ndarray,
    reference: SingleFrameReference,
    q_target: np.ndarray,
    weights: DaiSingleFrameWeights,
) -> dict[str, float]:
    robot = _as_points("robot_points", robot_points)
    normals = _normalize_rows(_as_points("robot_normals", robot_normals))
    fingertips = _as_points("fingertip_points", fingertip_points)
    object_points = _as_points("reference.object_points", reference.object_points)
    object_normals = _normalize_rows(_as_points("reference.object_normals", reference.object_normals))
    target_tips = _as_points("reference.fingertip_targets", reference.fingertip_targets)

    distances_sq = _pairwise_sq_dists(robot, object_points)
    nearest_obj_idx = np.argmin(distances_sq, axis=1)
    nearest_obj = object_points[nearest_obj_idx]
    nearest_normals = object_normals[nearest_obj_idx]
    point_to_surface = robot - nearest_obj
    signed_offset = np.sum(point_to_surface * nearest_normals, axis=1)
    penetration = float(np.mean(np.maximum(weights.penetration_margin - signed_offset, 0.0) ** 2))

    min_distance = float(np.sqrt(distances_sq.min())) if distances_sq.size else float("inf")
    missing_contact = 0.0
    rest_contact = 0.0
    if reference.contact_stage == "in_hand":
        missing_contact = max(min_distance - float(reference.contact_distance_threshold), 0.0) ** 2
    else:
        rest_contact = max(float(reference.contact_distance_threshold) - min_distance, 0.0) ** 2

    if len(normals) != len(robot):
        raise ValueError("robot_normals must have the same number of rows as robot_points.")
    normal_dot = np.sum(normals * (-nearest_normals), axis=1)
    normal_alignment = float(np.mean((1.0 - normal_dot) ** 2))

    tip_count = min(len(fingertips), len(target_tips))
    fingertip_position = (
        float(np.mean(np.sum((fingertips[:tip_count] - target_tips[:tip_count]) ** 2, axis=1)))
        if tip_count
        else 0.0
    )
    hand_position = float(np.sum((np.asarray(palm_position, dtype=np.float64) - reference.palm_position) ** 2))
    hand_orientation = _rotation_distance_sq(palm_rotation, reference.palm_rotation)
    finger_joint = _normalized_mse(q, q_target)

    return {
        "hand_position": hand_position,
        "hand_orientation": hand_orientation,
        "fingertip_position": fingertip_position,
        "finger_joint": finger_joint,
        "missing_contact": float(missing_contact),
        "rest_contact": float(rest_contact),
        "penetration": penetration,
        "normal_alignment": normal_alignment,
        "control_regularization": finger_joint,
        "min_contact_distance": min_distance,
    }


def _object_relative_loss(fingertip_points: np.ndarray, reference: SingleFrameReference) -> float:
    fingertips = _as_points("fingertip_points", fingertip_points)
    target = _as_points("reference.fingertip_targets", reference.fingertip_targets)
    if len(fingertips) == 0 or len(target) == 0:
        return 0.0
    object_center = _as_points("reference.object_points", reference.object_points).mean(axis=0)
    n = min(len(fingertips), len(target))
    return float(np.mean(np.sum(((fingertips[:n] - object_center) - (target[:n] - object_center)) ** 2, axis=1)))


def _rotation_distance_sq(rotation: np.ndarray, target: np.ndarray) -> float:
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3, 3):
        raise ValueError(f"rotations must both have shape (3, 3), got {r.shape} and {t.shape}.")
    trace = float(np.trace(t.T @ r))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    return angle * angle


def _normalized_mse(values: np.ndarray, target: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if v.shape != t.shape:
        raise ValueError(f"values and target must have matching shapes, got {v.shape} and {t.shape}.")
    return float(np.mean((v - t) ** 2)) if v.size else 0.0


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values / norms


def _as_points(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}.")
    return array
