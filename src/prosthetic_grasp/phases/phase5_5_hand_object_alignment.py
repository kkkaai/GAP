from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prosthetic_grasp.geometry.contact_clustering import closest_points_on_mesh
from prosthetic_grasp.geometry.surface_sampling import sample_mesh_surface


@dataclass
class Phase55HandObjectAlignmentConfig:
    """Align monocular MANO hand predictions to a metric object pose.

    HaMeR-style monocular hand reconstruction gives useful MANO articulation,
    but its global camera translation is not a metric RGB-D pose. This phase
    keeps the recovered hand shape/articulation and estimates a scene transform
    that places the hand in the same metric frame as the object.
    """

    method: str = "translation"
    num_hand_samples: int = 1200
    contact_fraction: float = 0.12
    contact_threshold_m: float = 0.015
    penetration_threshold_m: float = 0.003
    target_contact_distance_m: float = 0.008
    object_center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    se3_maxiter: int = 120
    se3_rotation_bound_rad: float = 0.75
    se3_translation_bound_m: float = 0.16
    depth_valid_min_m: float = 0.05
    depth_valid_max_m: float = 2.50
    depth_sample_percentile: float = 45.0
    random_seed: int = 17
    regularization_weight: float = 0.02
    depenetration_steps: int = 8
    depenetration_margin_m: float = 0.002

    def __post_init__(self) -> None:
        self.method = self.method.strip().lower()
        if self.method not in {"translation", "se3", "rgbd"}:
            raise ValueError(f"Unknown phase5.5 alignment method: {self.method!r}.")
        if not 0 < self.contact_fraction <= 1:
            raise ValueError(f"contact_fraction must be in (0, 1], got {self.contact_fraction}.")


@dataclass
class Phase55HandObjectAlignmentResult:
    status: str
    message: str
    method: str
    transform: np.ndarray
    vertices: np.ndarray
    keypoints_3d: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def align_hand_to_object(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_keypoints_3d: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    config: Phase55HandObjectAlignmentConfig | None = None,
    camera_k: np.ndarray | None = None,
    depth_m: np.ndarray | None = None,
    hand_mask: np.ndarray | None = None,
) -> Phase55HandObjectAlignmentResult:
    config = config or Phase55HandObjectAlignmentConfig()
    hand_vertices = _as_points(hand_vertices)
    hand_keypoints_3d = _as_points(hand_keypoints_3d)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    object_vertices = _as_points(object_vertices)
    object_faces = np.asarray(object_faces, dtype=np.int64)

    if config.method == "translation":
        transform, meta = _translation_alignment(hand_vertices, object_vertices, config)
    elif config.method == "rgbd":
        transform, meta = _rgbd_alignment(
            hand_vertices,
            object_vertices,
            config,
            camera_k=camera_k,
            depth_m=depth_m,
            hand_mask=hand_mask,
        )
    else:
        init_transform, init_meta = _translation_alignment(hand_vertices, object_vertices, config)
        transform, meta = _se3_alignment(
            hand_vertices,
            hand_faces,
            object_vertices,
            object_faces,
            config,
            init_transform=init_transform,
        )
        meta["initialization"] = init_meta

    transform, depenetration_meta = _depenetrate_transform(
        hand_vertices,
        hand_faces,
        object_vertices,
        object_faces,
        transform,
        config,
    )
    if depenetration_meta:
        meta["depenetration"] = depenetration_meta

    aligned_vertices = transform_points(hand_vertices, transform)
    aligned_keypoints = transform_points(hand_keypoints_3d, transform)
    metrics = evaluate_hand_object_alignment(
        aligned_vertices,
        hand_faces,
        aligned_keypoints,
        object_vertices,
        object_faces,
        contact_threshold_m=config.contact_threshold_m,
        penetration_threshold_m=config.penetration_threshold_m,
        num_hand_samples=config.num_hand_samples,
        seed=config.random_seed,
    )
    return Phase55HandObjectAlignmentResult(
        status="ok",
        message="Aligned monocular hand reconstruction to metric object frame.",
        method=config.method,
        transform=transform.astype(np.float64),
        vertices=aligned_vertices.astype(np.float64),
        keypoints_3d=aligned_keypoints.astype(np.float64),
        metrics=metrics,
        metadata=meta,
    )


def evaluate_hand_object_alignment(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_keypoints_3d: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    contact_threshold_m: float = 0.015,
    penetration_threshold_m: float = 0.003,
    num_hand_samples: int = 1200,
    seed: int = 17,
) -> dict[str, float]:
    hand_vertices = _as_points(hand_vertices)
    hand_keypoints_3d = _as_points(hand_keypoints_3d)
    samples = sample_mesh_surface(
        hand_vertices,
        np.asarray(hand_faces, dtype=np.int64),
        num_points=num_hand_samples,
        seed=seed,
        use_farthest_point_sampling=True,
        oversample_factor=8,
    )
    object_points, object_normals, distances = closest_points_on_mesh(
        object_vertices,
        object_faces,
        samples.points,
    )
    signed = np.sum((samples.points - object_points) * object_normals, axis=1)
    fingertip_indices = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
    fingertip_indices = fingertip_indices[fingertip_indices < len(hand_keypoints_3d)]
    tip_metrics: dict[str, float] = {}
    if len(fingertip_indices):
        _, _, tip_distances = closest_points_on_mesh(
            object_vertices,
            object_faces,
            hand_keypoints_3d[fingertip_indices],
        )
        tip_metrics = {
            "mean_fingertip_object_distance_m": float(np.mean(tip_distances)),
            "min_fingertip_object_distance_m": float(np.min(tip_distances)),
        }
    return {
        "hand_object_center_distance_m": float(
            np.linalg.norm(hand_vertices.mean(axis=0) - np.asarray(object_vertices).mean(axis=0))
        ),
        "mean_surface_distance_m": float(np.mean(distances)),
        "median_surface_distance_m": float(np.median(distances)),
        "min_surface_distance_m": float(np.min(distances)),
        "p10_surface_distance_m": float(np.percentile(distances, 10)),
        "contact_sample_count": int(np.sum(distances <= contact_threshold_m)),
        "contact_sample_fraction": float(np.mean(distances <= contact_threshold_m)),
        "penetration_sample_count": int(np.sum(signed < -penetration_threshold_m)),
        "penetration_sample_fraction": float(np.mean(signed < -penetration_threshold_m)),
        "mean_signed_surface_offset_m": float(np.mean(signed)),
        **tip_metrics,
    }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = _as_points(points)
    transform = np.asarray(transform, dtype=np.float64)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def _translation_alignment(
    hand_vertices: np.ndarray,
    object_vertices: np.ndarray,
    config: Phase55HandObjectAlignmentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    hand_center = hand_vertices.mean(axis=0)
    object_center = object_vertices.mean(axis=0) + np.asarray(config.object_center_offset_m, dtype=np.float64)
    translation = object_center - hand_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform, {
        "rule": "match_hand_center_to_object_center",
        "hand_center_before_m": hand_center.tolist(),
        "target_object_center_m": object_center.tolist(),
        "translation_m": translation.tolist(),
    }


def _rgbd_alignment(
    hand_vertices: np.ndarray,
    object_vertices: np.ndarray,
    config: Phase55HandObjectAlignmentConfig,
    *,
    camera_k: np.ndarray | None,
    depth_m: np.ndarray | None,
    hand_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_transform, meta = _translation_alignment(hand_vertices, object_vertices, config)
    meta["rule"] = "rgbd_depth_median_then_center_xy"
    if camera_k is None or depth_m is None or hand_mask is None:
        meta["rgbd_status"] = "missing_camera_depth_or_mask_fallback_to_translation"
        return base_transform, meta

    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(hand_mask) > 0
    if mask.shape[:2] != depth.shape[:2]:
        meta["rgbd_status"] = "mask_depth_shape_mismatch_fallback_to_translation"
        return base_transform, meta
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= config.depth_valid_min_m)
        & (depth <= config.depth_valid_max_m)
    )
    if int(np.sum(valid)) < 30:
        meta["rgbd_status"] = "insufficient_valid_hand_depth_fallback_to_translation"
        meta["valid_depth_pixels"] = int(np.sum(valid))
        return base_transform, meta

    ys, xs = np.nonzero(valid)
    z = depth[valid]
    target_z = float(np.percentile(z, config.depth_sample_percentile))
    k = np.asarray(camera_k, dtype=np.float64)
    target_x = float(np.median((xs - k[0, 2]) * z / k[0, 0]))
    target_y = float(np.median((ys - k[1, 2]) * z / k[1, 1]))
    hand_center = hand_vertices.mean(axis=0)
    object_center = object_vertices.mean(axis=0)
    target = np.array([object_center[0], object_center[1], target_z], dtype=np.float64)
    # If depth comes from a real hand mask, its xy median is more informative
    # than object center. With generated images we usually have no hand depth,
    # so this branch is mostly for future RGB-D captures.
    target[:2] = np.array([target_x, target_y], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = target - hand_center
    meta.update(
        {
            "rgbd_status": "ok",
            "valid_depth_pixels": int(np.sum(valid)),
            "depth_percentile": float(config.depth_sample_percentile),
            "target_hand_center_m": target.tolist(),
            "translation_m": transform[:3, 3].tolist(),
        }
    )
    return transform, meta


def _se3_alignment(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    config: Phase55HandObjectAlignmentConfig,
    *,
    init_transform: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    hand_samples = sample_mesh_surface(
        transform_points(hand_vertices, init_transform),
        hand_faces,
        num_points=config.num_hand_samples,
        seed=config.random_seed,
        use_farthest_point_sampling=True,
        oversample_factor=8,
    )
    _, _, init_distances = closest_points_on_mesh(object_vertices, object_faces, hand_samples.points)
    num_contact = max(8, int(len(init_distances) * config.contact_fraction))
    contact_indices = np.argsort(init_distances)[:num_contact]
    contact_source = hand_samples.points[contact_indices]

    init_inv = np.linalg.inv(init_transform)
    contact_source_raw = transform_points(contact_source, init_inv)
    object_center = object_vertices.mean(axis=0)
    object_tree = cKDTree(object_vertices)
    object_vertex_normals = _vertex_normals(object_vertices, object_faces)
    hand_center_raw = hand_vertices.mean(axis=0)
    initial_translation = init_transform[:3, 3].copy()

    def make_transform(params: np.ndarray) -> np.ndarray:
        rot = Rotation.from_rotvec(params[:3]).as_matrix()
        trans = initial_translation + params[3:6]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        return transform

    def evaluate(params: np.ndarray) -> float:
        transform = make_transform(params)
        contact_points = transform_points(contact_source_raw, transform)
        distances, nearest_indices = object_tree.query(contact_points)
        nearest = object_vertices[np.asarray(nearest_indices, dtype=np.int64)]
        normals = object_vertex_normals[np.asarray(nearest_indices, dtype=np.int64)]
        signed = np.sum((contact_points - nearest) * normals, axis=1)
        contact_loss = np.mean((distances - config.target_contact_distance_m) ** 2)
        penetration_loss = np.mean(np.maximum(-signed, 0.0) ** 2)
        center = transform_points(hand_center_raw[None, :], transform)[0]
        center_loss = np.sum((center - object_center) ** 2)
        rot_reg = np.sum(params[:3] ** 2)
        trans_reg = np.sum(params[3:6] ** 2)
        return float(
            120.0 * contact_loss
            + 180.0 * penetration_loss
            + 0.08 * center_loss
            + config.regularization_weight * (rot_reg + trans_reg)
        )

    bounds = [
        (-config.se3_rotation_bound_rad, config.se3_rotation_bound_rad),
        (-config.se3_rotation_bound_rad, config.se3_rotation_bound_rad),
        (-config.se3_rotation_bound_rad, config.se3_rotation_bound_rad),
        (-config.se3_translation_bound_m, config.se3_translation_bound_m),
        (-config.se3_translation_bound_m, config.se3_translation_bound_m),
        (-config.se3_translation_bound_m, config.se3_translation_bound_m),
    ]
    x0 = np.zeros(6, dtype=np.float64)
    initial_loss = evaluate(x0)
    result = minimize(
        evaluate,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(config.se3_maxiter), "ftol": 1e-11},
    )
    transform = make_transform(np.asarray(result.x, dtype=np.float64))
    return transform, {
        "rule": "se3_contact_refinement",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "initial_loss": float(initial_loss),
        "final_loss": float(result.fun),
        "pose_delta": np.asarray(result.x, dtype=np.float64).tolist(),
        "num_contact_source_points": int(len(contact_source_raw)),
        "initial_translation_m": initial_translation.tolist(),
        "translation_m": transform[:3, 3].tolist(),
    }


def _depenetrate_transform(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    transform: np.ndarray,
    config: Phase55HandObjectAlignmentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.depenetration_steps <= 0:
        return transform, {}
    updated = np.asarray(transform, dtype=np.float64).copy()
    total_shift = np.zeros(3, dtype=np.float64)
    steps = 0
    for _ in range(int(config.depenetration_steps)):
        samples = sample_mesh_surface(
            transform_points(hand_vertices, updated),
            hand_faces,
            num_points=min(int(config.num_hand_samples), 700),
            seed=config.random_seed + steps + 101,
            use_farthest_point_sampling=True,
            oversample_factor=6,
        )
        nearest, normals, _ = closest_points_on_mesh(object_vertices, object_faces, samples.points)
        signed = np.sum((samples.points - nearest) * normals, axis=1)
        bad = signed < config.depenetration_margin_m
        if not np.any(bad):
            break
        depths = config.depenetration_margin_m - signed[bad]
        shift = np.mean(normals[bad] * depths[:, None], axis=0)
        norm = float(np.linalg.norm(shift))
        if norm < 1e-6:
            break
        max_step = 0.025
        if norm > max_step:
            shift = shift / norm * max_step
        updated[:3, 3] += shift
        total_shift += shift
        steps += 1
    if steps == 0:
        return transform, {"steps": 0, "total_shift_m": [0.0, 0.0, 0.0]}
    return updated, {"steps": int(steps), "total_shift_m": total_shift.tolist()}


def _as_points(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected point array with shape (N, 3), got {array.shape}.")
    return array


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = _as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    for axis in range(3):
        np.add.at(normals, faces[:, axis], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals
