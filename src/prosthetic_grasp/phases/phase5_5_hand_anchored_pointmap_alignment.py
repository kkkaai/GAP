from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


TranslationMode = Literal["direct", "scale_along_foundationpose_translation"]


@dataclass
class HandAnchoredPointmapAlignmentConfig:
    """Do-as-I-Do style hand-anchored object pose correction for one image.

    The hand mesh is treated as the metric anchor. A pointmap provides the
    hand-object relative 3D offset in its own scale. We estimate pointmap scale
    from the visible hand surface, then update the object translation so the
    object lies in the same metric camera frame as the hand.
    """

    translation_mode: TranslationMode = "direct"
    max_hand_rays: int = 2000
    min_hand_hits: int = 10
    min_object_mask_pixels: int = 100
    mesh_scale: float = 1.0
    visible_mesh_center_from_mask: bool = True
    random_seed: int = 0


@dataclass
class HandAnchoredPointmapAlignmentResult:
    status: str
    message: str
    object_in_camera_optimized: np.ndarray
    pointmap_scale: float
    object_target_camera: np.ndarray
    hand_visible_center_camera: np.ndarray
    hand_visible_center_pointmap: np.ndarray
    object_center_pointmap: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def align_object_pose_with_hand_anchor(
    *,
    object_vertices_local: np.ndarray,
    object_faces: np.ndarray,
    object_in_camera: np.ndarray,
    hand_vertices_camera: np.ndarray,
    hand_faces: np.ndarray,
    pointmap: np.ndarray,
    camera_k: np.ndarray,
    hand_mask: np.ndarray,
    object_mask: np.ndarray,
    config: HandAnchoredPointmapAlignmentConfig | None = None,
) -> HandAnchoredPointmapAlignmentResult:
    config = config or HandAnchoredPointmapAlignmentConfig()
    object_vertices_local = _as_points(object_vertices_local)
    object_faces = np.asarray(object_faces, dtype=np.int64)
    hand_vertices_camera = _as_points(hand_vertices_camera)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    object_in_camera = np.asarray(object_in_camera, dtype=np.float64)
    if object_in_camera.shape != (4, 4):
        raise ValueError(f"object_in_camera must have shape (4, 4), got {object_in_camera.shape}.")
    pointmap = np.asarray(pointmap, dtype=np.float64)
    if pointmap.ndim != 3 or pointmap.shape[2] != 3:
        raise ValueError(f"pointmap must have shape (H, W, 3), got {pointmap.shape}.")
    camera_k = np.asarray(camera_k, dtype=np.float64)
    if camera_k.shape != (3, 3):
        raise ValueError(f"camera_k must have shape (3, 3), got {camera_k.shape}.")

    pm_h, pm_w = pointmap.shape[:2]
    hand_mask_bool = _resize_mask(hand_mask, pm_w, pm_h)
    object_mask_bool = _resize_mask(object_mask, pm_w, pm_h)
    object_valid = object_mask_bool & np.all(np.isfinite(pointmap), axis=2)
    if int(np.sum(object_valid)) < config.min_object_mask_pixels:
        raise ValueError(
            f"object mask has too few valid pointmap pixels: {int(np.sum(object_valid))}."
        )

    rng = np.random.default_rng(config.random_seed)
    hand_hits, hit_u, hit_v = raycast_first_hits(
        hand_vertices_camera,
        hand_faces,
        hand_mask_bool,
        camera_k,
        max_rays=config.max_hand_rays,
        rng=rng,
    )
    anchor_sample_method = "mask_raycast"
    if len(hand_hits) < config.min_hand_hits:
        hand_hits, hit_u, hit_v = projected_vertex_pointmap_samples(
            hand_vertices_camera,
            hand_mask_bool,
            camera_k,
            image_shape=(pm_h, pm_w),
        )
        anchor_sample_method = "projected_vertices_fallback"
    if len(hand_hits) < config.min_hand_hits:
        hand_hits, hit_u, hit_v = projected_vertex_pointmap_samples(
            hand_vertices_camera,
            None,
            camera_k,
            image_shape=(pm_h, pm_w),
        )
        anchor_sample_method = "projected_vertices_no_mask_fallback"
    if len(hand_hits) < config.min_hand_hits:
        raise ValueError(f"too few hand anchor samples: {len(hand_hits)}.")

    hand_pm = pointmap[hit_v, hit_u]
    valid_hand_pm = np.all(np.isfinite(hand_pm), axis=1) & (np.abs(hand_pm[:, 2]) > 1e-8)
    if int(np.sum(valid_hand_pm)) < config.min_hand_hits:
        raise ValueError(f"too few valid hand pointmap samples: {int(np.sum(valid_hand_pm))}.")
    hand_hits = hand_hits[valid_hand_pm]
    hand_pm = hand_pm[valid_hand_pm]

    h_real = hand_hits.mean(axis=0)
    h_pm = hand_pm.mean(axis=0)
    o_pm = pointmap[object_valid].mean(axis=0)
    if abs(float(h_pm[2])) < 1e-8:
        raise ValueError("hand pointmap depth is near zero.")
    pointmap_scale = float(h_real[2] / h_pm[2])
    object_target = h_real + pointmap_scale * (o_pm - h_pm)

    object_pose_new = object_in_camera.copy()
    object_vertices_scaled = object_vertices_local * float(config.mesh_scale)
    rotation = object_in_camera[:3, :3]
    translation = object_in_camera[:3, 3]
    rotated_vertices = object_vertices_scaled @ rotation.T

    if config.translation_mode == "scale_along_foundationpose_translation":
        center_rot = _visible_rotated_mesh_center(
            rotated_vertices,
            translation,
            object_mask_bool,
            camera_k,
            image_shape=(pm_h, pm_w),
        )
        if center_rot is None:
            center_rot = rotated_vertices.mean(axis=0)
        scale = compute_optimal_translation_scale(center_rot, translation, object_target)
        translation_new = translation * scale
        mode_meta: dict[str, Any] = {
            "translation_scale_optimized": float(scale),
            "rotated_visible_center_m": center_rot.tolist(),
        }
    elif config.translation_mode == "direct":
        center_local = object_vertices_scaled.mean(axis=0)
        center_rot = rotation @ center_local
        translation_new = object_target - center_rot
        mode_meta = {
            "object_center_local_scaled_m": center_local.tolist(),
            "rotated_object_center_m": center_rot.tolist(),
        }
    else:
        raise ValueError(f"Unknown translation_mode: {config.translation_mode!r}.")

    object_pose_new[:3, 3] = translation_new
    before_center = rotated_vertices.mean(axis=0) + translation
    after_center = rotated_vertices.mean(axis=0) + translation_new
    metrics = {
        "hand_pointmap_sample_count": int(len(hand_hits)),
        "object_pointmap_pixel_count": int(np.sum(object_valid)),
        "pointmap_scale": float(pointmap_scale),
        "object_center_error_before_m": float(np.linalg.norm(before_center - object_target)),
        "object_center_error_after_m": float(np.linalg.norm(after_center - object_target)),
        "object_translation_delta_m": float(np.linalg.norm(translation_new - translation)),
    }
    return HandAnchoredPointmapAlignmentResult(
        status="ok",
        message="Optimized object pose with Do-as-I-Do hand-anchored pointmap alignment.",
        object_in_camera_optimized=object_pose_new,
        pointmap_scale=pointmap_scale,
        object_target_camera=object_target,
        hand_visible_center_camera=h_real,
        hand_visible_center_pointmap=h_pm,
        object_center_pointmap=o_pm,
        metrics=metrics,
        metadata={
            "translation_mode": config.translation_mode,
            "anchor_sample_method": anchor_sample_method,
            "mesh_scale": float(config.mesh_scale),
            "original_translation_m": translation.tolist(),
            "optimized_translation_m": translation_new.tolist(),
            **mode_meta,
        },
    )


def raycast_first_hits(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    camera_k: np.ndarray,
    *,
    max_rays: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    vertices = _as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=int)
    if max_rays and len(xs) > max_rays:
        rng = rng or np.random.default_rng(0)
        keep = rng.choice(len(xs), size=int(max_rays), replace=False)
        xs = xs[keep]
        ys = ys[keep]
    k = np.asarray(camera_k, dtype=np.float64)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    dirs = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)], axis=1)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    origins = np.zeros_like(dirs)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    locations, index_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    return locations, xs[index_ray], ys[index_ray]


def projected_vertex_pointmap_samples(
    vertices: np.ndarray,
    mask: np.ndarray | None,
    camera_k: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame hand vertices and use their pixels as pointmap anchors.

    This is a fallback for single-image GAP experiments where the generated hand
    mask is unavailable and the lollipop mask may not coincide with the recovered
    HaMeR silhouette. Do-as-I-Do uses hand-mask raycasting when a reliable hand
    mask exists; projected vertices keep the pipeline runnable while exposing
    the weaker anchor source in metadata.
    """

    vertices = _as_points(vertices)
    h, w = image_shape
    k = np.asarray(camera_k, dtype=np.float64)
    z = vertices[:, 2]
    valid = z > 1e-8
    u = np.round(k[0, 0] * vertices[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.round(k[1, 1] * vertices[:, 1] / z + k[1, 2]).astype(np.int64)
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if mask is not None:
        mask_bool = np.asarray(mask).astype(bool)
        in_mask = np.zeros(len(vertices), dtype=bool)
        idx = np.where(valid)[0]
        in_mask[idx] = mask_bool[v[idx], u[idx]]
        valid &= in_mask
    idx = np.where(valid)[0]
    return vertices[idx], u[idx], v[idx]


def compute_optimal_translation_scale(
    rotated_center: np.ndarray,
    translation: np.ndarray,
    target_3d: np.ndarray,
) -> float:
    rotated_center = np.asarray(rotated_center, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    target_3d = np.asarray(target_3d, dtype=np.float64)
    denom = float(np.dot(translation, translation))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(translation, target_3d - rotated_center) / denom)


def transform_object_vertices(
    vertices_local: np.ndarray,
    object_in_camera: np.ndarray,
    *,
    mesh_scale: float = 1.0,
) -> np.ndarray:
    vertices = _as_points(vertices_local) * float(mesh_scale)
    pose = np.asarray(object_in_camera, dtype=np.float64)
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def _visible_rotated_mesh_center(
    rotated_vertices: np.ndarray,
    translation: np.ndarray,
    object_mask: np.ndarray,
    camera_k: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    h, w = image_shape
    verts_cam = rotated_vertices + translation
    z = verts_cam[:, 2]
    valid_z = z > 1e-8
    if not np.any(valid_z):
        return None
    k = np.asarray(camera_k, dtype=np.float64)
    u = np.round(k[0, 0] * verts_cam[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.round(k[1, 1] * verts_cam[:, 1] / z + k[1, 2]).astype(np.int64)
    inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return None
    in_mask = np.zeros(len(verts_cam), dtype=bool)
    idx = np.where(inside)[0]
    in_mask[idx] = object_mask[v[idx], u[idx]]
    if int(np.sum(in_mask)) < 10:
        return None
    return rotated_vertices[in_mask].mean(axis=0)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape[:2] != (height, width):
        image = Image.fromarray(array.astype(np.uint8))
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
        array = np.asarray(image)
    return array > 0


def _as_points(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected point array with shape (N, 3), got {array.shape}.")
    return array
