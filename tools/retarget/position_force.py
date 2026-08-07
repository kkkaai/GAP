#!/usr/bin/env python3
"""Run position-initialized force-closure retargeting for one HOI4D frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = REPO_ROOT / "tools" / "vis"
RETARGET_ROOT = REPO_ROOT
RETARGET_SRC = REPO_ROOT / "src"

for path in (REPO_ROOT, VIS_ROOT, RETARGET_SRC, REPO_ROOT / "tools" / "retarget"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contact_surface import (  # noqa: E402
    ROBOT_PROFILES,
    bounds_for_points,
    load_hoi4d_scene,
    load_robot_model,
)
from position import (  # noqa: E402
    ROBOT_MESH_MAX_FACES,
    as_jsonable,
    marker_trace,
    mesh_trace,
    point_trace,
    prepare_shared_case,
    run_position_retarget,
    segment_trace,
    simplify_mesh_for_html,
    style_scene_traces,
)
from scene_html import build_figure, write_html  # noqa: E402
from scene_png import (  # noqa: E402
    DEFAULT_CAD,
    DEFAULT_FRAME,
    DEFAULT_HAND,
    DEFAULT_MANO_ROOT,
    DEFAULT_SEQUENCE,
)

from prosthetic_grasp.geometry import (  # noqa: E402
    ContactPatchTarget,
    RetargetOptimizationConfig,
    materialize_assigned_robot_contacts,
    optimize_retarget_action,
    rank_candidates_by_strict_force_closure,
    retarget_loss_terms,
    robot_tip_from_link,
)
from prosthetic_grasp.geometry.contact_clustering import closest_points_on_mesh  # noqa: E402
from prosthetic_grasp.geometry.contact_retargeting import _patch_array  # noqa: E402
from utils.force_closure.dexgraspnet_fc import dexgraspnet_force_closure_energy  # noqa: E402
from utils.force_closure.strict_fc import evaluate_force_closure  # noqa: E402


DEFAULT_RGB = DEFAULT_SEQUENCE / "align_rgb" / "00074.jpg"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "runs"


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


class PointSetRigidTransform:
    """Rigid scene/robot transform fitted from corresponding points without scale."""

    def __init__(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
        source = np.asarray(source_points, dtype=np.float64)
        target = np.asarray(target_points, dtype=np.float64)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"source/target must both have shape (N, 3), got {source.shape} and {target.shape}.")
        self.source_center = source.mean(axis=0)
        self.target_center = target.mean(axis=0)
        source_centered = source - self.source_center
        target_centered = target - self.target_center
        u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        self.rotation = rotation

    def points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (points - self.source_center) @ self.rotation + self.target_center

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (points - self.target_center) @ self.rotation.T + self.source_center

    def robot_to_scene_pose(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self.rotation
        translation = self.source_center - self.target_center @ rotation.T
        return rotation, translation


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    )


def transform_normals(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def pose_from_params(params: np.ndarray, init_rotation: np.ndarray, init_translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    params = np.asarray(params, dtype=np.float64)
    delta_rotation = Rotation.from_rotvec(params[:3]).as_matrix()
    rotation = delta_rotation @ init_rotation
    translation = init_translation + params[3:6]
    return rotation, translation


def _normalize_weights_from_distances(distances_sq: np.ndarray, temperature: float) -> np.ndarray:
    logits = -distances_sq / max(float(temperature), 1e-12)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / max(float(weights.sum()), 1e-300)


def make_fingertip_projection_patches(
    robot_surface: Any,
    zero_surface: Any,
    object_vertices_robot: np.ndarray,
    object_faces: np.ndarray,
    fingertip_links: list[str],
    *,
    top_k: int,
    temperature: float,
) -> tuple[list[ContactPatchTarget], np.ndarray, np.ndarray, np.ndarray]:
    """Project initial robot fingertips to the object and bind patches to fingertip links."""

    surface_points = np.asarray(robot_surface.points, dtype=np.float64)
    surface_links = np.asarray(robot_surface.link_names).astype(str)
    zero_points = np.asarray(zero_surface.points, dtype=np.float64)
    zero_links = np.asarray(zero_surface.link_names).astype(str)

    tip_points = np.asarray(
        [robot_tip_from_link(surface_points, surface_links, link_name) for link_name in fingertip_links],
        dtype=np.float64,
    )
    object_points, object_normals, projection_distances = closest_points_on_mesh(
        object_vertices_robot,
        object_faces,
        tip_points,
    )

    patches: list[ContactPatchTarget] = []
    for index, (tip_link, tip_point, object_point, object_normal, distance) in enumerate(
        zip(fingertip_links, tip_points, object_points, object_normals, projection_distances)
    ):
        zero_tip = robot_tip_from_link(zero_points, zero_links, tip_link)
        link_indices = np.nonzero(zero_links == tip_link)[0].astype(np.int64)
        if len(link_indices) == 0:
            raise RuntimeError(f"No sampled surface points found for fingertip link {tip_link!r}.")
        link_dist_sq = np.sum((zero_points[link_indices] - zero_tip) ** 2, axis=1)
        k = min(int(top_k), len(link_indices))
        nearest_local = np.argpartition(link_dist_sq, kth=k - 1)[:k]
        nearest_local = nearest_local[np.argsort(link_dist_sq[nearest_local])]
        selected_indices = link_indices[nearest_local].astype(np.int64)
        selected_weights = _normalize_weights_from_distances(link_dist_sq[nearest_local], temperature)

        patch = ContactPatchTarget(
            cluster_label=index,
            patch_size=1,
            sample_indices=np.asarray([index], dtype=np.int64),
            representative_sample_index=index,
            mano_face_index=-1,
            mano_barycentric=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            mano_point_posed=tip_point.astype(np.float64),
            object_point_target=object_point.astype(np.float64),
            object_normal_target=object_normal.astype(np.float64),
            object_point_nearest=object_point.astype(np.float64),
            object_normal_nearest=object_normal.astype(np.float64),
            object_point_center=object_point.astype(np.float64),
            object_normal_center=object_normal.astype(np.float64),
            contact_distance=float(distance),
            mano_point_canonical=zero_tip.astype(np.float64),
            canonical_robot_target=zero_tip.astype(np.float64),
            robot_assignment_indices=selected_indices,
            robot_assignment_weights=selected_weights.astype(np.float64),
            robot_assignment_link_names=zero_links[selected_indices].astype(object),
        )
        patches.append(patch)

    return patches, tip_points, object_points, projection_distances


def _fc_score_to_dict(score: Any | None) -> dict[str, Any] | None:
    if score is None:
        return None
    fc = score.force_closure
    return {
        "candidate_index": int(score.candidate_index),
        "is_force_closure": bool(fc.is_force_closure),
        "epsilon": float(fc.epsilon),
        "origin_margin": float(fc.origin_margin),
        "rank": int(fc.rank),
        "num_wrenches": int(fc.num_wrenches),
        "dex_fc_energy": float(score.dex_fc_energy),
        "mean_projection_distance_m": float(score.mean_projection_distance),
        "projection_distances_m": np.asarray(score.projection_distances, dtype=np.float64).tolist(),
    }


def _robot_mesh_for_action(profile: dict[str, Any], action: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    from position import load_retargeted_robot_mesh  # noqa: E402

    return load_retargeted_robot_mesh(
        SimpleNamespace(
            action=action,
            metadata={
                "robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")),
                "robot_urdf_path": str(profile.get("urdf_path") or profile.get("xml_path")),
                "model_format": profile["model_format"],
                "wrist_link": profile["wrist_link"],
            },
        )
    )


def _assigned_contact_points(
    patches: list[ContactPatchTarget],
    surface: Any,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals = materialize_assigned_robot_contacts(patches, surface)
    return points.astype(np.float64), normals.astype(np.float64)


def make_object_closest_query(
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
) -> Any:
    """Build a reusable closest-point query for one object mesh."""

    vertices = np.asarray(object_vertices, dtype=np.float64)
    faces = np.asarray(object_faces, dtype=np.int64)
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

        def query(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            closest, distances, face_indices = trimesh.proximity.closest_point(
                mesh,
                np.asarray(points, dtype=np.float64),
            )
            normals = face_normals[np.asarray(face_indices, dtype=np.int64)]
            return (
                np.asarray(closest, dtype=np.float64),
                normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12),
                np.asarray(distances, dtype=np.float64),
            )

        return query
    except Exception:
        from scipy.spatial import cKDTree

        tree = cKDTree(vertices)
        face_vertices = vertices[faces]
        face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
        vertex_faces: list[list[int]] = [[] for _ in range(len(vertices))]
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                vertex_faces[int(vertex_index)].append(face_index)

        def query(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            distances, indices = tree.query(np.asarray(points, dtype=np.float64))
            normals = []
            for vertex_index in np.asarray(indices, dtype=np.int64):
                incident = vertex_faces[int(vertex_index)]
                if incident:
                    normal = face_normals[incident].mean(axis=0)
                    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                normals.append(normal)
            return vertices[indices], np.asarray(normals, dtype=np.float64), np.asarray(distances, dtype=np.float64)

        return query


def optimize_pose_and_action(
    robot_model: Any,
    robot_topology: Any,
    patches: list[ContactPatchTarget],
    object_vertices_scene: np.ndarray,
    object_faces: np.ndarray,
    mano_fingertips_scene: np.ndarray,
    fingertip_links: list[str],
    initial_action: np.ndarray,
    init_rotation: np.ndarray,
    init_translation: np.ndarray,
    *,
    fingertip_weight: float,
    contact_weight: float,
    surface_attraction_weight: float,
    normal_weight: float,
    fc_weight: float,
    penetration_weight: float,
    pose_regularization_weight: float,
    joint_regularization_weight: float,
    maxiter: int,
    friction_coef: float,
    num_friction_edges: int,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    lower, upper = robot_model.joint_bounds()
    initial_action = np.clip(np.asarray(initial_action, dtype=np.float64), lower, upper)
    pose0 = np.zeros(6, dtype=np.float64)
    x0 = np.concatenate([pose0, initial_action])
    bounds = [(-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35), (-0.08, 0.08), (-0.08, 0.08), (-0.08, 0.08)]
    bounds.extend(list(zip(lower, upper)))

    object_center = np.asarray(object_vertices_scene, dtype=np.float64).mean(axis=0)
    object_radius = float(np.max(np.linalg.norm(object_vertices_scene - object_center, axis=1)))
    closest_query = make_object_closest_query(object_vertices_scene, object_faces)
    target_contacts = _patch_array(patches, "object_point_target")
    target_normals = _patch_array(patches, "object_normal_target")
    joint_span = np.maximum(upper - lower, 1e-8)

    def evaluate(x: np.ndarray, include_terms: bool = False) -> float | dict[str, float]:
        rotation, translation = pose_from_params(x[:6], init_rotation, init_translation)
        action = np.asarray(x[6:], dtype=np.float64)
        surface = robot_model.materialize_surface(robot_topology, action)
        wrist_tips = np.asarray(
            [
                robot_tip_from_link(surface.points, surface.link_names.astype(str), link_name)
                for link_name in fingertip_links
            ],
            dtype=np.float64,
        )
        tips_scene = transform_points(wrist_tips, rotation, translation)
        contacts_wrist, normals_wrist = _assigned_contact_points(patches, surface)
        contacts_scene = transform_points(contacts_wrist, rotation, translation)
        normals_scene = transform_normals(normals_wrist, rotation)
        projected_contacts, projected_normals, surface_distances = closest_query(contacts_scene)

        fingertip_loss = float(np.mean(np.sum((tips_scene - mano_fingertips_scene) ** 2, axis=1)))
        contact_loss = float(np.mean(np.sum((contacts_scene - target_contacts) ** 2, axis=1)))
        surface_loss = float(np.mean(surface_distances**2))
        signed_surface_offset = np.sum((contacts_scene - projected_contacts) * projected_normals, axis=1)
        penetration_loss = float(np.mean(np.maximum(-signed_surface_offset, 0.0) ** 2))
        normal_dot = np.sum(normals_scene * projected_normals, axis=1)
        normal_loss = float(np.mean((normal_dot + 1.0) ** 2))
        pose_loss = float(np.mean((x[:3] / 0.35) ** 2) + np.mean((x[3:6] / 0.08) ** 2))
        joint_loss = float(np.mean(((action - initial_action) / joint_span) ** 2))
        fc_loss = 0.0
        if fc_weight > 0.0:
            fc_loss = float(
                dexgraspnet_force_closure_energy(
                    projected_contacts,
                    -projected_normals,
                    object_center=object_center,
                    torque_scale=1.0 / max(object_radius, 1e-6),
                    reduction="mean",
                )
            )
        total = (
            fingertip_weight * fingertip_loss
            + contact_weight * contact_loss
            + surface_attraction_weight * surface_loss
            + normal_weight * normal_loss
            + fc_weight * fc_loss
            + penetration_weight * penetration_loss
            + pose_regularization_weight * pose_loss
            + joint_regularization_weight * joint_loss
        )
        if include_terms:
            return {
                "fingertip": fingertip_loss,
                "contact": contact_loss,
                "surface_attraction": surface_loss,
                "normal": normal_loss,
                "dex_fc": fc_loss,
                "penetration": penetration_loss,
                "pose_regularization": pose_loss,
                "joint_regularization": joint_loss,
                "total": float(total),
            }
        return float(total)

    start = time.perf_counter()
    initial_terms = evaluate(x0, include_terms=True)
    result = minimize(
        lambda values: float(evaluate(values)),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    elapsed = time.perf_counter() - start
    x_best = np.asarray(result.x, dtype=np.float64)
    best_terms = evaluate(x_best, include_terms=True)
    rotation, translation = pose_from_params(x_best[:6], init_rotation, init_translation)
    action = x_best[6:]
    surface = robot_model.materialize_surface(robot_topology, action)
    contacts_wrist, normals_wrist = _assigned_contact_points(patches, surface)
    contacts_scene = transform_points(contacts_wrist, rotation, translation)
    normals_scene = transform_normals(normals_wrist, rotation)
    projected_points, projected_normals, projection_distances = closest_query(contacts_scene)
    fc = evaluate_force_closure(
        projected_points,
        projected_normals,
        friction_coef=friction_coef,
        num_edges=num_friction_edges,
        object_center=object_vertices_scene.mean(axis=0),
        torque_scale=1.0 / max(float(np.max(np.linalg.norm(object_vertices_scene - object_vertices_scene.mean(axis=0), axis=1))), 1e-6),
    )
    dex_score = float(
        dexgraspnet_force_closure_energy(
            projected_points,
            -projected_normals,
            object_center=object_vertices_scene.mean(axis=0),
            torque_scale=1.0 / max(float(np.max(np.linalg.norm(object_vertices_scene - object_vertices_scene.mean(axis=0), axis=1))), 1e-6),
            reduction="mean",
        )
    )
    return {
        "action": action,
        "pose_params": x_best[:6],
        "rotation": rotation,
        "translation": translation,
        "initial_loss_terms": initial_terms,
        "best_loss_terms": best_terms,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "elapsed_seconds": float(elapsed),
        "contacts_scene": contacts_scene,
        "contact_normals_scene": normals_scene,
        "projected_contact_points": projected_points,
        "projected_contact_normals": projected_normals,
        "projection_distances": projection_distances,
        "force_closure": fc,
        "dex_fc_energy": dex_score,
    }


def run_position_force(
    sequence: Path,
    frame: str,
    cad: Path,
    hand_path: Path,
    mano_root: Path,
    robot_profile: str,
    *,
    position_restarts: int,
    position_max_nfev: int,
    num_robot_samples: int,
    contact_top_k: int,
    contact_temperature: float,
    stage1_maxiter: int,
    stage2_maxiter: int,
    contact_weight: float,
    surface_attraction_weight: float,
    normal_weight: float,
    fc_weight: float,
    penetration_weight: float,
    regularization_weight: float,
    random_restarts: int,
    random_start_scale: float,
    friction_coef: float,
    num_friction_edges: int,
    select_candidate: str,
    seed: int,
) -> dict[str, Any]:
    if robot_profile not in ROBOT_PROFILES:
        raise ValueError(f"Unsupported robot_profile {robot_profile!r}.")

    position_result, position_seconds = run_position_retarget(
        hand_path,
        mano_root,
        robot_profile,
        optimization_restarts=position_restarts,
        max_nfev=position_max_nfev,
    )
    if getattr(position_result, "status", "") != "ok":
        raise RuntimeError(f"Position initialization failed: {position_result.status} {position_result.message}")

    hand, mano, object_vertices_scene, object_faces = load_hoi4d_scene(sequence, frame, cad, hand_path, mano_root)
    del hand, mano
    profile = ROBOT_PROFILES[robot_profile]
    robot_model = load_robot_model(robot_profile)
    robot_topology = robot_model.sample_surface_topology(
        num_points=num_robot_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    zero_surface = robot_model.materialize_surface(robot_topology, robot_model.zero_action)
    scene_to_robot = PointSetRigidTransform(
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        np.asarray(position_result.target_fingertips_wrist, dtype=np.float64),
    )
    init_rotation, init_translation = scene_to_robot.robot_to_scene_pose()

    initial_action = np.asarray(position_result.action, dtype=np.float64)
    initial_surface = robot_model.materialize_surface(robot_topology, initial_action)
    initial_surface_scene = SimpleNamespace(
        points=transform_points(initial_surface.points, init_rotation, init_translation),
        normals=transform_normals(initial_surface.normals, init_rotation),
        local_points=initial_surface.local_points,
        local_normals=initial_surface.local_normals,
        link_names=initial_surface.link_names,
    )
    patches, initial_tip_points, object_contacts, initial_projection_distances = make_fingertip_projection_patches(
        initial_surface_scene,
        zero_surface,
        object_vertices_scene,
        object_faces,
        profile["fingertip_links"],
        top_k=contact_top_k,
        temperature=contact_temperature,
    )

    del random_restarts, random_start_scale, select_candidate
    result = optimize_pose_and_action(
        robot_model,
        robot_topology,
        patches,
        object_vertices_scene,
        object_faces,
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        profile["fingertip_links"],
        initial_action,
        init_rotation,
        init_translation,
        fingertip_weight=25.0,
        contact_weight=contact_weight,
        surface_attraction_weight=surface_attraction_weight,
        normal_weight=normal_weight,
        fc_weight=fc_weight,
        penetration_weight=penetration_weight,
        pose_regularization_weight=regularization_weight,
        joint_regularization_weight=regularization_weight,
        maxiter=stage2_maxiter,
        friction_coef=friction_coef,
        num_friction_edges=num_friction_edges,
    )
    refine_seconds = float(result["elapsed_seconds"])
    best_action = np.asarray(result["action"], dtype=np.float64)
    best_rotation = np.asarray(result["rotation"], dtype=np.float64)
    best_translation = np.asarray(result["translation"], dtype=np.float64)
    best_surface = robot_model.materialize_surface(robot_topology, best_action)
    all_robot_contacts, all_robot_normals = materialize_assigned_robot_contacts(patches, best_surface)
    robot_contacts = all_robot_contacts[: len(profile["fingertip_links"])]
    robot_normals = all_robot_normals[: len(profile["fingertip_links"])]
    object_contact_targets = _patch_array(patches[: len(profile["fingertip_links"])], "object_point_target")
    object_contact_normals = _patch_array(patches[: len(profile["fingertip_links"])], "object_normal_target")
    robot_contacts_scene = transform_points(robot_contacts, best_rotation, best_translation)
    robot_normals_scene = transform_normals(robot_normals, best_rotation)
    contact_errors = np.linalg.norm(robot_contacts_scene - object_contact_targets, axis=1)

    robot_fingertips = np.asarray(
        [
            robot_tip_from_link(best_surface.points, best_surface.link_names.astype(str), link_name)
            for link_name in profile["fingertip_links"]
        ],
        dtype=np.float64,
    )
    mano_fingertips_scene = np.asarray(position_result.mano_fingertips, dtype=np.float64)
    robot_fingertips_scene = transform_points(robot_fingertips, best_rotation, best_translation)
    fingertip_errors = np.linalg.norm(robot_fingertips_scene - mano_fingertips_scene, axis=1)

    robot_surface_scene = transform_points(best_surface.points, best_rotation, best_translation)
    object_contacts_scene = object_contact_targets
    robot_wrist_scene = transform_points(np.zeros((1, 3), dtype=np.float64), best_rotation, best_translation)[0]

    robot_mesh_vertices_scene: list[list[float]] = []
    robot_mesh_faces: list[list[int]] = []
    robot_mesh = _robot_mesh_for_action(profile, best_action)
    if robot_mesh is not None:
        mesh_vertices, mesh_faces = simplify_mesh_for_html(*robot_mesh, max_faces=ROBOT_MESH_MAX_FACES)
        robot_mesh_vertices_scene = transform_points(mesh_vertices, best_rotation, best_translation).astype(float).tolist()
        robot_mesh_faces = mesh_faces.astype(int).tolist()

    return {
        "status": "ok",
        "route": "position_force_closure",
        "robot_profile": robot_profile,
        "action_names": list(robot_model.action_names),
        "action": best_action.astype(float).tolist(),
        "position_initialization": {
            "optimization_seconds": float(position_seconds),
            "status": position_result.status,
            "message": position_result.message,
            "mean_fingertip_error_m": float(np.mean(position_result.fingertip_error))
            if position_result.fingertip_error is not None
            else None,
            "max_fingertip_error_m": float(np.max(position_result.fingertip_error))
            if position_result.fingertip_error is not None
            else None,
            "result": as_jsonable(position_result),
        },
        "force_refinement": {
            "optimization_seconds": float(refine_seconds),
            "stage1_maxiter": int(stage1_maxiter),
            "stage2_maxiter": int(stage2_maxiter),
            "success": bool(result["success"]),
            "message": result["message"],
            "iterations": int(result["iterations"]),
            "selected_by": "joint_pose_loss",
            "initial_loss_terms": {key: float(value) for key, value in result["initial_loss_terms"].items()},
            "best_loss_terms": {key: float(value) for key, value in result["best_loss_terms"].items()},
        },
        "force_closure": {
            "is_force_closure": bool(result["force_closure"].is_force_closure),
            "epsilon": float(result["force_closure"].epsilon),
            "origin_margin": float(result["force_closure"].origin_margin),
            "rank": int(result["force_closure"].rank),
            "num_wrenches": int(result["force_closure"].num_wrenches),
            "dex_fc_energy": float(result["dex_fc_energy"]),
            "mean_projection_distance_m": float(np.mean(result["projection_distances"])),
            "projection_distances_m": np.asarray(result["projection_distances"], dtype=np.float64).tolist(),
        },
        "config": {
            "optimizer": "joint_wrist_pose_and_action",
            "fingertip_weight": 25.0,
            "contact_weight": float(contact_weight),
            "surface_attraction_weight": float(surface_attraction_weight),
            "normal_weight": float(normal_weight),
            "fc_weight": float(fc_weight),
            "penetration_weight": float(penetration_weight),
            "pose_regularization_weight": float(regularization_weight),
            "joint_regularization_weight": float(regularization_weight),
            "maxiter": int(stage2_maxiter),
        },
        "num_robot_samples": int(num_robot_samples),
        "contact_top_k": int(contact_top_k),
        "contact_temperature": float(contact_temperature),
        "initial_projection_distance": {
            "rule": "nearest_point_on_object_surface_from_initial_robot_fingertips",
            "mean_m": float(np.mean(initial_projection_distances)),
            "max_m": float(np.max(initial_projection_distances)),
            "per_finger_m": np.asarray(initial_projection_distances, dtype=np.float64).tolist(),
        },
        "contact_error": {
            "mean_m": float(np.mean(contact_errors)),
            "max_m": float(np.max(contact_errors)),
            "per_finger_m": contact_errors.astype(float).tolist(),
        },
        "fingertip_error": {
            "mean_m": float(np.mean(fingertip_errors)),
            "max_m": float(np.max(fingertip_errors)),
            "per_finger_m": fingertip_errors.astype(float).tolist(),
        },
        "arrays": {
            "object_vertices_scene": object_vertices_scene.astype(float).tolist(),
            "object_faces": object_faces.astype(int).tolist(),
            "robot_surface_points_scene": robot_surface_scene.astype(float).tolist(),
            "mano_fingertips_scene": mano_fingertips_scene.astype(float).tolist(),
            "robot_fingertips_scene": robot_fingertips_scene.astype(float).tolist(),
            "robot_contacts_scene": robot_contacts_scene.astype(float).tolist(),
            "object_contacts_scene": object_contacts_scene.astype(float).tolist(),
            "robot_wrist_scene": robot_wrist_scene.astype(float).tolist(),
            "robot_mesh_vertices_scene": robot_mesh_vertices_scene,
            "robot_mesh_faces": robot_mesh_faces,
            "object_contacts_robot": object_contact_targets.astype(float).tolist(),
            "object_contact_normals_robot": object_contact_normals.astype(float).tolist(),
            "robot_contacts_robot": robot_contacts.astype(float).tolist(),
            "robot_contact_normals_robot": robot_normals_scene.astype(float).tolist(),
            "initial_tip_points_robot": initial_tip_points.astype(float).tolist(),
            "initial_object_contacts_robot": object_contacts.astype(float).tolist(),
        },
        "metadata": {
            "model_format": profile["model_format"],
            "robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")),
            "wrist_link": profile["wrist_link"],
            "visual_alignment": "optimized_wrist_pose_rigid_only_no_scale",
            "scene_to_robot_frame": "fitted_from_mano_fingertips_to_position_only_target_fingertips",
            "wrist_constraint": "not_used_in_route2_scene_alignment_or_loss",
            "wrist_pose_optimization": "enabled",
            "candidate_selection": "joint_wrist_pose_loss",
            "projection_rule": "nearest_point_on_object_surface",
            "robot_mesh_max_faces": int(ROBOT_MESH_MAX_FACES),
        },
    }


def write_position_force_html(case_dir: Path, rgb: Path, base_meta: dict[str, Any], payload: dict[str, Any]) -> Path:
    fig, scene_meta = build_figure(
        Path(base_meta["sequence"]),
        str(base_meta["frame"]),
        Path(base_meta["cad"]),
        Path(base_meta["hand_pickle"]),
        Path(base_meta["mano_root"]),
    )
    style_scene_traces(fig)
    arrays = payload["arrays"]
    robot_points_scene = np.asarray(arrays["robot_surface_points_scene"], dtype=np.float64)
    mano_fingertips_scene = np.asarray(arrays["mano_fingertips_scene"], dtype=np.float64)
    robot_fingertips_scene = np.asarray(arrays["robot_fingertips_scene"], dtype=np.float64)
    robot_contacts_scene = np.asarray(arrays["robot_contacts_scene"], dtype=np.float64)
    object_contacts_scene = np.asarray(arrays["object_contacts_scene"], dtype=np.float64)
    robot_wrist_scene = np.asarray(arrays["robot_wrist_scene"], dtype=np.float64)
    robot_mesh_vertices = np.asarray(arrays.get("robot_mesh_vertices_scene") or [], dtype=np.float64)
    robot_mesh_faces = np.asarray(arrays.get("robot_mesh_faces") or [], dtype=np.int64)

    if robot_mesh_vertices.size and robot_mesh_faces.size:
        fig.add_trace(mesh_trace(f"{payload['robot_profile']} mesh", robot_mesh_vertices, robot_mesh_faces, "#2563eb", 0.30))
        vertex_step = max(len(robot_mesh_vertices) // 2500, 1)
        fig.add_trace(marker_trace(f"{payload['robot_profile']} mesh vertices", robot_mesh_vertices[::vertex_step], "#1d4ed8", 2))
    fig.add_trace(marker_trace(f"{payload['robot_profile']} surface samples", robot_points_scene, "#1d4ed8", 3))
    fig.add_trace(marker_trace("MANO fingertips", mano_fingertips_scene, "#16a34a", 7))
    fig.add_trace(marker_trace(f"{payload['robot_profile']} fingertips", robot_fingertips_scene, "#0f4cbd", 7))
    fig.add_trace(marker_trace("object projected contacts", object_contacts_scene, "#d9480f", 7))
    fig.add_trace(marker_trace(f"{payload['robot_profile']} contact points", robot_contacts_scene, "#1864ab", 7))
    fig.add_trace(segment_trace("fingertip error", mano_fingertips_scene, robot_fingertips_scene, "#adb5bd"))
    fig.add_trace(segment_trace("contact error", object_contacts_scene, robot_contacts_scene, "#495057"))
    fig.add_trace(point_trace("robot wrist", robot_wrist_scene, "#1d4ed8", 8, "diamond"))

    view_points = [
        robot_points_scene,
        mano_fingertips_scene,
        robot_fingertips_scene,
        robot_contacts_scene,
        object_contacts_scene,
        robot_wrist_scene.reshape(1, 3),
    ]
    if robot_mesh_vertices.size:
        view_points.append(robot_mesh_vertices[:: max(len(robot_mesh_vertices) // 5000, 1)])
    x_range, y_range, z_range = bounds_for_points(np.vstack(view_points), margin=0.18)
    fig.update_layout(
        title=f"HOI4D grasp + position force retargeting ({payload['robot_profile']})",
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(range=x_range, visible=False),
            yaxis=dict(range=y_range, visible=False),
            zaxis=dict(range=z_range, visible=False),
            aspectmode="cube",
        ),
    )

    fc = payload.get("force_closure") or {}
    errors = {
        "route": payload["route"],
        "robot hand": payload["robot_profile"],
        "visual alignment": "rigid only, no scale",
        "position init time": f"{payload['position_initialization']['optimization_seconds']:.3f} s",
        "force refine time": f"{payload['force_refinement']['optimization_seconds']:.3f} s",
        "mean fingertip error": f"{payload['fingertip_error']['mean_m'] * 1000.0:.2f} mm",
        "max fingertip error": f"{payload['fingertip_error']['max_m'] * 1000.0:.2f} mm",
        "mean contact error": f"{payload['contact_error']['mean_m'] * 1000.0:.2f} mm",
        "max contact error": f"{payload['contact_error']['max_m'] * 1000.0:.2f} mm",
        "strict force closure": str(fc.get("is_force_closure", "n/a")),
        "FC epsilon": f"{float(fc.get('epsilon', 0.0)):.6f}" if fc else "n/a",
        "Dex FC energy": f"{float(fc.get('dex_fc_energy', 0.0)):.6f}" if fc else "n/a",
        "status": payload["status"],
    }
    scene_meta.update(base_meta)
    scene_meta["errors"] = errors
    scene_meta["position_force"] = {
        key: payload[key]
        for key in (
            "route",
            "robot_profile",
            "position_initialization",
            "force_refinement",
            "force_closure",
            "config",
            "contact_error",
            "fingertip_error",
            "metadata",
        )
    }
    html = case_dir / f"scene_{payload['robot_profile']}_position_force.html"
    write_html(fig, scene_meta, rgb, html)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", default="bottle_frame74_position_force")
    parser.add_argument("--robot-profile", choices=sorted(ROBOT_PROFILES), default="shadow_hand")
    parser.add_argument("--position-restarts", type=int, default=4)
    parser.add_argument("--position-max-nfev", type=int, default=120)
    parser.add_argument("--num-robot-samples", type=int, default=1800)
    parser.add_argument("--contact-top-k", type=int, default=48)
    parser.add_argument("--contact-temperature", type=float, default=1e-4)
    parser.add_argument("--stage1-maxiter", type=int, default=50)
    parser.add_argument("--stage2-maxiter", type=int, default=160)
    parser.add_argument("--contact-weight", type=float, default=10.0)
    parser.add_argument("--surface-attraction-weight", type=float, default=80.0)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--fc-weight", type=float, default=0.05)
    parser.add_argument("--penetration-weight", type=float, default=120.0)
    parser.add_argument("--regularization-weight", type=float, default=0.005)
    parser.add_argument("--random-restarts", type=int, default=2)
    parser.add_argument("--random-start-scale", type=float, default=0.08)
    parser.add_argument("--friction-coef", type=float, default=0.7)
    parser.add_argument("--num-friction-edges", type=int, default=8)
    parser.add_argument("--select-candidate", choices=["loss", "force_closure"], default="loss")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb = args.rgb or args.sequence / "align_rgb" / f"{int(args.frame):05d}.jpg"
    case_dir = prepare_shared_case(
        args.sequence,
        str(args.frame),
        rgb,
        args.cad,
        args.hand,
        args.mano_root,
        args.output_root,
        args.case_id,
        args.overwrite,
    )
    payload = run_position_force(
        args.sequence,
        str(args.frame),
        args.cad,
        args.hand,
        args.mano_root,
        args.robot_profile,
        position_restarts=args.position_restarts,
        position_max_nfev=args.position_max_nfev,
        num_robot_samples=args.num_robot_samples,
        contact_top_k=args.contact_top_k,
        contact_temperature=args.contact_temperature,
        stage1_maxiter=args.stage1_maxiter,
        stage2_maxiter=args.stage2_maxiter,
        contact_weight=args.contact_weight,
        surface_attraction_weight=args.surface_attraction_weight,
        normal_weight=args.normal_weight,
        fc_weight=args.fc_weight,
        penetration_weight=args.penetration_weight,
        regularization_weight=args.regularization_weight,
        random_restarts=args.random_restarts,
        random_start_scale=args.random_start_scale,
        friction_coef=args.friction_coef,
        num_friction_edges=args.num_friction_edges,
        select_candidate=args.select_candidate,
        seed=args.seed,
    )
    base_meta = {
        "case_id": args.case_id,
        "sequence": str(args.sequence),
        "frame": str(args.frame),
        "rgb": str(rgb),
        "cad": str(args.cad),
        "hand_pickle": str(args.hand),
        "mano_root": str(args.mano_root),
        "robot_profile": args.robot_profile,
        "route": "position_force_closure",
    }
    json_path = case_dir / f"retargeted_{args.robot_profile}_position_force.json"
    payload_out = {**base_meta, **{k: v for k, v in payload.items() if k != "arrays"}}
    json_path.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_position_force_html(case_dir, case_dir / "original.jpg", base_meta, payload)
    print(f"Wrote {json_path}")
    print(f"Open: {html_path}")


if __name__ == "__main__":
    main()
