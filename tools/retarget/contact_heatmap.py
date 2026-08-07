#!/usr/bin/env python3
"""Run object-centric contact heatmap retargeting for one HOI4D frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = REPO_ROOT / "tools" / "vis"
RETARGET_ROOT = REPO_ROOT
RETARGET_SRC = REPO_ROOT / "src"

for path in (REPO_ROOT, VIS_ROOT, RETARGET_SRC, REPO_ROOT / "tools" / "retarget"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contact_surface import ROBOT_PROFILES, bounds_for_points, load_hoi4d_scene, load_robot_model  # noqa: E402
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
from position_force import (  # noqa: E402
    PointSetRigidTransform,
    pose_from_params,
    transform_normals,
    transform_points,
)
from scene_html import build_figure, write_html  # noqa: E402
from scene_png import DEFAULT_CAD, DEFAULT_FRAME, DEFAULT_HAND, DEFAULT_MANO_ROOT, DEFAULT_SEQUENCE  # noqa: E402

from prosthetic_grasp.geometry import robot_tip_from_link  # noqa: E402
from prosthetic_grasp.geometry.surface_sampling import sample_mesh_surface  # noqa: E402
from utils.force_closure.dexgraspnet_fc import dexgraspnet_force_closure_energy  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "runs"
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def aligned_distance(
    object_points: np.ndarray,
    object_normals: np.ndarray,
    hand_points: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    object_points = np.asarray(object_points, dtype=np.float64)
    object_normals = normalize_rows(object_normals)
    hand_points = np.asarray(hand_points, dtype=np.float64)
    distances, indices = cKDTree(hand_points).query(object_points)
    nearest = hand_points[np.asarray(indices, dtype=np.int64)]
    direction = normalize_rows(nearest - object_points)
    alignment = np.sum(direction * object_normals, axis=1)
    weighted = np.asarray(distances, dtype=np.float64) * np.exp(float(gamma) * (1.0 - alignment))
    return weighted, nearest


def heatmap_from_distance(distances: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-((np.asarray(distances, dtype=np.float64) / sigma) ** 2))


def heatmap_trace(name: str, points: np.ndarray, values: np.ndarray, colorscale: Any, size: float = 2.0) -> go.Scatter3d:
    return go.Scatter3d(
        name=name,
        x=points[:, 0].tolist(),
        y=points[:, 1].tolist(),
        z=points[:, 2].tolist(),
        mode="markers",
        marker=dict(
            size=size,
            color=np.asarray(values, dtype=np.float64).tolist(),
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            opacity=0.95,
            colorbar=dict(title=name[:18]),
        ),
        hovertemplate=f"{name}<br>value=%{{marker.color:.3f}}<extra></extra>",
    )


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


def optimize_heatmap_pose_and_action(
    robot_model: Any,
    robot_topology: Any,
    object_points: np.ndarray,
    object_normals: np.ndarray,
    target_heatmap: np.ndarray,
    mano_fingertips_scene: np.ndarray,
    fingertip_links: list[str],
    initial_action: np.ndarray,
    init_rotation: np.ndarray,
    init_translation: np.ndarray,
    *,
    heatmap_sigma: float,
    alignment_gamma: float,
    heatmap_weight: float,
    high_contact_weight: float,
    surface_attraction_weight: float,
    penetration_weight: float,
    fingertip_prior_weight: float,
    fc_weight: float,
    regularization_weight: float,
    optimizer: str,
    maxiter: int,
) -> dict[str, Any]:
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    lower, upper = robot_model.joint_bounds()
    initial_action = np.clip(np.asarray(initial_action, dtype=np.float64), lower, upper)
    x0 = np.concatenate([np.zeros(6, dtype=np.float64), initial_action])
    bounds = [(-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35), (-0.08, 0.08), (-0.08, 0.08), (-0.08, 0.08)]
    bounds.extend(list(zip(lower, upper)))

    object_points = np.asarray(object_points, dtype=np.float64)
    object_normals = normalize_rows(object_normals)
    target_heatmap = np.asarray(target_heatmap, dtype=np.float64)
    high_mask = target_heatmap >= max(0.25, float(np.quantile(target_heatmap, 0.88)))
    if not np.any(high_mask):
        high_mask[np.argmax(target_heatmap)] = True
    object_center = object_points.mean(axis=0)
    object_radius = float(np.max(np.linalg.norm(object_points - object_center, axis=1)))
    joint_span = np.maximum(upper - lower, 1e-8)

    def evaluate(x: np.ndarray, include_terms: bool = False) -> float | dict[str, Any]:
        rotation, translation = pose_from_params(x[:6], init_rotation, init_translation)
        action = np.asarray(x[6:], dtype=np.float64)
        surface = robot_model.materialize_surface(robot_topology, action)
        surface_points_scene = transform_points(surface.points, rotation, translation)
        surface_normals_scene = transform_normals(surface.normals, rotation)
        distances, nearest = aligned_distance(
            object_points,
            object_normals,
            surface_points_scene,
            gamma=alignment_gamma,
        )
        current_heatmap = heatmap_from_distance(distances, heatmap_sigma)

        weights = 1.0 + float(high_contact_weight) * target_heatmap
        heatmap_loss = float(np.mean(weights * (current_heatmap - target_heatmap) ** 2))
        surface_loss = float(np.mean((target_heatmap[high_mask] + 1e-3) * distances[high_mask] ** 2))
        signed_offset = np.sum((nearest - object_points) * object_normals, axis=1)
        penetration_loss = float(np.mean((target_heatmap + current_heatmap) * np.maximum(-signed_offset, 0.0) ** 2))

        wrist_tips = np.asarray(
            [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in fingertip_links],
            dtype=np.float64,
        )
        tips_scene = transform_points(wrist_tips, rotation, translation)
        fingertip_loss = float(np.mean(np.sum((tips_scene - mano_fingertips_scene) ** 2, axis=1)))

        fc_loss = 0.0
        if fc_weight > 0.0:
            top_count = min(8, len(object_points))
            top_indices = np.argsort(-(target_heatmap + current_heatmap))[:top_count]
            fc_loss = float(
                dexgraspnet_force_closure_energy(
                    object_points[top_indices],
                    -object_normals[top_indices],
                    object_center=object_center,
                    torque_scale=1.0 / max(object_radius, 1e-6),
                    reduction="mean",
                )
            )

        pose_loss = float(np.mean((x[:3] / 0.35) ** 2) + np.mean((x[3:6] / 0.08) ** 2))
        joint_loss = float(np.mean(((action - initial_action) / joint_span) ** 2))
        total = (
            float(heatmap_weight) * heatmap_loss
            + float(surface_attraction_weight) * surface_loss
            + float(penetration_weight) * penetration_loss
            + float(fingertip_prior_weight) * fingertip_loss
            + float(fc_weight) * fc_loss
            + float(regularization_weight) * (pose_loss + joint_loss)
        )
        if include_terms:
            return {
                "heatmap": heatmap_loss,
                "surface_attraction": surface_loss,
                "penetration": penetration_loss,
                "fingertip_prior": fingertip_loss,
                "dex_fc": fc_loss,
                "pose_regularization": pose_loss,
                "joint_regularization": joint_loss,
                "total": float(total),
                "mean_current_heatmap": float(np.mean(current_heatmap)),
                "max_current_heatmap": float(np.max(current_heatmap)),
                "mean_high_contact_distance_m": float(np.mean(distances[high_mask])),
            }
        return float(total)

    started_at = time.perf_counter()
    initial_terms = evaluate(x0, include_terms=True)
    method = str(optimizer)
    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if method == "L-BFGS-B":
        options["ftol"] = 1e-10
    else:
        options["xtol"] = 1e-4
        options["ftol"] = 1e-5
    result = minimize(
        lambda values: float(evaluate(values)),
        x0,
        method=method,
        bounds=bounds,
        options=options,
    )
    elapsed = time.perf_counter() - started_at

    x_best = np.asarray(result.x, dtype=np.float64)
    best_terms = evaluate(x_best, include_terms=True)
    rotation, translation = pose_from_params(x_best[:6], init_rotation, init_translation)
    action = x_best[6:]
    surface = robot_model.materialize_surface(robot_topology, action)
    surface_points_scene = transform_points(surface.points, rotation, translation)
    surface_normals_scene = transform_normals(surface.normals, rotation)
    distances, nearest = aligned_distance(object_points, object_normals, surface_points_scene, gamma=alignment_gamma)
    current_heatmap = heatmap_from_distance(distances, heatmap_sigma)

    robot_tree = cKDTree(surface_points_scene)
    _, robot_indices = robot_tree.query(object_points)
    robot_nearest_normals = surface_normals_scene[np.asarray(robot_indices, dtype=np.int64)]
    normal_alignment = np.sum(robot_nearest_normals * -object_normals, axis=1)

    return {
        "action": action,
        "pose_params": x_best[:6],
        "rotation": rotation,
        "translation": translation,
        "surface_points_scene": surface_points_scene,
        "surface_normals_scene": surface_normals_scene,
        "current_heatmap": current_heatmap,
        "heatmap_distance": distances,
        "nearest_robot_points_scene": nearest,
        "normal_alignment": normal_alignment,
        "initial_loss_terms": initial_terms,
        "best_loss_terms": best_terms,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "elapsed_seconds": float(elapsed),
    }


def run_contact_heatmap(
    sequence: Path,
    frame: str,
    cad: Path,
    hand_path: Path,
    mano_root: Path,
    robot_profile: str,
    *,
    position_restarts: int,
    position_max_nfev: int,
    num_object_samples: int,
    num_mano_samples: int,
    num_robot_samples: int,
    heatmap_sigma: float,
    alignment_gamma: float,
    heatmap_weight: float,
    high_contact_weight: float,
    surface_attraction_weight: float,
    penetration_weight: float,
    fingertip_prior_weight: float,
    fc_weight: float,
    regularization_weight: float,
    optimizer: str,
    maxiter: int,
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

    _, mano, object_vertices_scene, object_faces = load_hoi4d_scene(sequence, frame, cad, hand_path, mano_root)
    object_samples = sample_mesh_surface(
        object_vertices_scene,
        object_faces,
        num_points=num_object_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    mano_samples = sample_mesh_surface(
        mano.vertices,
        mano.faces,
        num_points=num_mano_samples,
        seed=seed + 1,
        use_farthest_point_sampling=True,
    )
    target_distances, _ = aligned_distance(
        object_samples.points,
        object_samples.normals,
        mano_samples.points,
        gamma=alignment_gamma,
    )
    target_heatmap = heatmap_from_distance(target_distances, heatmap_sigma)

    profile = ROBOT_PROFILES[robot_profile]
    robot_model = load_robot_model(robot_profile)
    robot_topology = robot_model.sample_surface_topology(
        num_points=num_robot_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    scene_to_robot = PointSetRigidTransform(
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        np.asarray(position_result.target_fingertips_wrist, dtype=np.float64),
    )
    init_rotation, init_translation = scene_to_robot.robot_to_scene_pose()
    initial_action = np.asarray(position_result.action, dtype=np.float64)
    mano_fingertips_scene = np.asarray(position_result.mano_fingertips, dtype=np.float64)
    initial_surface = robot_model.materialize_surface(robot_topology, initial_action)
    initial_surface_points_scene = transform_points(initial_surface.points, init_rotation, init_translation)
    initial_distances, _ = aligned_distance(
        object_samples.points,
        object_samples.normals,
        initial_surface_points_scene,
        gamma=alignment_gamma,
    )
    initial_heatmap = heatmap_from_distance(initial_distances, heatmap_sigma)

    result = optimize_heatmap_pose_and_action(
        robot_model,
        robot_topology,
        object_samples.points,
        object_samples.normals,
        target_heatmap,
        mano_fingertips_scene,
        profile["fingertip_links"],
        initial_action,
        init_rotation,
        init_translation,
        heatmap_sigma=heatmap_sigma,
        alignment_gamma=alignment_gamma,
        heatmap_weight=heatmap_weight,
        high_contact_weight=high_contact_weight,
        surface_attraction_weight=surface_attraction_weight,
        penetration_weight=penetration_weight,
        fingertip_prior_weight=fingertip_prior_weight,
        fc_weight=fc_weight,
        regularization_weight=regularization_weight,
        optimizer=optimizer,
        maxiter=maxiter,
    )

    best_action = np.asarray(result["action"], dtype=np.float64)
    best_rotation = np.asarray(result["rotation"], dtype=np.float64)
    best_translation = np.asarray(result["translation"], dtype=np.float64)
    best_surface = robot_model.materialize_surface(robot_topology, best_action)
    robot_fingertips = np.asarray(
        [robot_tip_from_link(best_surface.points, best_surface.link_names.astype(str), link) for link in profile["fingertip_links"]],
        dtype=np.float64,
    )
    robot_fingertips_scene = transform_points(robot_fingertips, best_rotation, best_translation)
    fingertip_errors = np.linalg.norm(robot_fingertips_scene - mano_fingertips_scene, axis=1)
    robot_wrist_scene = transform_points(np.zeros((1, 3), dtype=np.float64), best_rotation, best_translation)[0]

    target_high = target_heatmap >= max(0.25, float(np.quantile(target_heatmap, 0.88)))
    heatmap_error = np.abs(np.asarray(result["current_heatmap"]) - target_heatmap)
    robot_mesh_vertices_scene: list[list[float]] = []
    robot_mesh_faces: list[list[int]] = []
    robot_mesh = _robot_mesh_for_action(profile, best_action)
    if robot_mesh is not None:
        mesh_vertices, mesh_faces = simplify_mesh_for_html(*robot_mesh, max_faces=ROBOT_MESH_MAX_FACES)
        robot_mesh_vertices_scene = transform_points(mesh_vertices, best_rotation, best_translation).astype(float).tolist()
        robot_mesh_faces = mesh_faces.astype(int).tolist()

    return {
        "status": "ok",
        "route": "contact_heatmap_matching",
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
        "heatmap_refinement": {
            "optimization_seconds": float(result["elapsed_seconds"]),
            "success": bool(result["success"]),
            "message": result["message"],
            "iterations": int(result["iterations"]),
            "initial_loss_terms": {key: float(value) for key, value in result["initial_loss_terms"].items()},
            "best_loss_terms": {key: float(value) for key, value in result["best_loss_terms"].items()},
        },
        "heatmap_error": {
            "mse": float(np.mean((np.asarray(result["current_heatmap"]) - target_heatmap) ** 2)),
            "weighted_mse": float(np.mean((1.0 + high_contact_weight * target_heatmap) * (np.asarray(result["current_heatmap"]) - target_heatmap) ** 2)),
            "mean_abs": float(np.mean(heatmap_error)),
            "max_abs": float(np.max(heatmap_error)),
            "target_mean": float(np.mean(target_heatmap)),
            "target_max": float(np.max(target_heatmap)),
            "current_mean": float(np.mean(result["current_heatmap"])),
            "current_max": float(np.max(result["current_heatmap"])),
            "high_contact_count": int(np.sum(target_high)),
            "mean_high_contact_distance_m": float(np.mean(np.asarray(result["heatmap_distance"])[target_high])),
            "mean_high_contact_abs_error": float(np.mean(heatmap_error[target_high])),
            "mean_normal_alignment": float(np.mean(np.asarray(result["normal_alignment"])[target_high])),
        },
        "fingertip_error": {
            "mean_m": float(np.mean(fingertip_errors)),
            "max_m": float(np.max(fingertip_errors)),
            "per_finger_m": fingertip_errors.astype(float).tolist(),
        },
        "config": {
            "optimizer": "joint_wrist_pose_and_action",
            "scipy_method": str(optimizer),
            "heatmap_formula": "exp(-(aligned_distance / sigma)^2)",
            "aligned_distance": "nearest distance weighted by exp(gamma * (1 - dot(direction_to_hand, object_normal)))",
            "num_object_samples": int(num_object_samples),
            "num_mano_samples": int(num_mano_samples),
            "num_robot_samples": int(num_robot_samples),
            "heatmap_sigma": float(heatmap_sigma),
            "alignment_gamma": float(alignment_gamma),
            "heatmap_weight": float(heatmap_weight),
            "high_contact_weight": float(high_contact_weight),
            "surface_attraction_weight": float(surface_attraction_weight),
            "penetration_weight": float(penetration_weight),
            "fingertip_prior_weight": float(fingertip_prior_weight),
            "fc_weight": float(fc_weight),
            "regularization_weight": float(regularization_weight),
            "maxiter": int(maxiter),
        },
        "arrays": {
            "mano_vertices": mano.vertices.astype(float).tolist(),
            "mano_faces": mano.faces.astype(int).tolist(),
            "object_vertices_scene": object_vertices_scene.astype(float).tolist(),
            "object_faces": object_faces.astype(int).tolist(),
            "object_sample_points_scene": object_samples.points.astype(float).tolist(),
            "object_sample_normals_scene": object_samples.normals.astype(float).tolist(),
            "target_heatmap": target_heatmap.astype(float).tolist(),
            "initial_heatmap": initial_heatmap.astype(float).tolist(),
            "current_heatmap": np.asarray(result["current_heatmap"], dtype=np.float64).astype(float).tolist(),
            "heatmap_error": heatmap_error.astype(float).tolist(),
            "nearest_robot_points_scene": np.asarray(result["nearest_robot_points_scene"], dtype=np.float64).astype(float).tolist(),
            "robot_surface_points_scene": np.asarray(result["surface_points_scene"], dtype=np.float64).astype(float).tolist(),
            "robot_surface_normals_scene": np.asarray(result["surface_normals_scene"], dtype=np.float64).astype(float).tolist(),
            "mano_fingertips_scene": mano_fingertips_scene.astype(float).tolist(),
            "robot_fingertips_scene": robot_fingertips_scene.astype(float).tolist(),
            "robot_wrist_scene": robot_wrist_scene.astype(float).tolist(),
            "robot_mesh_vertices_scene": robot_mesh_vertices_scene,
            "robot_mesh_faces": robot_mesh_faces,
        },
        "metadata": {
            "model_format": profile["model_format"],
            "robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")),
            "wrist_link": profile["wrist_link"],
            "visual_alignment": "optimized_wrist_pose_rigid_only_no_scale",
            "scene_to_robot_frame": "fitted_from_mano_fingertips_to_position_only_target_fingertips",
            "reference": "GenDexGrasp object-centric contact map objective adapted to scipy/robot-surface retargeting",
            "robot_mesh_max_faces": int(ROBOT_MESH_MAX_FACES),
        },
    }


def write_contact_heatmap_html(case_dir: Path, rgb: Path, base_meta: dict[str, Any], payload: dict[str, Any]) -> Path:
    base_fig, scene_meta = build_figure(
        Path(base_meta["sequence"]),
        str(base_meta["frame"]),
        Path(base_meta["cad"]),
        Path(base_meta["hand_pickle"]),
        Path(base_meta["mano_root"]),
    )
    style_scene_traces(base_fig)
    arrays = payload["arrays"]
    object_points = np.asarray(arrays["object_sample_points_scene"], dtype=np.float64)
    target_heatmap = np.asarray(arrays["target_heatmap"], dtype=np.float64)
    initial_heatmap = np.asarray(arrays["initial_heatmap"], dtype=np.float64)
    current_heatmap = np.asarray(arrays["current_heatmap"], dtype=np.float64)
    robot_mesh_vertices = np.asarray(arrays.get("robot_mesh_vertices_scene") or [], dtype=np.float64)
    robot_mesh_faces = np.asarray(arrays.get("robot_mesh_faces") or [], dtype=np.int64)

    blue_red = [
        [0.0, "#2563eb"],
        [0.25, "#38bdf8"],
        [0.5, "#facc15"],
        [0.75, "#fb923c"],
        [1.0, "#dc2626"],
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene", "colspan": 2}, None], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Retargeted Meshes", "Initial Robot Heatmap", "Optimized Robot Heatmap"),
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
        row_heights=[0.62, 0.38],
    )
    for trace in base_fig.data:
        name = str(getattr(trace, "name", ""))
        if name == "object CAD" or "MANO hand" in name:
            fig.add_trace(trace, row=1, col=1)
    if robot_mesh_vertices.size and robot_mesh_faces.size:
        fig.add_trace(mesh_trace(f"{payload['robot_profile']} mesh", robot_mesh_vertices, robot_mesh_faces, "#2563eb", 0.34), row=1, col=1)

    for col in (2, 3):
        for trace in base_fig.data:
            if str(getattr(trace, "name", "")) == "object CAD":
                copied = go.Mesh3d(trace)
                copied.opacity = 0.35
                copied.showlegend = False
                fig.add_trace(copied, row=2, col=col - 1)
    initial_trace = heatmap_trace("initial robot heatmap", object_points, initial_heatmap, blue_red, size=2.2)
    initial_trace.marker.colorbar = dict(title="initial", x=0.48, y=0.18, len=0.30, thickness=12)
    final_trace = heatmap_trace("optimized robot heatmap", object_points, current_heatmap, blue_red, size=2.2)
    final_trace.marker.colorbar = dict(title="optimized", x=1.0, y=0.18, len=0.30, thickness=12)
    fig.add_trace(initial_trace, row=2, col=1)
    fig.add_trace(final_trace, row=2, col=2)

    view_points = [
        object_points,
    ]
    if robot_mesh_vertices.size:
        view_points.append(robot_mesh_vertices[:: max(len(robot_mesh_vertices) // 5000, 1)])
    x_range, y_range, z_range = bounds_for_points(np.vstack(view_points), margin=0.18)
    scene_common = dict(
        xaxis=dict(range=x_range, visible=False),
        yaxis=dict(range=y_range, visible=False),
        zaxis=dict(range=z_range, visible=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.25, y=-1.7, z=0.85), up=dict(x=0, y=0, z=1)),
    )
    fig.update_layout(
        title=dict(
            text=f"Contact heatmap retargeting ({payload['robot_profile']})",
            x=0.5,
            y=0.985,
            font=dict(size=18),
        ),
        margin=dict(l=0, r=8, t=58, b=0),
        paper_bgcolor="white",
        showlegend=False,
        scene=scene_common,
        scene2=scene_common,
        scene3=scene_common,
    )

    heat = payload["heatmap_error"]
    errors = {
        "route": payload["route"],
        "robot hand": payload["robot_profile"],
        "position init time": f"{payload['position_initialization']['optimization_seconds']:.3f} s",
        "heatmap refine time": f"{payload['heatmap_refinement']['optimization_seconds']:.3f} s",
        "heatmap MSE": f"{heat['mse']:.6f}",
        "mean heatmap abs err": f"{heat['mean_abs']:.4f}",
        "high contact count": heat["high_contact_count"],
        "mean high-contact dist": f"{heat['mean_high_contact_distance_m'] * 1000.0:.2f} mm",
        "mean fingertip error": f"{payload['fingertip_error']['mean_m'] * 1000.0:.2f} mm",
        "max fingertip error": f"{payload['fingertip_error']['max_m'] * 1000.0:.2f} mm",
        "status": payload["status"],
    }
    scene_meta.update(base_meta)
    scene_meta["errors"] = errors
    scene_meta["contact_heatmap"] = {
        key: payload[key]
        for key in (
            "route",
            "robot_profile",
            "position_initialization",
            "heatmap_refinement",
            "heatmap_error",
            "fingertip_error",
            "config",
            "metadata",
        )
    }
    html = case_dir / f"scene_{payload['robot_profile']}_contact_heatmap.html"
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
    parser.add_argument("--case-id", default="bottle_frame74_contact_heatmap")
    parser.add_argument("--robot-profile", choices=sorted(ROBOT_PROFILES), default="shadow_hand")
    parser.add_argument("--position-restarts", type=int, default=4)
    parser.add_argument("--position-max-nfev", type=int, default=120)
    parser.add_argument("--num-object-samples", type=int, default=512)
    parser.add_argument("--num-mano-samples", type=int, default=1600)
    parser.add_argument("--num-robot-samples", type=int, default=1200)
    parser.add_argument("--heatmap-sigma", type=float, default=0.018)
    parser.add_argument("--alignment-gamma", type=float, default=1.0)
    parser.add_argument("--heatmap-weight", type=float, default=30.0)
    parser.add_argument("--high-contact-weight", type=float, default=6.0)
    parser.add_argument("--surface-attraction-weight", type=float, default=75.0)
    parser.add_argument("--penetration-weight", type=float, default=120.0)
    parser.add_argument("--fingertip-prior-weight", type=float, default=8.0)
    parser.add_argument("--fc-weight", type=float, default=0.02)
    parser.add_argument("--regularization-weight", type=float, default=0.005)
    parser.add_argument("--optimizer", choices=["Powell", "L-BFGS-B"], default="Powell")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip Plotly HTML rendering for large batch runs.")
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
        render_input_html=not args.no_html,
    )
    payload = run_contact_heatmap(
        args.sequence,
        str(args.frame),
        args.cad,
        args.hand,
        args.mano_root,
        args.robot_profile,
        position_restarts=args.position_restarts,
        position_max_nfev=args.position_max_nfev,
        num_object_samples=args.num_object_samples,
        num_mano_samples=args.num_mano_samples,
        num_robot_samples=args.num_robot_samples,
        heatmap_sigma=args.heatmap_sigma,
        alignment_gamma=args.alignment_gamma,
        heatmap_weight=args.heatmap_weight,
        high_contact_weight=args.high_contact_weight,
        surface_attraction_weight=args.surface_attraction_weight,
        penetration_weight=args.penetration_weight,
        fingertip_prior_weight=args.fingertip_prior_weight,
        fc_weight=args.fc_weight,
        regularization_weight=args.regularization_weight,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
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
        "route": "contact_heatmap_matching",
    }
    json_path = case_dir / f"retargeted_{args.robot_profile}_contact_heatmap.json"
    payload_out = {**base_meta, **{key: value for key, value in payload.items() if key != "arrays"}}
    json_path.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = None
    if not args.no_html:
        html_path = write_contact_heatmap_html(case_dir, case_dir / "original.jpg", base_meta, payload)
    print(f"Wrote {json_path}")
    if html_path is not None:
        print(f"Open: {html_path}")


if __name__ == "__main__":
    main()
