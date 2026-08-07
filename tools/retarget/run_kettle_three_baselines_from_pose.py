#!/usr/bin/env python3
"""Run three retargeting baselines on the generated kettle phase4 samples.

Inputs are the artifacts produced by:
  - tools/pose/run_kettle_foundationpose_test.py
  - tools/pose/run_kettle_hamer_official.py or tools/pose/run_kettle_hamer_direct_bbox.py
  - tools/pose/run_kettle_phase5_5_alignment.py

The script adapts those artifacts into the same internal geometry used by the
HOI4D retargeting baselines. By default it uses phase5.5 SE(3)-aligned hand
meshes/keypoints when an alignment directory is available.
"""

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
RETARGET_ROOT = REPO_ROOT / "tools" / "retarget"
for path in (SRC_ROOT, RETARGET_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prosthetic_grasp.common.types import Phase5HandPrediction, Phase5ManoResult  # noqa: E402
from prosthetic_grasp.phases.phase6_prosthetic_action import (  # noqa: E402
    Phase6ProstheticAction,
    Phase6ProstheticActionConfig,
)
from prosthetic_grasp.geometry import robot_tip_from_link  # noqa: E402
from prosthetic_grasp.geometry.surface_sampling import sample_mesh_surface  # noqa: E402

from contact_surface import ROBOT_PROFILES, load_robot_model  # noqa: E402
from contact_heatmap import (  # noqa: E402
    aligned_distance,
    heatmap_from_distance,
    normalize_rows,
    optimize_heatmap_pose_and_action,
)
from position import load_retargeted_robot_mesh, simplify_mesh_for_html  # noqa: E402
from position_force import (  # noqa: E402
    PointSetRigidTransform,
    make_fingertip_projection_patches,
    optimize_pose_and_action,
    transform_normals,
    transform_points,
)


SAMPLE_IDS = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def collect_cases(output_root: Path, hamer_subdir: str, fallback_hamer_subdir: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for sample_id in SAMPLE_IDS:
        sample_dir = output_root / sample_id
        for ldir in sorted(sample_dir.glob("lollipop_*")):
            rgb = ldir / "phase4_inpaint_full.png"
            hamer_dir = ldir / hamer_subdir
            if not (hamer_dir / "hand_00_camera.obj").exists() and fallback_hamer_subdir:
                fallback_dir = ldir / fallback_hamer_subdir
                if (fallback_dir / "hand_00_camera.obj").exists():
                    hamer_dir = fallback_dir
            keypoints = hamer_dir / "hand_00_keypoints_camera.npy"
            vertices = hamer_dir / "hand_00_vertices_camera.npy"
            hand_mesh = hamer_dir / "hand_00_camera.obj"
            pose = ldir / "pose_foundationpose" / "object_in_camera.txt"
            if rgb.exists() and keypoints.exists() and vertices.exists() and hand_mesh.exists() and pose.exists():
                cases.append(
                    {
                        "sample_id": sample_id,
                        "lollipop": ldir.name,
                        "dir": ldir,
                        "rgb": rgb,
                        "keypoints": keypoints,
                        "vertices": vertices,
                        "hand_mesh": hand_mesh,
                        "hamer_dir": hamer_dir,
                        "hamer_subdir": hamer_dir.name,
                        "object_pose": pose,
                    }
                )
    return cases


def load_scene_mesh(mesh_path: Path, pose_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pose = np.loadtxt(pose_path, dtype=np.float64)
    vertices_h = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64), np.ones((len(mesh.vertices), 1))], axis=1)
    vertices_scene = (pose @ vertices_h.T).T[:, :3]
    return trimesh.Trimesh(vertices=vertices_scene, faces=np.asarray(mesh.faces), process=False)


def make_phase5(case: dict[str, Any], hand_shift: np.ndarray | None = None) -> Phase5ManoResult:
    hand_shift = np.zeros(3, dtype=np.float64) if hand_shift is None else np.asarray(hand_shift, dtype=np.float64)
    keypoints = np.load(case["keypoints"]).astype(np.float64) + hand_shift.reshape(1, 3)
    vertices = np.load(case["vertices"]).astype(np.float64) + hand_shift.reshape(1, 3)
    meta_path = Path(case.get("hamer_dir", case["dir"] / "pose_hamer")) / "hamer_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    bbox = np.asarray(meta.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    hand = Phase5HandPrediction(
        hand_index=0,
        is_right=True,
        bbox_xyxy=bbox,
        keypoints_2d=np.zeros((21, 2), dtype=np.float32),
        keypoint_score_mean=1.0,
        vertices=vertices.astype(np.float32),
        keypoints_3d=keypoints.astype(np.float32),
        pred_cam=np.zeros(3, dtype=np.float32),
        pred_cam_t_crop=np.zeros(3, dtype=np.float32),
        pred_cam_t_full=np.zeros(3, dtype=np.float32),
        focal_length=0.0,
        mano_params={},
    )
    hand_mesh = trimesh.load_mesh(str(case["hand_mesh"]), process=False)
    return Phase5ManoResult(
        status="ok",
        message="HaMeR direct bbox result adapted from saved keypoints.",
        faces=np.asarray(hand_mesh.faces, dtype=np.int64),
        hands=[hand],
    )


def robot_mesh_for_action(profile: dict[str, Any], action: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    return load_retargeted_robot_mesh(
        SimpleNamespace(
            action=np.asarray(action, dtype=np.float64),
            metadata={
                "robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")),
                "robot_urdf_path": str(profile.get("urdf_path") or profile.get("xml_path")),
                "model_format": profile["model_format"],
                "wrist_link": profile["wrist_link"],
            },
        )
    )


def export_transformed_mesh(
    out_path: Path,
    profile: dict[str, Any],
    action: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_data = robot_mesh_for_action(profile, action)
    if mesh_data is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    vertices, faces = mesh_data
    vertices_scene = transform_points(vertices, rotation, translation)
    trimesh.Trimesh(vertices=vertices_scene, faces=faces, process=False).export(out_path)
    return vertices_scene, faces


def route_position_only(
    phase5: Phase5ManoResult,
    robot_profile: str,
    position_restarts: int,
    position_max_nfev: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    cfg = Phase6ProstheticActionConfig(
        robot_profile=robot_profile,
        optimization_restarts=position_restarts,
        max_nfev=position_max_nfev,
    )
    result = Phase6ProstheticAction(cfg).run(phase5)
    if result.status != "ok":
        raise RuntimeError(f"position-only failed: {result.status} {result.message}")
    fit = PointSetRigidTransform(
        np.asarray(result.mano_fingertips, dtype=np.float64),
        np.asarray(result.target_fingertips_wrist, dtype=np.float64),
    )
    rotation, translation = fit.robot_to_scene_pose()
    return result, rotation, translation


def route_position_force(
    position_result: Any,
    object_mesh: trimesh.Trimesh,
    robot_profile: str,
    *,
    num_robot_samples: int,
    contact_top_k: int,
    contact_temperature: float,
    maxiter: int,
    seed: int,
) -> dict[str, Any]:
    profile = ROBOT_PROFILES[robot_profile]
    robot_model = load_robot_model(robot_profile)
    robot_topology = robot_model.sample_surface_topology(
        num_points=num_robot_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    zero_surface = robot_model.materialize_surface(robot_topology, robot_model.zero_action)
    fit = PointSetRigidTransform(
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        np.asarray(position_result.target_fingertips_wrist, dtype=np.float64),
    )
    init_rotation, init_translation = fit.robot_to_scene_pose()
    initial_action = np.asarray(position_result.action, dtype=np.float64)
    initial_surface = robot_model.materialize_surface(robot_topology, initial_action)
    initial_surface_scene = SimpleNamespace(
        points=transform_points(initial_surface.points, init_rotation, init_translation),
        normals=transform_normals(initial_surface.normals, init_rotation),
        local_points=initial_surface.local_points,
        local_normals=initial_surface.local_normals,
        link_names=initial_surface.link_names,
    )
    patches, _, _, initial_projection_distances = make_fingertip_projection_patches(
        initial_surface_scene,
        zero_surface,
        np.asarray(object_mesh.vertices, dtype=np.float64),
        np.asarray(object_mesh.faces, dtype=np.int64),
        profile["fingertip_links"],
        top_k=contact_top_k,
        temperature=contact_temperature,
    )
    result = optimize_pose_and_action(
        robot_model,
        robot_topology,
        patches,
        np.asarray(object_mesh.vertices, dtype=np.float64),
        np.asarray(object_mesh.faces, dtype=np.int64),
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        profile["fingertip_links"],
        initial_action,
        init_rotation,
        init_translation,
        fingertip_weight=25.0,
        contact_weight=40.0,
        surface_attraction_weight=70.0,
        normal_weight=2.0,
        fc_weight=0.01,
        penetration_weight=120.0,
        pose_regularization_weight=0.004,
        joint_regularization_weight=0.004,
        maxiter=maxiter,
        friction_coef=0.8,
        num_friction_edges=8,
    )
    surface = robot_model.materialize_surface(robot_topology, np.asarray(result["action"], dtype=np.float64))
    robot_tips = np.asarray(
        [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in profile["fingertip_links"]],
        dtype=np.float64,
    )
    robot_tips_scene = transform_points(robot_tips, result["rotation"], result["translation"])
    fingertip_errors = np.linalg.norm(robot_tips_scene - np.asarray(position_result.mano_fingertips), axis=1)
    return {
        "route": "position_force",
        "action": np.asarray(result["action"], dtype=np.float64),
        "rotation": np.asarray(result["rotation"], dtype=np.float64),
        "translation": np.asarray(result["translation"], dtype=np.float64),
        "fingertip_errors": fingertip_errors,
        "initial_projection_distances": np.asarray(initial_projection_distances, dtype=np.float64),
        "loss_terms": result["best_loss_terms"],
        "success": bool(result["success"]),
        "message": str(result["message"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "force_closure": {
            "is_force_closure": bool(result["force_closure"].is_force_closure),
            "epsilon": float(result["force_closure"].epsilon),
            "rank": int(result["force_closure"].rank),
            "dex_fc_energy": float(result["dex_fc_energy"]),
        },
    }


def route_contact_heatmap(
    position_result: Any,
    hand_mesh: trimesh.Trimesh,
    object_mesh: trimesh.Trimesh,
    robot_profile: str,
    *,
    num_object_samples: int,
    num_mano_samples: int,
    num_robot_samples: int,
    maxiter: int,
    seed: int,
) -> dict[str, Any]:
    profile = ROBOT_PROFILES[robot_profile]
    robot_model = load_robot_model(robot_profile)
    object_samples = sample_mesh_surface(
        np.asarray(object_mesh.vertices, dtype=np.float64),
        np.asarray(object_mesh.faces, dtype=np.int64),
        num_points=num_object_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    hand_samples = sample_mesh_surface(
        np.asarray(hand_mesh.vertices, dtype=np.float64),
        np.asarray(hand_mesh.faces, dtype=np.int64),
        num_points=num_mano_samples,
        seed=seed + 1,
        use_farthest_point_sampling=True,
    )
    heatmap_sigma = 0.018
    alignment_gamma = 1.0
    target_distances, _ = aligned_distance(
        object_samples.points,
        object_samples.normals,
        hand_samples.points,
        gamma=alignment_gamma,
    )
    target_heatmap = heatmap_from_distance(target_distances, heatmap_sigma)
    robot_topology = robot_model.sample_surface_topology(
        num_points=num_robot_samples,
        seed=seed,
        use_farthest_point_sampling=True,
    )
    fit = PointSetRigidTransform(
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        np.asarray(position_result.target_fingertips_wrist, dtype=np.float64),
    )
    init_rotation, init_translation = fit.robot_to_scene_pose()
    result = optimize_heatmap_pose_and_action(
        robot_model,
        robot_topology,
        object_samples.points,
        normalize_rows(object_samples.normals),
        target_heatmap,
        np.asarray(position_result.mano_fingertips, dtype=np.float64),
        profile["fingertip_links"],
        np.asarray(position_result.action, dtype=np.float64),
        init_rotation,
        init_translation,
        heatmap_sigma=heatmap_sigma,
        alignment_gamma=alignment_gamma,
        heatmap_weight=25.0,
        high_contact_weight=5.0,
        surface_attraction_weight=55.0,
        penetration_weight=100.0,
        fingertip_prior_weight=8.0,
        fc_weight=0.01,
        regularization_weight=0.004,
        optimizer="L-BFGS-B",
        maxiter=maxiter,
    )
    surface = robot_model.materialize_surface(robot_topology, np.asarray(result["action"], dtype=np.float64))
    robot_tips = np.asarray(
        [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in profile["fingertip_links"]],
        dtype=np.float64,
    )
    robot_tips_scene = transform_points(robot_tips, result["rotation"], result["translation"])
    fingertip_errors = np.linalg.norm(robot_tips_scene - np.asarray(position_result.mano_fingertips), axis=1)
    heatmap_error = np.asarray(result["current_heatmap"]) - target_heatmap
    high_mask = target_heatmap >= max(0.25, float(np.quantile(target_heatmap, 0.88)))
    if not np.any(high_mask):
        high_mask[np.argmax(target_heatmap)] = True
    return {
        "route": "contact_heatmap",
        "action": np.asarray(result["action"], dtype=np.float64),
        "rotation": np.asarray(result["rotation"], dtype=np.float64),
        "translation": np.asarray(result["translation"], dtype=np.float64),
        "fingertip_errors": fingertip_errors,
        "loss_terms": result["best_loss_terms"],
        "success": bool(result["success"]),
        "message": str(result["message"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "heatmap": {
            "mse": float(np.mean(heatmap_error**2)),
            "mean_abs": float(np.mean(np.abs(heatmap_error))),
            "target_max": float(np.max(target_heatmap)),
            "current_max": float(np.max(result["current_heatmap"])),
            "mean_high_contact_distance_m": float(np.mean(np.asarray(result["heatmap_distance"])[high_mask])),
        },
    }


def mesh_trace(name: str, mesh: trimesh.Trimesh, color: str, opacity: float) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return go.Mesh3d(
        name=name,
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
    )


def array_mesh_trace(name: str, vertices: np.ndarray, faces: np.ndarray, color: str, opacity: float) -> go.Mesh3d:
    return go.Mesh3d(
        name=name,
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
    )


def marker_trace(name: str, points: np.ndarray, color: str, size: int) -> go.Scatter3d:
    return go.Scatter3d(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color),
    )


def write_case_html(
    out_path: Path,
    case_name: str,
    object_mesh: trimesh.Trimesh,
    hand_mesh: trimesh.Trimesh,
    mano_fingertips: np.ndarray,
    route_meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("position-only", "position-force", "contact-heatmap"),
    )
    route_names = ["position_only", "position_force", "contact_heatmap"]
    colors = {
        "position_only": "#2563eb",
        "position_force": "#dc2626",
        "contact_heatmap": "#16a34a",
    }
    all_points = [np.asarray(object_mesh.vertices), np.asarray(hand_mesh.vertices), mano_fingertips]
    for col, route in enumerate(route_names, start=1):
        fig.add_trace(mesh_trace("object", object_mesh, "#9ca3af", 0.28), row=1, col=col)
        fig.add_trace(mesh_trace("HaMeR hand", hand_mesh, "#f59e0b", 0.22), row=1, col=col)
        fig.add_trace(marker_trace("MANO fingertips", mano_fingertips, "#111827", 4), row=1, col=col)
        vertices, faces, fingertips = route_meshes[route]
        if len(vertices) and len(faces):
            fig.add_trace(array_mesh_trace(route, vertices, faces, colors[route], 0.42), row=1, col=col)
            all_points.append(vertices[:: max(1, len(vertices) // 2000)])
        fig.add_trace(marker_trace(f"{route} fingertips", fingertips, colors[route], 4), row=1, col=col)
    points = np.vstack(all_points)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = float(np.max(points.max(axis=0) - points.min(axis=0)) * 0.56 + 0.03)
    scene = dict(
        xaxis=dict(range=[center[0] - radius, center[0] + radius], visible=False),
        yaxis=dict(range=[center[1] - radius, center[1] + radius], visible=False),
        zaxis=dict(range=[center[2] - radius, center[2] + radius], visible=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.35, y=-1.45, z=0.9), up=dict(x=0, y=0, z=1)),
    )
    fig.update_layout(
        title=f"Kettle retargeting baselines: {case_name}",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=False,
        scene=scene,
        scene2=scene,
        scene3=scene,
        paper_bgcolor="white",
    )
    out_path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def action_fingertips_scene(robot_profile: str, action: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    profile = ROBOT_PROFILES[robot_profile]
    robot_model = load_robot_model(robot_profile)
    topology = robot_model.sample_surface_topology(num_points=300, seed=101, use_farthest_point_sampling=True)
    surface = robot_model.materialize_surface(topology, np.asarray(action, dtype=np.float64))
    tips = np.asarray(
        [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in profile["fingertip_links"]],
        dtype=np.float64,
    )
    return transform_points(tips, rotation, translation)


def run_case(
    case: dict[str, Any],
    object_mesh_path: Path,
    output_root: Path,
    robot_profile: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_name = f"{case['sample_id']}_{case['lollipop']}"
    case_out = output_root / case["sample_id"] / case["lollipop"]
    case_out.mkdir(parents=True, exist_ok=True)

    object_mesh = load_scene_mesh(object_mesh_path, case["object_pose"])
    alignment_case_dir = None
    if args.alignment_root and args.alignment_method:
        alignment_case_dir = args.alignment_root / case["sample_id"] / case["lollipop"] / args.alignment_method
    if alignment_case_dir and (alignment_case_dir / "aligned_hand.obj").exists():
        hand_mesh_raw = trimesh.load_mesh(str(alignment_case_dir / "aligned_hand.obj"), process=False)
        case = dict(case)
        case["vertices"] = alignment_case_dir / "aligned_vertices.npy"
        case["keypoints"] = alignment_case_dir / "aligned_keypoints.npy"
        case["hand_mesh"] = alignment_case_dir / "aligned_hand.obj"
        hand_shift = np.zeros(3, dtype=np.float64)
        raw_hand_center = np.asarray(hand_mesh_raw.vertices, dtype=np.float64).mean(axis=0)
        alignment_note = {
            "source": "phase5_5_alignment",
            "alignment_root": args.alignment_root,
            "alignment_method": args.alignment_method,
            "alignment_case_dir": alignment_case_dir,
        }
    else:
        hand_mesh_raw = trimesh.load_mesh(str(case["hand_mesh"]), process=False)
        raw_hand_center = np.asarray(hand_mesh_raw.vertices, dtype=np.float64).mean(axis=0)
        alignment_note = {
            "source": "inline_shift",
            "hand_scene_alignment": args.hand_scene_alignment,
        }
    object_center = np.asarray(object_mesh.vertices, dtype=np.float64).mean(axis=0)
    if not (alignment_case_dir and (alignment_case_dir / "aligned_hand.obj").exists()):
        hand_shift = np.zeros(3, dtype=np.float64)
        if args.hand_scene_alignment == "z_center":
            hand_shift[2] = object_center[2] - raw_hand_center[2]
        elif args.hand_scene_alignment == "object_center":
            hand_shift = object_center - raw_hand_center
    hand_mesh = trimesh.Trimesh(
        vertices=np.asarray(hand_mesh_raw.vertices, dtype=np.float64) + hand_shift.reshape(1, 3),
        faces=np.asarray(hand_mesh_raw.faces, dtype=np.int64),
        process=False,
    )
    phase5 = make_phase5(case, hand_shift=hand_shift)
    mano_fingertips = phase5.hands[0].keypoints_3d[[4, 8, 12, 16, 20]].astype(np.float64)

    object_mesh.export(case_out / "object_scene.obj")
    hand_mesh.export(case_out / "hamer_hand_scene.obj")

    started = time.perf_counter()
    pos_result, pos_rotation, pos_translation = route_position_only(
        phase5,
        robot_profile,
        args.position_restarts,
        args.position_max_nfev,
    )
    pos_seconds = time.perf_counter() - started

    profile = ROBOT_PROFILES[robot_profile]
    pos_vertices, pos_faces = export_transformed_mesh(
        case_out / "robot_position_only_scene.obj",
        profile,
        np.asarray(pos_result.action, dtype=np.float64),
        pos_rotation,
        pos_translation,
    )
    pos_tips_scene = action_fingertips_scene(robot_profile, pos_result.action, pos_rotation, pos_translation)

    pf = route_position_force(
        pos_result,
        object_mesh,
        robot_profile,
        num_robot_samples=args.num_robot_samples,
        contact_top_k=args.contact_top_k,
        contact_temperature=args.contact_temperature,
        maxiter=args.force_maxiter,
        seed=args.seed,
    )
    pf_vertices, pf_faces = export_transformed_mesh(
        case_out / "robot_position_force_scene.obj",
        profile,
        pf["action"],
        pf["rotation"],
        pf["translation"],
    )
    pf_tips_scene = action_fingertips_scene(robot_profile, pf["action"], pf["rotation"], pf["translation"])

    hm = route_contact_heatmap(
        pos_result,
        hand_mesh,
        object_mesh,
        robot_profile,
        num_object_samples=args.num_object_samples,
        num_mano_samples=args.num_mano_samples,
        num_robot_samples=args.num_robot_samples,
        maxiter=args.heatmap_maxiter,
        seed=args.seed,
    )
    hm_vertices, hm_faces = export_transformed_mesh(
        case_out / "robot_contact_heatmap_scene.obj",
        profile,
        hm["action"],
        hm["rotation"],
        hm["translation"],
    )
    hm_tips_scene = action_fingertips_scene(robot_profile, hm["action"], hm["rotation"], hm["translation"])

    route_meshes = {
        "position_only": (pos_vertices, pos_faces, pos_tips_scene),
        "position_force": (pf_vertices, pf_faces, pf_tips_scene),
        "contact_heatmap": (hm_vertices, hm_faces, hm_tips_scene),
    }
    write_case_html(
        case_out / "retarget_three_baselines.html",
        case_name,
        object_mesh,
        hand_mesh,
        mano_fingertips,
        route_meshes,
    )

    payload = {
        "case": case_name,
        "sample_id": case["sample_id"],
        "lollipop": case["lollipop"],
        "robot_profile": robot_profile,
        "input": {
            "rgb": case["rgb"],
            "object_pose": case["object_pose"],
            "hand_mesh": case["hand_mesh"],
            "keypoints": case["keypoints"],
            "hamer_dir": case.get("hamer_dir"),
            "hamer_subdir": case.get("hamer_subdir"),
            "object_mesh": object_mesh_path,
            "hand_scene_alignment": args.hand_scene_alignment,
            "alignment": alignment_note,
            "hand_shift_m": hand_shift,
            "raw_hand_center_m": raw_hand_center,
            "object_center_m": object_center,
        },
        "outputs": {
            "case_dir": case_out,
            "html": case_out / "retarget_three_baselines.html",
        },
        "position_only": {
            "status": pos_result.status,
            "elapsed_seconds": pos_seconds,
            "action_names": pos_result.action_names,
            "action": pos_result.action,
            "mean_fingertip_error_m": float(np.mean(pos_result.fingertip_error)),
            "max_fingertip_error_m": float(np.max(pos_result.fingertip_error)),
            "per_finger_error_m": pos_result.fingertip_error,
            "scale": pos_result.metadata.get("scale"),
        },
        "position_force": {
            "success": pf["success"],
            "message": pf["message"],
            "elapsed_seconds": pf["elapsed_seconds"],
            "action": pf["action"],
            "mean_fingertip_error_m": float(np.mean(pf["fingertip_errors"])),
            "max_fingertip_error_m": float(np.max(pf["fingertip_errors"])),
            "per_finger_error_m": pf["fingertip_errors"],
            "mean_initial_projection_distance_m": float(np.mean(pf["initial_projection_distances"])),
            "loss_terms": pf["loss_terms"],
            "force_closure": pf["force_closure"],
        },
        "contact_heatmap": {
            "success": hm["success"],
            "message": hm["message"],
            "elapsed_seconds": hm["elapsed_seconds"],
            "action": hm["action"],
            "mean_fingertip_error_m": float(np.mean(hm["fingertip_errors"])),
            "max_fingertip_error_m": float(np.max(hm["fingertip_errors"])),
            "per_finger_error_m": hm["fingertip_errors"],
            "loss_terms": hm["loss_terms"],
            "heatmap": hm["heatmap"],
        },
    }
    (case_out / "retarget_three_baselines_result.json").write_text(
        json.dumps(as_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def write_summary_html(summary: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for item in summary:
        rel = Path(item["outputs"]["html"]).relative_to(out_path.parent)
        rows.append(
            "<tr>"
            f"<td><a href='{rel.as_posix()}'>{item['case']}</a></td>"
            f"<td>{item['position_only']['mean_fingertip_error_m'] * 1000.0:.1f}</td>"
            f"<td>{item['position_force']['mean_fingertip_error_m'] * 1000.0:.1f}</td>"
            f"<td>{item['position_force']['mean_initial_projection_distance_m'] * 1000.0:.1f}</td>"
            f"<td>{item['position_force']['force_closure']['is_force_closure']}</td>"
            f"<td>{item['contact_heatmap']['mean_fingertip_error_m'] * 1000.0:.1f}</td>"
            f"<td>{item['contact_heatmap']['heatmap']['mse']:.4f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Kettle Retargeting Baselines</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #d1d5db; padding: 6px 9px; }}
th {{ background: #f3f4f6; }}
</style></head><body>
<h1>Kettle Retargeting Baselines</h1>
<p>Errors are mean fingertip distances in millimeters unless noted.</p>
<table>
<tr><th>case</th><th>position-only</th><th>position-force</th><th>PF init object proj.</th><th>PF force closure</th><th>heatmap</th><th>heatmap MSE</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test")
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=REPO_ROOT
        / "outputs/0713test_phase1_4_vlm_qwen37_test/_pose_foundationpose_assets/kettle-decimated-100k-scale0.2.obj",
    )
    parser.add_argument(
        "--retarget-output",
        type=Path,
        default=REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test/kettle_retarget_three_baselines",
    )
    parser.add_argument("--robot-profile", default="folding_hand_right", choices=sorted(ROBOT_PROFILES))
    parser.add_argument("--position-restarts", type=int, default=3)
    parser.add_argument("--position-max-nfev", type=int, default=90)
    parser.add_argument("--num-object-samples", type=int, default=256)
    parser.add_argument("--num-mano-samples", type=int, default=700)
    parser.add_argument("--num-robot-samples", type=int, default=650)
    parser.add_argument("--contact-top-k", type=int, default=12)
    parser.add_argument("--contact-temperature", type=float, default=8e-5)
    parser.add_argument("--force-maxiter", type=int, default=35)
    parser.add_argument("--heatmap-maxiter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hamer-subdir", default="pose_hamer_official")
    parser.add_argument("--fallback-hamer-subdir", default="pose_hamer")
    parser.add_argument(
        "--hand-scene-alignment",
        choices=["none", "z_center", "object_center"],
        default="object_center",
        help=(
            "Fallback diagnostic alignment used only when phase5.5 alignment is disabled or unavailable. "
            "'object_center' preserves hand shape/pose but translates the hand mesh and keypoints near the object."
        ),
    )
    parser.add_argument("--alignment-root", type=Path, default=None)
    parser.add_argument("--alignment-method", choices=["translation", "se3", "rgbd"], default="se3")
    parser.add_argument(
        "--disable-phase55",
        action="store_true",
        help="Do not auto-use phase5.5 alignment; fall back to --hand-scene-alignment.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.disable_phase55 and args.alignment_root is None:
        suffix = args.hamer_subdir.replace("pose_", "")
        candidates = [
            args.output_root / f"kettle_phase5_5_alignment_{suffix}",
            args.output_root / "kettle_phase5_5_alignment_official",
            args.output_root / "kettle_phase5_5_alignment",
        ]
        args.alignment_root = next((path for path in candidates if path.exists()), None)
    if args.disable_phase55:
        args.alignment_method = ""

    cases = collect_cases(args.output_root, args.hamer_subdir, args.fallback_hamer_subdir)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No complete kettle pose/hand cases found.")
    args.retarget_output.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case['sample_id']} {case['lollipop']}", flush=True)
        try:
            payload = run_case(case, args.object_mesh, args.retarget_output, args.robot_profile, args)
            summary.append(payload)
        except Exception as exc:  # keep batch progress visible for rough-pose pilot data
            failed = {
                "case": f"{case['sample_id']}_{case['lollipop']}",
                "sample_id": case["sample_id"],
                "lollipop": case["lollipop"],
                "status": "failed",
                "error": str(exc),
            }
            print(f"  failed: {exc}", flush=True)
            summary.append(failed)

    summary_file = args.retarget_output / "summary.json"
    summary_file.write_text(json.dumps(as_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    ok_summary = [item for item in summary if item.get("status") != "failed"]
    if ok_summary:
        write_summary_html(ok_summary, args.retarget_output / "summary.html")
    print(f"Wrote {summary_file}", flush=True)


if __name__ == "__main__":
    main()
