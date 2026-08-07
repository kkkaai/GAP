#!/usr/bin/env python3
"""Run contact-patch + soft-surface retargeting for one HOI4D frame."""

from __future__ import annotations

import argparse
import json
import shutil
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

for path in (VIS_ROOT, RETARGET_SRC, REPO_ROOT / "tools" / "retarget"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from position import (  # noqa: E402
    ROBOT_MESH_MAX_FACES,
    marker_trace,
    mesh_trace,
    point_trace,
    prepare_shared_case,
    load_retargeted_robot_mesh,
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
    load_hand,
    load_obj,
    load_objpose,
    transform_object,
    try_make_mano_scene,
)

from prosthetic_grasp.geometry import (  # noqa: E402
    RetargetOptimizationConfig,
    assign_patches_to_robot_surface,
    build_contact_patch_targets,
    build_hand_frame,
    canonicalize_patch_targets,
    extract_mano_object_contact_clusters,
    load_robot_surface_model,
    map_canonical_mano_patches_to_robot_frame,
    materialize_assigned_robot_contacts,
    optimize_retarget_action,
    retarget_loss_terms,
    robot_tip_from_link,
)


ROBOT_PROFILES = {
    "shadow_hand": {
        "model_dir": RETARGET_ROOT / "hand" / "shadow_hand",
        "model_format": "urdf",
        "urdf_path": RETARGET_ROOT / "hand" / "shadow_hand" / "shadowhand.urdf",
        "wrist_link": "robot0:palm",
        "index_tip_link": "robot0:ffdistal",
        "middle_tip_link": "robot0:mfdistal",
        "little_tip_link": "robot0:lfdistal",
        "fingertip_links": [
            "robot0:thdistal",
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
        ],
    },
    "folding_hand_right": {
        "model_dir": RETARGET_ROOT / "hand" / "folding_hand_right",
        "model_format": "xml",
        "xml_path": RETARGET_ROOT / "hand" / "folding_hand_right" / "folding_hand_right.xml",
        "wrist_link": "base_link",
        "index_tip_link": "ff_3",
        "middle_tip_link": "mf_2",
        "little_tip_link": "lf_3",
        "fingertip_links": ["th_4", "ff_3", "mf_2", "rf_3", "lf_3"],
    },
    "inspire_hand": {
        "model_dir": RETARGET_ROOT / "hand" / "inspire_hand_ftp" / "urdf",
        "model_format": "urdf",
        "urdf_path": RETARGET_ROOT / "hand" / "inspire_hand_ftp" / "urdf" / "inspire_right.urdf",
        "wrist_link": "right_base_link",
        "index_tip_link": "right_index_2",
        "middle_tip_link": "right_middle_2",
        "little_tip_link": "right_little_2",
        "fingertip_links": [
            "right_thumb_4",
            "right_index_2",
            "right_middle_2",
            "right_ring_2",
            "right_little_2",
        ],
    },
}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def bounds_for_points(points: np.ndarray, margin: float = 0.12) -> tuple[list[float], list[float], list[float]]:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5 + margin)
    if radius <= 1e-9:
        radius = margin
    return (
        [float(center[0] - radius), float(center[0] + radius)],
        [float(center[1] - radius), float(center[1] + radius)],
        [float(center[2] - radius), float(center[2] + radius)],
    )


def load_hoi4d_scene(sequence: Path, frame: str, cad: Path, hand_path: Path, mano_root: Path):
    hand = load_hand(hand_path)
    mano = try_make_mano_scene(hand, hand_path, mano_root)
    if mano is None:
        raise RuntimeError("Could not build MANO mesh for contact retargeting.")
    objpose = load_objpose(sequence / "objpose" / f"{int(frame)}.json")
    obj_vertices, obj_faces = load_obj(cad)
    object_vertices = transform_object(obj_vertices, objpose)
    return hand, mano, object_vertices, obj_faces


def make_zero_mano(hand: dict, hand_path: Path, mano_root: Path):
    zero = dict(hand)
    zero["poseCoeff"] = np.zeros_like(np.asarray(hand["poseCoeff"], dtype=np.float32))
    zero["trans"] = np.zeros(3, dtype=np.float32)
    mano = try_make_mano_scene(zero, hand_path, mano_root)
    if mano is None:
        raise RuntimeError("Could not build zero MANO mesh.")
    return mano


class RigidFrameTransform:
    def __init__(self, wrist: np.ndarray, scene_frame: np.ndarray, robot_frame: np.ndarray) -> None:
        self.wrist = np.asarray(wrist, dtype=np.float64)
        self.scene_frame = np.asarray(scene_frame, dtype=np.float64)
        self.robot_frame = np.asarray(robot_frame, dtype=np.float64)

    def points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return ((points - self.wrist) @ self.scene_frame) @ self.robot_frame.T

    def normals(self, normals: np.ndarray) -> np.ndarray:
        normals = np.asarray(normals, dtype=np.float64)
        return normalize_rows((normals @ self.scene_frame) @ self.robot_frame.T)

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (points @ self.robot_frame) @ self.scene_frame.T + self.wrist

    def inverse_normals(self, normals: np.ndarray) -> np.ndarray:
        normals = np.asarray(normals, dtype=np.float64)
        return normalize_rows((normals @ self.robot_frame) @ self.scene_frame.T)


def build_scene_to_robot_rigid_transform(mano_keypoints: np.ndarray, robot_zero_surface: Any, profile: dict[str, Any]):
    robot_points = np.asarray(robot_zero_surface.points, dtype=np.float64)
    robot_links = np.asarray(robot_zero_surface.link_names).astype(str)
    scene_frame = build_hand_frame(
        wrist=mano_keypoints[0],
        index_mcp=mano_keypoints[5],
        middle_mcp=mano_keypoints[9],
        little_mcp=mano_keypoints[17],
    )
    robot_frame = build_hand_frame(
        wrist=np.zeros(3, dtype=np.float64),
        index_mcp=robot_tip_from_link(robot_points, robot_links, profile["index_tip_link"]),
        middle_mcp=robot_tip_from_link(robot_points, robot_links, profile["middle_tip_link"]),
        little_mcp=robot_tip_from_link(robot_points, robot_links, profile["little_tip_link"]),
    )
    return RigidFrameTransform(mano_keypoints[0], scene_frame, robot_frame)


def transform_patch_object_targets_to_robot(patches: list[Any], transform: RigidFrameTransform) -> None:
    for patch in patches:
        patch.object_point_target = transform.points(patch.object_point_target[None, :])[0]
        patch.object_normal_target = transform.normals(patch.object_normal_target[None, :])[0]
        patch.object_point_nearest = transform.points(patch.object_point_nearest[None, :])[0]
        patch.object_normal_nearest = transform.normals(patch.object_normal_nearest[None, :])[0]
        patch.object_point_center = transform.points(patch.object_point_center[None, :])[0]
        patch.object_normal_center = transform.normals(patch.object_normal_center[None, :])[0]


def load_robot_model(profile_name: str):
    profile = ROBOT_PROFILES[profile_name]
    kwargs: dict[str, Any] = {"wrist_link": profile["wrist_link"], "model_format": profile["model_format"]}
    if profile["model_format"] == "urdf":
        kwargs["urdf_path"] = profile["urdf_path"]
    else:
        kwargs["xml_path"] = profile["xml_path"]
    return load_robot_surface_model(profile["model_dir"], **kwargs)


def run_contact_surface(
    sequence: Path,
    frame: str,
    cad: Path,
    hand_path: Path,
    mano_root: Path,
    robot_profile: str,
    *,
    num_mano_samples: int,
    num_robot_samples: int,
    contact_threshold: float,
    cluster_radius: float,
    min_cluster_size: int,
    assignment_top_k: int,
    temperature: float,
    maxiter: int,
) -> dict[str, Any]:
    if robot_profile not in ROBOT_PROFILES:
        raise ValueError(f"Unsupported robot_profile {robot_profile!r}.")

    hand, mano, object_vertices, object_faces = load_hoi4d_scene(sequence, frame, cad, hand_path, mano_root)
    contact = extract_mano_object_contact_clusters(
        mano.vertices,
        mano.faces,
        object_vertices,
        object_faces,
        num_mano_samples=num_mano_samples,
        contact_threshold=contact_threshold,
        cluster_radius=cluster_radius,
        min_cluster_size=min_cluster_size,
        cluster_space="object",
        seed=7,
        use_farthest_point_sampling=True,
        oversample_factor=16,
    )
    patches = build_contact_patch_targets(contact, representative="center")
    if not patches:
        raise RuntimeError(
            f"No contact patches extracted: samples={len(contact.contact_sample_indices)}, "
            f"threshold={contact_threshold}."
        )

    zero_mano = make_zero_mano(hand, hand_path, mano_root)
    canonicalize_patch_targets(patches, zero_mano.vertices, zero_mano.faces)

    robot_model = load_robot_model(robot_profile)
    profile = ROBOT_PROFILES[robot_profile]
    robot_topology = robot_model.sample_surface_topology(
        num_points=num_robot_samples,
        seed=7,
        use_farthest_point_sampling=True,
    )
    robot_zero_surface = robot_model.materialize_surface(robot_topology, robot_model.zero_action)

    canonical_mapping = map_canonical_mano_patches_to_robot_frame(
        patches,
        zero_mano.keypoints,
        robot_zero_surface,
        index_tip_link=profile["index_tip_link"],
        middle_tip_link=profile["middle_tip_link"],
        little_tip_link=profile["little_tip_link"],
        min_scale=1.0,
        max_scale=1.0,
    )

    scene_to_robot = build_scene_to_robot_rigid_transform(mano.keypoints, robot_zero_surface, profile)
    object_vertices_robot = scene_to_robot.points(object_vertices)
    transform_patch_object_targets_to_robot(patches, scene_to_robot)

    assignment = assign_patches_to_robot_surface(
        patches,
        robot_zero_surface,
        method="soft_surface",
        top_k=assignment_top_k,
        temperature=temperature,
    )

    config = RetargetOptimizationConfig(
        contact_weight=1.0,
        normal_weight=0.0,
        fc_weight=0.0,
        regularization_weight=1e-4,
        maxiter_stage1=maxiter,
        maxiter_stage2=0,
        object_center=object_vertices_robot.mean(axis=0),
    )
    initial_terms = retarget_loss_terms(robot_model.zero_action, robot_model, robot_topology, patches, config)
    start = time.perf_counter()
    result = optimize_retarget_action(robot_model, robot_topology, patches, config=config)
    elapsed = time.perf_counter() - start
    best_surface = robot_model.materialize_surface(robot_topology, result.best.action)
    robot_contacts, robot_normals = materialize_assigned_robot_contacts(patches, best_surface)
    object_contacts = np.asarray([patch.object_point_target for patch in patches], dtype=np.float64)
    contact_errors = np.linalg.norm(robot_contacts - object_contacts, axis=1)
    mano_fingertips_scene = mano.keypoints[[4, 8, 12, 16, 20]]
    robot_fingertips = np.asarray(
        [
            robot_tip_from_link(best_surface.points, best_surface.link_names.astype(str), link_name)
            for link_name in profile["fingertip_links"]
        ],
        dtype=np.float64,
    )
    robot_fingertips_scene = scene_to_robot.inverse_points(robot_fingertips)
    robot_surface_scene = scene_to_robot.inverse_points(best_surface.points)
    robot_contacts_scene = scene_to_robot.inverse_points(robot_contacts)
    object_contacts_scene = scene_to_robot.inverse_points(object_contacts)
    robot_wrist_scene = scene_to_robot.inverse_points(np.zeros((1, 3), dtype=np.float64))[0]
    robot_mesh = load_retargeted_robot_mesh(
        SimpleNamespace(action=result.best.action, metadata={"robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")), "robot_urdf_path": str(profile.get("urdf_path") or profile.get("xml_path")), "model_format": profile["model_format"], "wrist_link": profile["wrist_link"]})
    )
    robot_mesh_vertices_scene = None
    robot_mesh_faces = None
    if robot_mesh is not None:
        robot_mesh_vertices, robot_mesh_faces = robot_mesh
        robot_mesh_vertices, robot_mesh_faces = simplify_mesh_for_html(robot_mesh_vertices, robot_mesh_faces)
        robot_mesh_vertices_scene = scene_to_robot.inverse_points(robot_mesh_vertices)

    return {
        "status": "ok",
        "route": "contact_patch_soft_surface",
        "robot_profile": robot_profile,
        "action_names": list(robot_model.action_names),
        "action": result.best.action.astype(float).tolist(),
        "num_mano_samples": int(num_mano_samples),
        "num_robot_samples": int(num_robot_samples),
        "contact_threshold": float(contact_threshold),
        "cluster_radius": float(cluster_radius),
        "num_contact_samples": int(len(contact.contact_sample_indices)),
        "num_contact_patches": int(len(patches)),
        "patch_sizes": [int(patch.patch_size) for patch in patches],
        "assignment": {
            "scheme": assignment.scheme,
            "top_k": int(assignment_top_k),
            "temperature": float(temperature),
            "mean_distance": float(assignment.metrics["mean_distance"]),
            "max_distance": float(assignment.metrics["max_distance"]),
            "assigned_link_names": assignment.assigned_link_names.astype(str).tolist()
            if assignment.assigned_link_names is not None
            else [],
        },
        "canonical_mapping": {
            "scale": float(canonical_mapping.scale),
            "mano_mid_length": float(canonical_mapping.mano_mid_length),
            "robot_mid_length": float(canonical_mapping.robot_mid_length),
        },
        "optimization_seconds": float(elapsed),
        "optimization": {
            "success": bool(result.best.success),
            "message": result.best.message,
            "iterations": int(result.best.iterations),
            "initial_loss_terms": {key: float(value) for key, value in initial_terms.items()},
            "best_loss_terms": {key: float(value) for key, value in result.best.loss_terms.items()},
        },
        "contact_error": {
            "mean_m": float(np.mean(contact_errors)),
            "max_m": float(np.max(contact_errors)),
            "per_patch_m": contact_errors.astype(float).tolist(),
        },
        "arrays": {
            "mano_vertices": mano.vertices.astype(float).tolist(),
            "mano_faces": mano.faces.astype(int).tolist(),
            "object_vertices_scene": object_vertices.astype(float).tolist(),
            "object_faces": object_faces.astype(int).tolist(),
            "object_vertices_robot": object_vertices_robot.astype(float).tolist(),
            "robot_surface_points": best_surface.points.astype(float).tolist(),
            "robot_surface_normals": best_surface.normals.astype(float).tolist(),
            "robot_surface_points_scene": robot_surface_scene.astype(float).tolist(),
            "robot_contacts": robot_contacts.astype(float).tolist(),
            "robot_contact_normals": robot_normals.astype(float).tolist(),
            "robot_contacts_scene": robot_contacts_scene.astype(float).tolist(),
            "object_contacts": object_contacts.astype(float).tolist(),
            "object_contacts_scene": object_contacts_scene.astype(float).tolist(),
            "mano_fingertips_scene": mano_fingertips_scene.astype(float).tolist(),
            "robot_fingertips_scene": robot_fingertips_scene.astype(float).tolist(),
            "robot_wrist_scene": robot_wrist_scene.astype(float).tolist(),
            "robot_mesh_vertices_scene": robot_mesh_vertices_scene.astype(float).tolist()
            if robot_mesh_vertices_scene is not None
            else [],
            "robot_mesh_faces": robot_mesh_faces.astype(int).tolist() if robot_mesh_faces is not None else [],
        },
        "metadata": {
            "model_format": profile["model_format"],
            "robot_model_path": str(profile.get("urdf_path") or profile.get("xml_path")),
            "wrist_link": profile["wrist_link"],
            "visual_alignment": "rigid_only_no_scale",
            "robot_mesh_max_faces": int(ROBOT_MESH_MAX_FACES),
        },
    }


def write_contact_html(case_dir: Path, rgb: Path, base_meta: dict[str, Any], payload: dict[str, Any]) -> Path:
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
    object_contacts_scene = np.asarray(arrays["object_contacts_scene"], dtype=np.float64)
    robot_contacts_scene = np.asarray(arrays["robot_contacts_scene"], dtype=np.float64)
    mano_fingertips_scene = np.asarray(arrays["mano_fingertips_scene"], dtype=np.float64)
    robot_fingertips_scene = np.asarray(arrays["robot_fingertips_scene"], dtype=np.float64)
    robot_wrist_scene = np.asarray(arrays["robot_wrist_scene"], dtype=np.float64)
    robot_mesh_vertices = np.asarray(arrays.get("robot_mesh_vertices_scene") or [], dtype=np.float64)
    robot_mesh_faces = np.asarray(arrays.get("robot_mesh_faces") or [], dtype=np.int64)
    if robot_mesh_vertices.size and robot_mesh_faces.size:
        fig.add_trace(
            mesh_trace(
                f"{payload['robot_profile']} mesh",
                robot_mesh_vertices,
                robot_mesh_faces,
                "#2563eb",
                0.28,
            )
        )
        vertex_step = max(len(robot_mesh_vertices) // 2500, 1)
        fig.add_trace(
            marker_trace(
                f"{payload['robot_profile']} mesh vertices",
                robot_mesh_vertices[::vertex_step],
                "#1d4ed8",
                2,
            )
        )
    fig.add_trace(marker_trace(f"{payload['robot_profile']} surface samples", robot_points_scene, "#1d4ed8", 3))
    fig.add_trace(marker_trace("MANO fingertips", mano_fingertips_scene, "#16a34a", 7))
    fig.add_trace(marker_trace(f"{payload['robot_profile']} fingertips", robot_fingertips_scene, "#0f4cbd", 7))
    fig.add_trace(marker_trace("object contact patches", object_contacts_scene, "#d9480f", 6))
    fig.add_trace(marker_trace(f"{payload['robot_profile']} soft contacts", robot_contacts_scene, "#1864ab", 6))
    fig.add_trace(segment_trace("contact error", object_contacts_scene, robot_contacts_scene, "#868e96"))
    fig.add_trace(point_trace("robot wrist", robot_wrist_scene, "#1d4ed8", 8, "diamond"))
    view_points = [
        robot_points_scene,
        object_contacts_scene,
        robot_contacts_scene,
        mano_fingertips_scene,
        robot_fingertips_scene,
        robot_wrist_scene.reshape(1, 3),
    ]
    if robot_mesh_vertices.size:
        view_points.append(robot_mesh_vertices[:: max(len(robot_mesh_vertices) // 5000, 1)])
    x_range, y_range, z_range = bounds_for_points(np.vstack(view_points), margin=0.18)
    fig.update_layout(scene=dict(xaxis=dict(range=x_range, visible=False), yaxis=dict(range=y_range, visible=False), zaxis=dict(range=z_range, visible=False), aspectmode="cube"))

    errors = {
        "route": payload["route"],
        "robot hand": payload["robot_profile"],
        "assignment": payload["assignment"]["scheme"],
        "visual alignment": "rigid only, no scale",
        "contact patches": payload["num_contact_patches"],
        "contact samples": payload["num_contact_samples"],
        "optimization time": f"{payload['optimization_seconds']:.3f} s",
        "mean contact error": f"{payload['contact_error']['mean_m'] * 1000.0:.2f} mm",
        "max contact error": f"{payload['contact_error']['max_m'] * 1000.0:.2f} mm",
        "status": payload["status"],
    }
    scene_meta.update(base_meta)
    scene_meta["errors"] = errors
    scene_meta["contact_surface"] = {
        key: payload[key]
        for key in (
            "route",
            "robot_profile",
            "num_contact_samples",
            "num_contact_patches",
            "assignment",
            "canonical_mapping",
            "optimization_seconds",
            "optimization",
            "contact_error",
            "metadata",
        )
    }
    html = case_dir / f"scene_{payload['robot_profile']}_contact_surface.html"
    write_html(fig, scene_meta, rgb, html)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "runs")
    parser.add_argument("--case-id", default="bottle_frame74_contact_surface")
    parser.add_argument("--robot-profile", choices=sorted(ROBOT_PROFILES), default="shadow_hand")
    parser.add_argument("--num-mano-samples", type=int, default=2500)
    parser.add_argument("--num-robot-samples", type=int, default=1800)
    parser.add_argument("--contact-threshold", type=float, default=0.012)
    parser.add_argument("--cluster-radius", type=float, default=0.018)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--assignment-top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
    payload = run_contact_surface(
        args.sequence,
        str(args.frame),
        args.cad,
        args.hand,
        args.mano_root,
        args.robot_profile,
        num_mano_samples=args.num_mano_samples,
        num_robot_samples=args.num_robot_samples,
        contact_threshold=args.contact_threshold,
        cluster_radius=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
        assignment_top_k=args.assignment_top_k,
        temperature=args.temperature,
        maxiter=args.maxiter,
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
        "route": "contact_patch_soft_surface",
    }
    json_path = case_dir / f"retargeted_{args.robot_profile}_contact_surface.json"
    payload_out = {**base_meta, **{k: v for k, v in payload.items() if k != "arrays"}}
    json_path.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    html_path = write_contact_html(case_dir, rgb, base_meta, payload)
    print(f"Wrote {json_path}")
    print(f"Open: {html_path}")


if __name__ == "__main__":
    main()
