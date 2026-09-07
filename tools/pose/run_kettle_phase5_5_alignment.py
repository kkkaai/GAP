#!/usr/bin/env python3
"""Test phase5.5 hand-object alignment on generated kettle cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.phases.phase5_5_hand_object_alignment import (  # noqa: E402
    Phase55HandObjectAlignmentConfig,
    align_hand_to_object,
    evaluate_hand_object_alignment,
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
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def first_existing(paths: list[Path]) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def collect_cases(output_root: Path, hamer_subdir: str, fallback_hamer_subdir: str) -> list[dict[str, Any]]:
    cases = []
    for sample_id in SAMPLE_IDS:
        for ldir in sorted((output_root / sample_id).glob("lollipop_*")):
            hamer_dir = ldir / hamer_subdir
            if not (hamer_dir / "hand_00_camera.obj").exists() and fallback_hamer_subdir:
                fallback_dir = ldir / fallback_hamer_subdir
                if (fallback_dir / "hand_00_camera.obj").exists():
                    hamer_dir = fallback_dir
            hand_mesh = hamer_dir / "hand_00_camera.obj"
            hand_vertices = hamer_dir / "hand_00_vertices_camera.npy"
            hand_keypoints = hamer_dir / "hand_00_keypoints_camera.npy"
            pose = ldir / "pose_foundationpose" / "object_in_camera.txt"
            if hand_mesh.exists() and hand_vertices.exists() and hand_keypoints.exists() and pose.exists():
                source_id = sample_id
                cases.append(
                    {
                        "sample_id": sample_id,
                        "lollipop": ldir.name,
                        "dir": ldir,
                        "hand_mesh": hand_mesh,
                        "hand_vertices": hand_vertices,
                        "hand_keypoints": hand_keypoints,
                        "hamer_dir": hamer_dir,
                        "hamer_subdir": hamer_dir.name,
                        "object_pose": pose,
                        "depth": REPO_ROOT / "0713test" / source_id / "depth_meters.npy",
                        "hand_mask": first_existing(
                            [
                                ldir / "pose_hamer_official" / "sam3_1_text_hand" / "sam3_1_text_hand_mask.png",
                                ldir / "pose_hamer_official" / "sam3_1_text_hand_mask.png",
                                ldir / "pose_hamer_official" / "sam2_hand_mask.png",
                                ldir / "pose_hamer_official" / "official_hand_bbox_mask.png",
                                ldir / "lollipop_mask.png",
                            ]
                        ),
                    }
                )
    return cases


def load_scene_object(mesh_path: Path, pose_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pose = np.loadtxt(pose_path, dtype=np.float64)
    vertices_h = np.concatenate([np.asarray(mesh.vertices), np.ones((len(mesh.vertices), 1))], axis=1)
    vertices = (pose @ vertices_h.T).T[:, :3]
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def mesh_trace(name: str, mesh: trimesh.Trimesh, color: str, opacity: float) -> go.Mesh3d:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(
        name=name,
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
    )


def point_trace(name: str, points: np.ndarray, color: str, size: int = 3) -> go.Scatter3d:
    points = np.asarray(points, dtype=np.float64)
    return go.Scatter3d(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color),
    )


def write_alignment_html(
    out_file: Path,
    case_name: str,
    object_mesh: trimesh.Trimesh,
    aligned: dict[str, trimesh.Trimesh],
    keypoints: dict[str, np.ndarray],
) -> None:
    methods = ["translation", "se3", "rgbd"]
    colors = {"translation": "#2563eb", "se3": "#dc2626", "rgbd": "#16a34a"}
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("translation", "se3 contact", "rgbd/fallback"),
    )
    all_points = [np.asarray(object_mesh.vertices)]
    for col, method in enumerate(methods, start=1):
        fig.add_trace(mesh_trace("object", object_mesh, "#9ca3af", 0.30), row=1, col=col)
        fig.add_trace(mesh_trace(method, aligned[method], colors[method], 0.42), row=1, col=col)
        tips = keypoints[method][[4, 8, 12, 16, 20]]
        fig.add_trace(point_trace(f"{method} fingertips", tips, "#111827", 4), row=1, col=col)
        all_points.append(np.asarray(aligned[method].vertices))
    points = np.vstack([p[:: max(1, len(p) // 3000)] for p in all_points])
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = float(np.max(points.max(axis=0) - points.min(axis=0)) * 0.58 + 0.03)
    scene = dict(
        xaxis=dict(range=[center[0] - radius, center[0] + radius], visible=False),
        yaxis=dict(range=[center[1] - radius, center[1] + radius], visible=False),
        zaxis=dict(range=[center[2] - radius, center[2] + radius], visible=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.25, y=-1.45, z=0.90), up=dict(x=0, y=0, z=1)),
    )
    fig.update_layout(
        title=f"Phase5.5 hand-object alignment: {case_name}",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=False,
        paper_bgcolor="white",
        scene=scene,
        scene2=scene,
        scene3=scene,
    )
    out_file.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def run_case(
    case: dict[str, Any],
    object_mesh_path: Path,
    output_root: Path,
    camera_k: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_name = f"{case['sample_id']}_{case['lollipop']}"
    out_dir = output_root / case["sample_id"] / case["lollipop"]
    out_dir.mkdir(parents=True, exist_ok=True)

    object_mesh = load_scene_object(object_mesh_path, case["object_pose"])
    raw_hand_mesh = trimesh.load_mesh(str(case["hand_mesh"]), process=False)
    hand_vertices = np.load(case["hand_vertices"]).astype(np.float64)
    hand_keypoints = np.load(case["hand_keypoints"]).astype(np.float64)
    hand_faces = np.asarray(raw_hand_mesh.faces, dtype=np.int64)

    depth = np.load(case["depth"]).astype(np.float64) if Path(case["depth"]).exists() else None
    mask = cv2.imread(str(case["hand_mask"]), cv2.IMREAD_GRAYSCALE) if Path(case["hand_mask"]).exists() else None

    raw_metrics = evaluate_hand_object_alignment(
        hand_vertices,
        hand_faces,
        hand_keypoints,
        np.asarray(object_mesh.vertices),
        np.asarray(object_mesh.faces),
        num_hand_samples=args.num_hand_samples,
        seed=args.seed,
    )
    results = {}
    aligned_meshes = {}
    aligned_keypoints = {}
    for method in ("translation", "se3", "rgbd"):
        cfg = Phase55HandObjectAlignmentConfig(
            method=method,
            num_hand_samples=args.num_hand_samples,
            se3_maxiter=args.se3_maxiter,
            random_seed=args.seed,
        )
        result = align_hand_to_object(
            hand_vertices,
            hand_faces,
            hand_keypoints,
            np.asarray(object_mesh.vertices),
            np.asarray(object_mesh.faces),
            config=cfg,
            camera_k=camera_k,
            depth_m=depth,
            hand_mask=mask,
        )
        method_dir = out_dir / method
        method_dir.mkdir(exist_ok=True)
        mesh = trimesh.Trimesh(vertices=result.vertices, faces=hand_faces, process=False)
        mesh.export(method_dir / "aligned_hand.obj")
        np.save(method_dir / "aligned_vertices.npy", result.vertices)
        np.save(method_dir / "aligned_keypoints.npy", result.keypoints_3d)
        np.savetxt(method_dir / "hand_to_metric_camera.txt", result.transform)
        results[method] = {
            "status": result.status,
            "message": result.message,
            "transform": result.transform,
            "metrics": result.metrics,
            "metadata": result.metadata,
            "files": {
                "aligned_hand_obj": method_dir / "aligned_hand.obj",
                "aligned_vertices": method_dir / "aligned_vertices.npy",
                "aligned_keypoints": method_dir / "aligned_keypoints.npy",
                "transform": method_dir / "hand_to_metric_camera.txt",
            },
        }
        aligned_meshes[method] = mesh
        aligned_keypoints[method] = result.keypoints_3d

    object_mesh.export(out_dir / "object_scene.obj")
    write_alignment_html(out_dir / "phase5_5_alignment.html", case_name, object_mesh, aligned_meshes, aligned_keypoints)
    payload = {
        "case": case_name,
        "sample_id": case["sample_id"],
        "lollipop": case["lollipop"],
        "inputs": {
            "hand_mesh": case["hand_mesh"],
            "hand_keypoints": case["hand_keypoints"],
            "hamer_dir": case["hamer_dir"],
            "hamer_subdir": case["hamer_subdir"],
            "object_pose": case["object_pose"],
            "object_mesh": object_mesh_path,
            "depth": case["depth"],
            "hand_mask": case["hand_mask"],
            "camera_k": camera_k,
        },
        "outputs": {
            "case_dir": out_dir,
            "html": out_dir / "phase5_5_alignment.html",
        },
        "raw_metrics": raw_metrics,
        "methods": results,
    }
    (out_dir / "phase5_5_alignment_result.json").write_text(
        json.dumps(as_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def write_summary(summary: list[dict[str, Any]], out_root: Path) -> None:
    (out_root / "summary.json").write_text(json.dumps(as_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for item in summary:
        rel = Path(item["outputs"]["html"]).relative_to(out_root)
        raw = item["raw_metrics"]
        cells = [
            f"<td><a href='{rel.as_posix()}'>{item['case']}</a></td>",
            f"<td>{raw['hand_object_center_distance_m']:.3f}</td>",
        ]
        for method in ("translation", "se3", "rgbd"):
            m = item["methods"][method]["metrics"]
            meta = item["methods"][method]["metadata"]
            cells.append(f"<td>{m['mean_surface_distance_m'] * 1000:.1f}</td>")
            cells.append(f"<td>{m['contact_sample_fraction']:.3f}</td>")
            cells.append(f"<td>{m['penetration_sample_fraction']:.3f}</td>")
            if method == "rgbd":
                cells.append(f"<td>{meta.get('rgbd_status', 'n/a')}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase5.5 Alignment Summary</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
td, th {{ border: 1px solid #d1d5db; padding: 5px 7px; }}
th {{ background: #f3f4f6; }}
</style></head><body>
<h1>Phase5.5 Hand-Object Alignment</h1>
<p>Distances are in meters for raw center distance and millimeters for mean surface distance.</p>
<table>
<tr>
<th>case</th><th>raw center dist</th>
<th>translation dist</th><th>translation contact</th><th>translation pen</th>
<th>se3 dist</th><th>se3 contact</th><th>se3 pen</th>
<th>rgbd dist</th><th>rgbd contact</th><th>rgbd pen</th><th>rgbd status</th>
</tr>
{''.join(rows)}
</table>
</body></html>"""
    (out_root / "summary.html").write_text(html, encoding="utf-8")


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
        "--alignment-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--hamer-subdir", default="pose_hamer_official")
    parser.add_argument("--fallback-hamer-subdir", default="pose_hamer")
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--num-hand-samples", type=int, default=900)
    parser.add_argument("--se3-maxiter", type=int, default=90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alignment_output is None:
        suffix = args.hamer_subdir.replace("pose_", "")
        args.alignment_output = args.output_root / f"kettle_phase5_5_alignment_{suffix}"
    args.alignment_output.mkdir(parents=True, exist_ok=True)
    camera_k = np.asarray([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cases = collect_cases(args.output_root, args.hamer_subdir, args.fallback_hamer_subdir)
    if args.limit > 0:
        cases = cases[: args.limit]
    summary = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases):02d}] {case['sample_id']} {case['lollipop']}", flush=True)
        summary.append(run_case(case, args.object_mesh, args.alignment_output, camera_k, args))
    write_summary(summary, args.alignment_output)
    print(f"Wrote {args.alignment_output / 'summary.json'}", flush=True)
    print(f"Wrote {args.alignment_output / 'summary.html'}", flush=True)


if __name__ == "__main__":
    main()
