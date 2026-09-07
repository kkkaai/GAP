#!/usr/bin/env python3
"""Run Do-as-I-Do style hand-anchored pointmap alignment on kettle cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import trimesh
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.phases.phase5_5_hand_anchored_pointmap_alignment import (  # noqa: E402
    HandAnchoredPointmapAlignmentConfig,
    align_object_pose_with_hand_anchor,
    transform_object_vertices,
)


KETTLE_SAMPLE_IDS = [
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
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def load_intrinsics(path: Path, fallback: np.ndarray) -> np.ndarray:
    if not path.exists():
        return fallback.astype(np.float64)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    vals = [float(x) for x in path.read_text(encoding="utf-8").split()]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 9:
        return arr.reshape(3, 3)
    raise ValueError(f"Cannot parse intrinsics: {path}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8))
    image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def collect_cases(output_root: Path, hamer_subdir: str, hand_mask_name: str | None) -> list[dict[str, Path | str]]:
    cases: list[dict[str, Path | str]] = []
    for sample_id in KETTLE_SAMPLE_IDS:
        source_dir = REPO_ROOT / "0713test" / sample_id
        for lollipop_dir in sorted((output_root / sample_id).glob("lollipop_*")):
            pointmap_dir = lollipop_dir / "pointmap"
            pointmap = pointmap_dir / "phase4_inpaint_full_pointmap.npy"
            intr_txt = pointmap_dir / "phase4_inpaint_full_intrinsics.txt"
            intr_npy = pointmap_dir / "phase4_inpaint_full_intrinsics.npy"
            hamer_dir = lollipop_dir / hamer_subdir
            projected_hamer_mask = hamer_dir / "hand_00_projected_mask.png"
            hand_mask = projected_hamer_mask if projected_hamer_mask.exists() else lollipop_dir / "lollipop_mask.png"
            mask_source = "hamer_projected_mask" if projected_hamer_mask.exists() else "lollipop_mask_fallback"
            if hand_mask_name:
                candidate = lollipop_dir / hand_mask_name
                if not candidate.exists():
                    candidate = hamer_dir / hand_mask_name
                if candidate.exists():
                    hand_mask = candidate
                    mask_source = hand_mask_name
            required = [
                lollipop_dir / "phase4_inpaint_full.png",
                pointmap,
                hamer_dir / "hand_00_vertices_camera.npy",
                hamer_dir / "hand_00_camera.obj",
                lollipop_dir / "pose_foundationpose" / "object_in_camera.txt",
                source_dir / "object_mask.png",
                hand_mask,
            ]
            if all(p.exists() for p in required):
                cases.append(
                    {
                        "sample_id": sample_id,
                        "lollipop": lollipop_dir.name,
                        "lollipop_dir": lollipop_dir,
                        "rgb": lollipop_dir / "phase4_inpaint_full.png",
                        "pointmap": pointmap,
                        "intrinsics": intr_txt if intr_txt.exists() else intr_npy,
                        "hand_vertices": hamer_dir / "hand_00_vertices_camera.npy",
                        "hand_mesh": hamer_dir / "hand_00_camera.obj",
                        "object_pose": lollipop_dir / "pose_foundationpose" / "object_in_camera.txt",
                        "object_mask": source_dir / "object_mask.png",
                        "hand_mask": hand_mask,
                        "hand_mask_source": mask_source,
                    }
                )
    return cases


def project_points(points: np.ndarray, k: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = z > 1e-8
    u = np.round(k[0, 0] * points[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.round(k[1, 1] * points[:, 1] / z + k[1, 2]).astype(np.int64)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def write_overlay(
    path: Path,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    hand_mask: np.ndarray,
    object_vertices_before: np.ndarray,
    object_vertices_after: np.ndarray,
    hand_vertices: np.ndarray,
    k: np.ndarray,
) -> None:
    h, w = rgb.shape[:2]
    vis = rgb.copy()
    for mask, color, alpha in ((object_mask > 0, (0, 180, 0), 0.25), (hand_mask > 0, (255, 140, 0), 0.22)):
        if mask.shape[:2] != (h, w):
            mask = resize_mask(mask.astype(np.uint8), w, h)
        overlay = np.zeros_like(vis)
        overlay[mask] = color
        blended = vis[mask].astype(np.float32) * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
        vis[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    image = Image.fromarray(vis)
    draw = ImageDraw.Draw(image)
    for points, color, step in (
        (object_vertices_before, (0, 0, 255), 8),
        (object_vertices_after, (255, 0, 0), 8),
        (hand_vertices, (255, 255, 0), 3),
    ):
        uu, vv, valid = project_points(points, k, w, h)
        idx = np.where(valid)[0][:: max(1, len(points) // 2000 * step)]
        for i in idx:
            x, y = int(uu[i]), int(vv[i])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    draw.text((10, 10), "blue=original object, red=optimized object, yellow=hand", fill=(255, 255, 255))
    image.save(path)


def write_html(path: Path, object_before: trimesh.Trimesh, object_after: trimesh.Trimesh, hand: trimesh.Trimesh) -> None:
    def mesh_trace(name: str, mesh: trimesh.Trimesh, color: str, opacity: float) -> go.Mesh3d:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        return go.Mesh3d(
            name=name, x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity, showscale=False
        )

    fig = go.Figure()
    fig.add_trace(mesh_trace("object_before", object_before, "#2563eb", 0.22))
    fig.add_trace(mesh_trace("object_after", object_after, "#dc2626", 0.34))
    fig.add_trace(mesh_trace("hand_anchor", hand, "#f59e0b", 0.42))
    pts = np.vstack([
        np.asarray(object_before.vertices)[:: max(1, len(object_before.vertices) // 2500)],
        np.asarray(object_after.vertices)[:: max(1, len(object_after.vertices) // 2500)],
        np.asarray(hand.vertices),
    ])
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = float(np.max(pts.max(axis=0) - pts.min(axis=0)) * 0.60 + 0.03)
    fig.update_layout(
        title="Hand-anchored pointmap alignment",
        scene=dict(
            xaxis=dict(range=[center[0] - radius, center[0] + radius], visible=False),
            yaxis=dict(range=[center[1] - radius, center[1] + radius], visible=False),
            zaxis=dict(range=[center[2] - radius, center[2] + radius], visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=42, b=0),
    )
    path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def run_case(case: dict[str, Path | str], args: argparse.Namespace, object_mesh: trimesh.Trimesh, fallback_k: np.ndarray) -> dict[str, Any]:
    lollipop_dir = Path(case["lollipop_dir"])
    out_dir = lollipop_dir / "phase5_5_hand_anchored_pointmap"
    out_dir.mkdir(exist_ok=True)

    rgb = read_rgb(Path(case["rgb"]))
    object_mask = read_mask(Path(case["object_mask"]))
    hand_mask = read_mask(Path(case["hand_mask"]))
    pointmap = np.load(Path(case["pointmap"])).astype(np.float64)
    k = load_intrinsics(Path(case["intrinsics"]), fallback_k)
    hand_vertices = np.load(Path(case["hand_vertices"])).astype(np.float64)
    hand_mesh = load_mesh(Path(case["hand_mesh"]))
    object_pose = np.loadtxt(Path(case["object_pose"]), dtype=np.float64)

    config = HandAnchoredPointmapAlignmentConfig(
        translation_mode=args.translation_mode,
        mesh_scale=args.mesh_scale,
        max_hand_rays=args.max_hand_rays,
        random_seed=args.seed,
    )
    result = align_object_pose_with_hand_anchor(
        object_vertices_local=np.asarray(object_mesh.vertices),
        object_faces=np.asarray(object_mesh.faces),
        object_in_camera=object_pose,
        hand_vertices_camera=hand_vertices,
        hand_faces=np.asarray(hand_mesh.faces),
        pointmap=pointmap,
        camera_k=k,
        hand_mask=hand_mask,
        object_mask=object_mask,
        config=config,
    )
    np.savetxt(out_dir / "object_in_camera_optimized.txt", result.object_in_camera_optimized)
    np.savetxt(out_dir / "object_in_camera_original.txt", object_pose)

    before_vertices = transform_object_vertices(np.asarray(object_mesh.vertices), object_pose, mesh_scale=args.mesh_scale)
    after_vertices = transform_object_vertices(
        np.asarray(object_mesh.vertices), result.object_in_camera_optimized, mesh_scale=args.mesh_scale
    )
    object_before = trimesh.Trimesh(vertices=before_vertices, faces=np.asarray(object_mesh.faces), process=False)
    object_after = trimesh.Trimesh(vertices=after_vertices, faces=np.asarray(object_mesh.faces), process=False)
    hand_scene = trimesh.Trimesh(vertices=hand_vertices, faces=np.asarray(hand_mesh.faces), process=False)
    object_before.export(out_dir / "object_original_scene.obj")
    object_after.export(out_dir / "object_optimized_scene.obj")
    hand_scene.export(out_dir / "hand_anchor_scene.obj")
    write_overlay(out_dir / "alignment_overlay_2d.png", rgb, object_mask, hand_mask, before_vertices, after_vertices, hand_vertices, k)
    write_html(out_dir / "alignment_3d.html", object_before, object_after, hand_scene)

    payload = {
        "status": result.status,
        "message": result.message,
        "sample_id": case["sample_id"],
        "lollipop": case["lollipop"],
        "inputs": {
            "rgb": case["rgb"],
            "pointmap": case["pointmap"],
            "intrinsics": case["intrinsics"],
            "hand_vertices": case["hand_vertices"],
            "hand_mesh": case["hand_mesh"],
            "hand_mask": case["hand_mask"],
            "hand_mask_source": case["hand_mask_source"],
            "object_mask": case["object_mask"],
            "object_pose": case["object_pose"],
            "object_mesh": args.mesh_file,
        },
        "outputs": {
            "object_in_camera_optimized": out_dir / "object_in_camera_optimized.txt",
            "overlay_2d": out_dir / "alignment_overlay_2d.png",
            "alignment_3d": out_dir / "alignment_3d.html",
        },
        "pointmap_scale": result.pointmap_scale,
        "object_target_camera": result.object_target_camera,
        "hand_visible_center_camera": result.hand_visible_center_camera,
        "hand_visible_center_pointmap": result.hand_visible_center_pointmap,
        "object_center_pointmap": result.object_center_pointmap,
        "metrics": result.metrics,
        "metadata": result.metadata,
    }
    (out_dir / "alignment_result.json").write_text(json.dumps(as_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh-file",
        default=str(REPO_ROOT / "objects/2026.7.14 3D test hunyuan/processed/kettle-decimated-100k.obj"),
    )
    parser.add_argument("--mesh-scale", type=float, default=0.20)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test"),
    )
    parser.add_argument("--hamer-subdir", default="pose_hamer_official")
    parser.add_argument("--hand-mask-name", default=None)
    parser.add_argument(
        "--translation-mode",
        choices=["direct", "scale_along_foundationpose_translation"],
        default="direct",
    )
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--max-hand-rays", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    object_mesh = load_mesh(Path(args.mesh_file).expanduser().resolve())
    fallback_k = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cases = collect_cases(output_root, args.hamer_subdir, args.hand_mask_name)
    if not cases:
        raise SystemExit("No complete cases found. Generate pointmaps first.")

    results = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases):02d}] {case['sample_id']} {case['lollipop']}")
        results.append(run_case(case, args, object_mesh, fallback_k))

    summary_file = Path(args.summary_file) if args.summary_file else output_root / "kettle_hand_anchored_pointmap_alignment_summary.json"
    summary = {
        "method": "do_as_i_do_hand_anchored_pointmap_single_frame",
        "num_cases": len(results),
        "results": results,
    }
    summary_file.write_text(json.dumps(as_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
