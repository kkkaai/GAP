#!/usr/bin/env python3
"""Run fingertip position-only retargeting for one HOI4D test frame."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = REPO_ROOT / "tools" / "vis"
RETARGET_ROOT = REPO_ROOT
RETARGET_SRC = REPO_ROOT / "src"
ROBOT_MESH_MAX_FACES = 60000

for path in (VIS_ROOT, RETARGET_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retarget_case import append_index
from scene_html import build_figure, mesh_trace, write_html
from scene_png import (
    DEFAULT_CAD,
    DEFAULT_FRAME,
    DEFAULT_HAND,
    DEFAULT_MANO_ROOT,
    DEFAULT_SEQUENCE,
    load_hand,
    try_make_mano_scene,
)

from prosthetic_grasp.common.types import Phase5HandPrediction, Phase5ManoResult
from prosthetic_grasp.geometry import load_robot_surface_model
from prosthetic_grasp.phases.phase6_prosthetic_action import (
    Phase6ProstheticAction,
    Phase6ProstheticActionConfig,
)


DEFAULT_RGB = DEFAULT_SEQUENCE / "align_rgb" / "00074.jpg"
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def make_phase5_from_hoi4d(hand_path: Path, mano_root: Path) -> Phase5ManoResult:
    hand_data = load_hand(hand_path)
    mano = try_make_mano_scene(hand_data, hand_path, mano_root)
    if mano is None:
        raise RuntimeError(
            "Could not build MANO mesh/keypoints. Check manopth, torch, and MANO model files."
        )
    prediction = Phase5HandPrediction(
        hand_index=0,
        is_right=(mano.side == "right"),
        bbox_xyxy=None,
        keypoints_2d=hand_data.get("kps2D"),
        keypoint_score_mean=1.0,
        vertices=mano.vertices,
        keypoints_3d=mano.keypoints,
        pred_cam=None,
        pred_cam_t_crop=None,
        pred_cam_t_full=None,
        focal_length=0.0,
        mano_params={
            "poseCoeff": np.asarray(hand_data.get("poseCoeff", []), dtype=np.float32),
            "beta": np.asarray(hand_data.get("beta", []), dtype=np.float32),
            "trans": np.asarray(hand_data.get("trans", []), dtype=np.float32),
        },
    )
    return Phase5ManoResult(status="ok", message="Loaded from HOI4D MANO annotation.", faces=mano.faces, hands=[prediction])


def run_position_retarget(
    hand: Path,
    mano_root: Path,
    robot_profile: str,
    optimization_restarts: int,
    max_nfev: int,
) -> tuple[Any, float]:
    phase5 = make_phase5_from_hoi4d(hand, mano_root)
    config_kwargs: dict[str, Any] = {
        "robot_profile": robot_profile,
        "optimization_restarts": optimization_restarts,
        "max_nfev": max_nfev,
    }
    if robot_profile == "shadow_hand":
        config_kwargs["robot_urdf_path"] = str(RETARGET_ROOT / "hand" / "shadow_hand" / "shadowhand.urdf")
    elif robot_profile == "folding_hand":
        config_kwargs["robot_xml_path"] = str(RETARGET_ROOT / "hand" / "folding_hand" / "folding.xml")
        config_kwargs["model_format"] = "xml"
    elif robot_profile == "folding_hand_right":
        config_kwargs["robot_xml_path"] = str(
            RETARGET_ROOT / "hand" / "folding_hand_right" / "folding_hand_right.xml"
        )
        config_kwargs["model_format"] = "xml"
    elif robot_profile == "inspire_hand":
        config_kwargs["robot_urdf_path"] = str(
            RETARGET_ROOT / "hand" / "inspire_hand_ftp" / "urdf" / "inspire_right.urdf"
        )

    retargeter = Phase6ProstheticAction(Phase6ProstheticActionConfig(**config_kwargs))
    started_at = time.perf_counter()
    result = retargeter.run(phase5)
    elapsed_seconds = time.perf_counter() - started_at
    return result, elapsed_seconds


def marker_trace(name: str, points: np.ndarray, color: str, size: int = 6) -> go.Scatter3d:
    return go.Scatter3d(
        name=name,
        x=points[:, 0].tolist(),
        y=points[:, 1].tolist(),
        z=points[:, 2].tolist(),
        mode="markers+text",
        marker=dict(color=color, size=size),
        text=["thumb", "index", "middle", "ring", "little"][: len(points)],
        textposition="top center",
        hoverinfo="skip",
    )


def point_trace(name: str, point: np.ndarray, color: str, size: int = 7, symbol: str = "diamond") -> go.Scatter3d:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return go.Scatter3d(
        name=name,
        x=[float(point[0])],
        y=[float(point[1])],
        z=[float(point[2])],
        mode="markers+text",
        marker=dict(color=color, size=size, symbol=symbol),
        text=[name],
        textposition="top center",
        hoverinfo="skip",
    )


def segment_trace(name: str, starts: np.ndarray, ends: np.ndarray, color: str) -> go.Scatter3d:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in zip(starts, ends):
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return go.Scatter3d(
        name=name,
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=5),
        hoverinfo="skip",
    )


def axis_range(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.25
    if radius < 1e-6:
        radius = 0.05
    return (
        [float(center[0] - radius), float(center[0] + radius)],
        [float(center[1] - radius), float(center[1] + radius)],
        [float(center[2] - radius), float(center[2] + radius)],
    )


def align_robot_points_to_scene(
    target_wrist: np.ndarray,
    scene_tips: np.ndarray,
    robot_points: np.ndarray,
    scale: bool = True,
) -> np.ndarray:
    """Map robot wrist-frame points into the HOI4D scene for visualization."""

    source = np.asarray(target_wrist, dtype=np.float64)
    target = np.asarray(scene_tips, dtype=np.float64)
    points = np.asarray(robot_points, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    source_norm = np.linalg.norm(source_centered)
    target_norm = np.linalg.norm(target_centered)
    scale_factor = float(target_norm / source_norm) if scale and source_norm > 1e-9 else 1.0
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    return ((points - source_center) * scale_factor) @ rotation + target_center


def rigid_robot_points_to_scene(
    scene_fingertips: np.ndarray,
    robot_target_fingertips: np.ndarray,
    robot_points: np.ndarray,
) -> np.ndarray:
    """Map robot wrist-frame points to scene with the same no-scale fit used by routes 2/3."""

    scene = np.asarray(scene_fingertips, dtype=np.float64)
    target = np.asarray(robot_target_fingertips, dtype=np.float64)
    points = np.asarray(robot_points, dtype=np.float64)
    scene_center = scene.mean(axis=0)
    target_center = target.mean(axis=0)
    scene_centered = scene - scene_center
    target_centered = target - target_center
    u, _, vt = np.linalg.svd(scene_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    return (points - target_center) @ rotation.T + scene_center


def load_retargeted_robot_mesh(result: Any) -> tuple[np.ndarray, np.ndarray] | None:
    metadata = getattr(result, "metadata", {}) or {}
    action = np.asarray(getattr(result, "action", []), dtype=np.float64)
    if action.size == 0:
        return None

    robot_path = metadata.get("robot_model_path") or metadata.get("robot_urdf_path")
    wrist_link = metadata.get("wrist_link", "robot0:palm")
    model_format = metadata.get("model_format", "urdf")
    if not robot_path:
        return None

    robot_path = Path(robot_path)
    model = load_robot_surface_model(
        robot_path.parent,
        model_format=model_format,
        urdf_path=robot_path if model_format == "urdf" else None,
        xml_path=robot_path if model_format in {"xml", "mjcf"} else None,
        wrist_link=wrist_link,
    )
    vertices, faces = model.link_mesh(action)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    return vertices, faces


def simplify_mesh_for_html(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int = ROBOT_MESH_MAX_FACES,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= max_faces:
        return vertices, faces

    step = int(np.ceil(len(faces) / max_faces))
    sampled_faces = faces[::step]
    used_vertices, remapped_faces = np.unique(sampled_faces.reshape(-1), return_inverse=True)
    compact_vertices = vertices[used_vertices]
    compact_faces = remapped_faces.reshape(sampled_faces.shape)
    return compact_vertices, compact_faces


def format_error(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000.0:.2f} mm"


def style_scene_traces(fig: go.Figure) -> None:
    for trace in fig.data:
        name = str(getattr(trace, "name", ""))
        if "MANO hand" in name:
            trace.opacity = 0.34
            trace.color = "#16a34a"
        elif name == "object CAD":
            trace.opacity = 0.72


def write_retarget_html(
    case_dir: Path,
    rgb: Path,
    base_meta: dict[str, Any],
    result: Any,
    optimization_seconds: float,
    scale_visualization: bool = True,
    output_suffix: str = "",
) -> Path:
    fig, scene_meta = build_figure(
        Path(base_meta["sequence"]),
        str(base_meta["frame"]),
        Path(base_meta["cad"]),
        Path(base_meta["hand_pickle"]),
        Path(base_meta["mano_root"]),
    )
    style_scene_traces(fig)
    if result.target_fingertips_wrist is not None and result.prosthetic_fingertips_wrist is not None:
        target_wrist = np.asarray(result.target_fingertips_wrist, dtype=np.float64)
        robot_wrist = np.asarray(result.prosthetic_fingertips_wrist, dtype=np.float64)
        scene_tips = np.asarray(result.mano_fingertips, dtype=np.float64)
        if scale_visualization:
            target_scene = align_robot_points_to_scene(
                target_wrist, scene_tips, target_wrist, scale=True
            )
            robot_scene = align_robot_points_to_scene(
                target_wrist, scene_tips, robot_wrist, scale=True
            )
        else:
            target_scene = scene_tips
            robot_scene = rigid_robot_points_to_scene(scene_tips, target_wrist, robot_wrist)
        robot_mesh = load_retargeted_robot_mesh(result)
        if robot_mesh is not None:
            robot_vertices, robot_faces = robot_mesh
            robot_vertices, robot_faces = simplify_mesh_for_html(robot_vertices, robot_faces)
            if scale_visualization:
                robot_vertices_scene = align_robot_points_to_scene(
                    target_wrist, scene_tips, robot_vertices, scale=True
                )
            else:
                robot_vertices_scene = rigid_robot_points_to_scene(scene_tips, target_wrist, robot_vertices)
            fig.add_trace(
                mesh_trace(
                    f"{base_meta['robot_profile']} mesh",
                    robot_vertices_scene,
                    robot_faces,
                    "#2563eb",
                    0.30,
                )
            )
        if result.mano_wrist is not None:
            mano_wrist = np.asarray(result.mano_wrist, dtype=np.float64)
            fig.add_trace(point_trace("MANO wrist", mano_wrist, "#15803d", 7, "diamond"))
        if scale_visualization:
            robot_wrist_scene = align_robot_points_to_scene(
                target_wrist,
                scene_tips,
                np.zeros((1, 3), dtype=np.float64),
                scale=True,
            )[0]
        else:
            robot_wrist_scene = rigid_robot_points_to_scene(
                scene_tips,
                target_wrist,
                np.zeros((1, 3), dtype=np.float64),
            )[0]
        fig.add_trace(point_trace(f"{base_meta['robot_profile']} wrist", robot_wrist_scene, "#1d4ed8", 7, "diamond"))
        fig.add_trace(marker_trace("MANO fingertip targets", target_scene, "#d9480f", 5))
        fig.add_trace(marker_trace(f"{base_meta['robot_profile']} retargeted fingertips", robot_scene, "#1864ab", 5))
        fig.add_trace(segment_trace("position error", target_scene, robot_scene, "#868e96"))

    fig.update_layout(
        title="HOI4D grasp + position-only retargeting",
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    errors = {
        "robot hand": base_meta["robot_profile"],
        "visual alignment": "scale + rigid" if scale_visualization else "rigid only",
        "optimization time": f"{optimization_seconds:.3f} s",
    }
    if result.fingertip_error is not None:
        fingertip_error = np.asarray(result.fingertip_error, dtype=np.float64)
        for name, value in zip(["thumb", "index", "middle", "ring", "little"], fingertip_error):
            errors[f"{name} error"] = format_error(float(value))
        errors["mean fingertip error"] = format_error(float(np.mean(fingertip_error)))
        errors["max fingertip error"] = format_error(float(np.max(fingertip_error)))
    errors["status"] = result.status

    scene_meta.update(base_meta)
    scene_meta["errors"] = errors
    scene_meta["retargeting"] = {
        "route": "position_only_fingertips",
        "status": result.status,
        "message": result.message,
        "action_names": list(result.action_names),
        "optimization_seconds": float(optimization_seconds),
        "mean_fingertip_error": float(np.mean(result.fingertip_error)) if result.fingertip_error is not None else None,
        "max_fingertip_error": float(np.max(result.fingertip_error)) if result.fingertip_error is not None else None,
    }
    scene_meta["visual_alignment"] = "scale_rigid" if scale_visualization else "rigid_only"
    robot_html = case_dir / f"scene_{base_meta['robot_profile']}{output_suffix}.html"
    write_html(fig, scene_meta, rgb, robot_html)
    if not output_suffix:
        shutil.copy2(robot_html, case_dir / "scene.html")
        shutil.copy2(robot_html, case_dir / "retargeted_scene.html")
    return robot_html


def prepare_shared_case(
    sequence: Path,
    frame: str,
    rgb: Path,
    cad: Path,
    hand: Path,
    mano_root: Path,
    output_root: Path,
    case_id: str,
    overwrite: bool,
    render_input_html: bool = True,
) -> Path:
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    original_dst = case_dir / "original.jpg"
    if overwrite or not original_dst.exists():
        shutil.copy2(rgb, original_dst)

    input_html = case_dir / "input_scene.html"
    input_json = case_dir / "input_scene.json"
    if overwrite or not input_json.exists() or (render_input_html and not input_html.exists()):
        if render_input_html:
            fig, scene_meta = build_figure(sequence, frame, cad, hand, mano_root)
        else:
            scene_meta = {}
        scene_meta.update(
            {
                "case_id": case_id,
                "case_dir": str(case_dir),
                "original_image": str(original_dst),
                "input_scene_html": str(input_html),
                "input_scene_json": str(input_json),
            }
        )
        if render_input_html:
            write_html(fig, scene_meta, original_dst, input_html)
        input_json.write_text(json.dumps(scene_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    append_index(
        output_root / "index.json",
        {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "sequence": str(sequence),
            "frame": frame,
            "original_image": str(original_dst),
            "input_scene_html": str(input_html),
            "input_scene_json": str(input_json),
            "retargeting_status": "pending",
        },
    )
    return case_dir


def update_case_index(output_root: Path, case_id: str, status: str, scene_html: Path) -> None:
    index_path = output_root / "index.json"
    if not index_path.exists():
        return
    data = json.loads(index_path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("case_id") == case_id:
            item["retargeting_status"] = status
            item["scene_html"] = str(scene_html)
            item[f"{scene_html.stem}_html"] = str(scene_html)
            item[f"{scene_html.stem}_json"] = str(output_root / case_id / f"retargeted_{scene_html.stem.removeprefix('scene_')}.json")
            break
    index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", default="bottle_frame74_position_only")
    parser.add_argument("--robot-profile", default="shadow_hand")
    parser.add_argument("--optimization-restarts", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--scale-visualization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale robot visualization onto MANO fingertips. Defaults to rigid-only/no-scale for baseline comparability.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip Plotly HTML rendering for large batch runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = prepare_shared_case(
        sequence=args.sequence,
        frame=args.frame,
        rgb=args.rgb,
        cad=args.cad,
        hand=args.hand,
        mano_root=args.mano_root,
        output_root=args.output_root,
        case_id=args.case_id or "bottle_frame74_position_only",
        overwrite=args.overwrite,
        render_input_html=not args.no_html,
    )

    result, optimization_seconds = run_position_retarget(
        hand=args.hand,
        mano_root=args.mano_root,
        robot_profile=args.robot_profile,
        optimization_restarts=args.optimization_restarts,
        max_nfev=args.max_nfev,
    )
    payload = {
        "case_id": args.case_id,
        "route": "position_only_fingertips",
        "sequence": str(args.sequence),
        "frame": str(args.frame),
        "cad": str(args.cad),
        "hand_pickle": str(args.hand),
        "mano_root": str(args.mano_root),
        "robot_profile": args.robot_profile,
        "optimization_seconds": optimization_seconds,
        "retargeting_package": str(RETARGET_ROOT),
        "result": as_jsonable(result),
    }
    robot_json = case_dir / f"retargeted_{args.robot_profile}.json"
    robot_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(robot_json, case_dir / "retargeted_scene.json")

    base_meta = {
        "case_id": args.case_id,
        "sequence": str(args.sequence),
        "frame": str(args.frame),
        "cad": str(args.cad),
        "hand_pickle": str(args.hand),
        "mano_root": str(args.mano_root),
        "robot_profile": args.robot_profile,
    }
    scene_html = None
    if not args.no_html:
        scene_html = write_retarget_html(
            case_dir,
            case_dir / "original.jpg",
            base_meta,
            result,
            optimization_seconds,
            scale_visualization=args.scale_visualization,
        )
        update_case_index(
            args.output_root,
            args.case_id,
            "position_only_done" if result.status == "ok" else result.status,
            scene_html,
        )
    print(f"Wrote position retarget case: {case_dir}")
    print(f"Wrote {robot_json}")
    if scene_html is not None:
        print(f"Open: {scene_html}")


if __name__ == "__main__":
    main()
