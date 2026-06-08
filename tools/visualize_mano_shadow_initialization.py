from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import (  # noqa: E402
    action_summary,
    load_robot_surface_model,
    mano_pose_to_shadow_action,
)
from tools.test_phase6_cylinder_grasp import create_power_grasp_mano  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the OmniDexGrasp-style MANO-pose to Shadow initial joint action."
    )
    parser.add_argument("--mano-root", default="models")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--pose-scale", type=float, default=0.40)
    parser.add_argument("--robot-model-dir", default="hand/shadow_hand")
    parser.add_argument("--robot-format", default="urdf", choices=["mjcf", "xml", "urdf", "auto"])
    parser.add_argument("--wrist-link", default="robot0:palm")
    parser.add_argument("--mano-shadow-init-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/mano_shadow_initialization")
    parser.add_argument("--output-prefix", default="mano_to_shadow_init")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mano = create_power_grasp_mano(args.mano_root, seed=args.seed, pose_scale=args.pose_scale)
    robot_model = load_robot_surface_model(
        args.robot_model_dir,
        model_format=args.robot_format,
        wrist_link=args.wrist_link,
    )
    lower, upper = robot_model.joint_bounds()
    initialization = mano_pose_to_shadow_action(
        mano.hand_pose,
        robot_model.action_names,
        lower=lower,
        upper=upper,
        scale=args.mano_shadow_init_scale,
    )

    zero_vertices, zero_faces = robot_model.link_mesh(robot_model.zero_action)
    init_vertices, init_faces = robot_model.link_mesh(initialization.action)

    html_path = output_dir / f"{args.output_prefix}.html"
    write_html(
        html_path,
        mano_vertices=mano.vertices,
        mano_faces=mano.faces,
        zero_vertices=zero_vertices,
        zero_faces=zero_faces,
        init_vertices=init_vertices,
        init_faces=init_faces,
    )

    summary = {
        "status": "ok",
        "seed": int(args.seed),
        "pose_scale": float(args.pose_scale),
        "mano_shadow_init_scale": float(args.mano_shadow_init_scale),
        "num_clipped": int(initialization.num_clipped),
        "clipped_joint_names": list(initialization.clipped_joint_names),
        "raw_action_summary": action_summary(initialization.raw_action, robot_model.action_names),
        "clipped_action_summary": action_summary(initialization.action, robot_model.action_names),
        "outputs": {"html": str(html_path)},
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_html(
    output_path: Path,
    *,
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    zero_vertices: np.ndarray,
    zero_faces: np.ndarray,
    init_vertices: np.ndarray,
    init_faces: np.ndarray,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Posed MANO input", "Shadow zero pose and MANO-derived initial pose"),
        horizontal_spacing=0.02,
    )
    fig.add_trace(
        go.Mesh3d(
            x=mano_vertices[:, 0],
            y=mano_vertices[:, 1],
            z=mano_vertices[:, 2],
            i=mano_faces[:, 0],
            j=mano_faces[:, 1],
            k=mano_faces[:, 2],
            name="posed MANO",
            color="rgba(35, 170, 95, 0.55)",
            opacity=0.55,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Mesh3d(
            x=zero_vertices[:, 0],
            y=zero_vertices[:, 1],
            z=zero_vertices[:, 2],
            i=zero_faces[:, 0],
            j=zero_faces[:, 1],
            k=zero_faces[:, 2],
            name="Shadow zero pose",
            color="rgba(180, 180, 180, 0.18)",
            opacity=0.18,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Mesh3d(
            x=init_vertices[:, 0],
            y=init_vertices[:, 1],
            z=init_vertices[:, 2],
            i=init_faces[:, 0],
            j=init_faces[:, 1],
            k=init_faces[:, 2],
            name="Shadow MANO-derived init",
            color="rgba(40, 110, 230, 0.55)",
            opacity=0.55,
        ),
        row=1,
        col=2,
    )
    axis_layout = dict(
        aspectmode="data",
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
    )
    fig.update_layout(
        title="MANO bending gesture mapped to Shadow initial joint action",
        scene=axis_layout,
        scene2=axis_layout,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(x=0.01, y=0.99),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
