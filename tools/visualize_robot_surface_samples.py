from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import RobotSurfaceTopology, load_robot_surface_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample fixed link-local surface points for a prosthetic/robot hand and visualize them."
    )
    parser.add_argument("--model-dir", default="hand/folding_hand_right")
    parser.add_argument("--model-format", choices=["auto", "mjcf", "xml", "urdf"], default="auto")
    parser.add_argument("--xml-path", default="")
    parser.add_argument("--urdf-path", default="")
    parser.add_argument("--wrist-link", default="base_link")
    parser.add_argument("--pose", choices=["zero", "random"], default="zero")
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--oversample-factor", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument(
        "--topology-npz",
        default="",
        help="Optional existing robot samples npz containing local_points, local_normals, and link_names.",
    )
    parser.add_argument("--exclude-link", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/robot_surface_samples")
    parser.add_argument("--output-prefix", default="folding_hand_right_surface")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_robot_surface_model(
        args.model_dir,
        model_format=args.model_format,
        xml_path=args.xml_path or None,
        urdf_path=args.urdf_path or None,
        wrist_link=args.wrist_link,
    )
    if args.topology_npz:
        topology = load_topology_npz(args.topology_npz)
        topology_source = f"loaded topology from {args.topology_npz}"
    else:
        topology = model.sample_surface_topology(
            num_points=args.num_points,
            seed=args.seed,
            exclude_links=set(args.exclude_link),
            use_farthest_point_sampling=not args.no_fps,
            oversample_factor=args.oversample_factor,
        )
        topology_source = "sampled topology on current model"
    if args.pose == "zero":
        action = model.zero_action.copy()
    else:
        lower, upper = model.joint_bounds()
        rng = np.random.default_rng(args.seed)
        action = rng.uniform(lower, upper)
    samples = model.materialize_surface(topology, action)
    mesh_vertices, mesh_faces = model.link_mesh(action)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{args.output_prefix}_{args.pose}_samples.npz"
    output_html = output_dir / f"{args.output_prefix}_{args.pose}_samples.html"
    np.savez_compressed(
        output_npz,
        points=samples.points.astype(np.float32),
        normals=samples.normals.astype(np.float32),
        local_points=samples.local_points.astype(np.float32),
        local_normals=samples.local_normals.astype(np.float32),
        link_names=samples.link_names.astype(str),
        action=action.astype(np.float32),
        action_names=np.asarray(model.action_names),
        model_dir=str(Path(args.model_dir)),
        model_path=str(model.xml_path.relative_to(REPO_ROOT) if model.xml_path.is_relative_to(REPO_ROOT) else model.xml_path),
        wrist_link=model.wrist_link,
    )
    fig = make_figure(
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        sample_points=samples.points,
        link_names=samples.link_names,
        title=f"{Path(args.model_dir).name} surface samples - {args.pose}",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")

    print(f"model: {args.model_dir}")
    print(f"model path: {model.xml_path}")
    print(f"topology: {topology_source}")
    print(f"pose: {args.pose}")
    print(f"actions: {len(action)} {model.action_names}")
    print(f"samples: {samples.points.shape}")
    print(f"saved samples: {output_npz}")
    print(f"visualization: {output_html}")
    if args.show:
        fig.show()


def load_topology_npz(path: str | Path) -> RobotSurfaceTopology:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"topology npz does not exist: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for key in ["local_points", "local_normals", "link_names"]:
        if key not in data:
            raise KeyError(f"{npz_path} must contain {key!r}.")
    return RobotSurfaceTopology(
        local_points=np.asarray(data["local_points"], dtype=np.float64),
        local_normals=np.asarray(data["local_normals"], dtype=np.float64),
        link_names=np.asarray(data["link_names"].astype(str), dtype=object),
    )


def make_figure(
    *,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    sample_points: np.ndarray,
    link_names: np.ndarray,
    title: str,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Robot surface sample HTML visualization requires plotly.") from exc

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            name="robot hand mesh",
            color="rgba(40, 100, 230, 0.32)",
            opacity=0.32,
            flatshading=False,
        )
    )
    unique_links = {name: i for i, name in enumerate(sorted(set(link_names.astype(str))))}
    colors = np.asarray([unique_links[str(name)] for name in link_names], dtype=np.float64)
    fig.add_trace(
        go.Scatter3d(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            z=sample_points[:, 2],
            mode="markers",
            name=f"fixed link-local samples ({len(sample_points)})",
            marker=dict(size=2.2, color=colors, colorscale="Turbo", showscale=True),
            text=link_names.astype(str),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


if __name__ == "__main__":
    main()
