from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in [SRC_ROOT, TOOLS_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prosthetic_grasp.geometry import load_robot_surface_model, run_all_contact_mapping_schemes, sample_mano_surface
from test_contact_mapping_schemes import (
    map_mano_points_to_robot_wrist_frame,
    materialize_points_from_topology,
)
from test_mano_contact_clustering import (
    assign_samples_to_fingers,
    build_synthetic_contact_set,
    make_mano_21_joints,
    patch_numpy_legacy_aliases,
    representative_indices,
)
from prosthetic_grasp.geometry.contact_clustering import radius_connected_components
from test_contact_mapping_schemes import write_html as write_mapping_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark contact mapping schemes on random synthetic MANO contact patches."
    )
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--save-html-first", type=int, default=5)
    parser.add_argument("--mano-root", default="models")
    parser.add_argument("--robot-model-dir", default="hand/shadow_hand")
    parser.add_argument("--robot-model-format", default="urdf")
    parser.add_argument("--robot-urdf-path", default="hand/shadow_hand/shadowhand.urdf")
    parser.add_argument("--robot-wrist-link", default="robot0:palm")
    parser.add_argument("--num-robot-samples", type=int, default=2000)
    parser.add_argument("--num-mano-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--pose-scale", type=float, default=0.45)
    parser.add_argument("--patch-radius", type=float, default=0.010)
    parser.add_argument("--cluster-radius", type=float, default=0.014)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--min-seed-distance", type=float, default=0.025)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--spread-sigma", type=float, default=0.025)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--representative", choices=["center", "nearest"], default="center")
    parser.add_argument("--output-dir", default="outputs/contact_mapping_benchmark")
    parser.add_argument("--output-prefix", default="shadow_hand_random50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_model = load_robot_surface_model(
        args.robot_model_dir,
        model_format=args.robot_model_format,
        urdf_path=args.robot_urdf_path,
        wrist_link=args.robot_wrist_link,
    )
    robot_topology = robot_model.sample_surface_topology(num_points=args.num_robot_samples, seed=args.seed)
    robot_samples = robot_model.materialize_surface(robot_topology, robot_model.zero_action)
    robot_points = robot_samples.points
    robot_link_names = robot_samples.link_names
    mano_generator = ManoGenerator(args.mano_root)
    zero_vertices, zero_faces, zero_keypoints = mano_generator.create(seed=0, pose_scale=0.0)

    per_run = []
    failures = []
    scheme_names = None
    for run_index in range(args.num_runs):
        run_seed = args.seed + run_index
        try:
            run = run_once(args, run_seed, robot_points, robot_link_names, mano_generator, zero_vertices, zero_faces, zero_keypoints)
            per_run.append(run)
            scheme_names = scheme_names or [result["scheme"] for result in run["results"]]
            if run_index < args.save_html_first:
                write_mapping_html(
                    output_dir / f"{args.output_prefix}_run{run_index:03d}.html",
                    np.asarray(run["target_points"], dtype=np.float64),
                    robot_points,
                    robot_link_names,
                    run["_raw_results"],
                )
        except Exception as exc:
            failures.append({"run_index": run_index, "seed": run_seed, "error": f"{type(exc).__name__}: {exc}"})

    for run in per_run:
        run.pop("_raw_results", None)
    summary = summarize(per_run, scheme_names or [])
    payload = {
        "config": vars(args),
        "num_success": len(per_run),
        "num_failures": len(failures),
        "failures": failures,
        "summary": summary,
        "runs": per_run,
    }
    json_path = output_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"success: {len(per_run)}/{args.num_runs}, failures: {len(failures)}")
    for scheme, stats in summary.items():
        print(
            f"{scheme}: time_mean={stats['elapsed_seconds_mean']:.6f}s "
            f"time_p95={stats['elapsed_seconds_p95']:.6f}s "
            f"dist_mean={stats['mean_distance_mean']:.6f} "
            f"dist_p95={stats['mean_distance_p95']:.6f} "
            f"max_dist_mean={stats['max_distance_mean']:.6f}"
        )
    print(f"saved benchmark: {json_path}")
    if args.save_html_first > 0:
        print(f"saved html examples: {output_dir / (args.output_prefix + '_run000.html')} ...")


def run_once(
    args,
    seed: int,
    robot_points: np.ndarray,
    robot_link_names: np.ndarray,
    mano_generator,
    zero_vertices: np.ndarray,
    zero_faces: np.ndarray,
    zero_keypoints: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    vertices, faces, keypoints = mano_generator.create(seed=seed, pose_scale=args.pose_scale)
    samples = sample_mano_surface(
        vertices,
        faces,
        num_points=args.num_mano_samples,
        seed=seed,
        oversample_factor=20,
    )
    assignments = assign_samples_to_fingers(samples.points, keypoints)
    contact_indices, contact_points, seed_indices = build_synthetic_contact_set(
        samples.points,
        assignments,
        patch_radius=args.patch_radius,
        min_seed_distance=args.min_seed_distance,
        rng=rng,
    )
    seed_points = np.stack([samples.points[index] for index in seed_indices.values()], axis=0)
    labels = radius_connected_components(
        contact_points,
        radius=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
    )
    nearest_reps, center_reps = representative_indices(contact_points, labels, seed_points)
    rep_local_indices = center_reps if args.representative == "center" else nearest_reps
    if len(rep_local_indices) == 0:
        raise RuntimeError("No valid clusters found.")
    rep_sample_indices = contact_indices[np.asarray(rep_local_indices, dtype=np.int64)]
    face_indices = samples.face_indices[rep_sample_indices]
    barycentric = samples.barycentric[rep_sample_indices]

    canonical_points = materialize_points_from_topology(zero_vertices, zero_faces, face_indices, barycentric)
    target_points, metadata = map_mano_points_to_robot_wrist_frame(
        canonical_points,
        zero_keypoints,
        robot_points,
        robot_link_names,
        index_tip_link="robot0:ffdistal",
        middle_tip_link="robot0:mfdistal",
        little_tip_link="robot0:lfdistal",
        min_scale=0.25,
        max_scale=4.0,
    )
    results = run_all_contact_mapping_schemes(
        target_points,
        robot_points,
        robot_link_names=robot_link_names,
        temperature=args.temperature,
        spread_sigma=args.spread_sigma,
        diversity_weight=args.diversity_weight,
    )
    return {
        "seed": seed,
        "num_clusters": int(len(rep_local_indices)),
        "num_contact_points": int(len(contact_points)),
        "target_points": target_points.astype(float).tolist(),
        "scale": float(metadata["scale"]),
        "results": [
            {
                "scheme": result.scheme,
                "elapsed_seconds": float(result.elapsed_seconds),
                "mean_distance": float(result.metrics["mean_distance"]),
                "max_distance": float(result.metrics["max_distance"]),
                "min_distance": float(result.metrics["min_distance"]),
                "assigned_link_names": result.assigned_link_names.astype(str).tolist()
                if result.assigned_link_names is not None
                else [],
            }
            for result in results
        ],
        "_raw_results": results,
    }


def summarize(runs: list[dict[str, object]], scheme_names: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for scheme in scheme_names:
        scheme_rows = [
            result
            for run in runs
            for result in run["results"]
            if result["scheme"] == scheme
        ]
        if not scheme_rows:
            continue
        summary[scheme] = {
            "elapsed_seconds_mean": mean([row["elapsed_seconds"] for row in scheme_rows]),
            "elapsed_seconds_median": percentile([row["elapsed_seconds"] for row in scheme_rows], 50),
            "elapsed_seconds_p95": percentile([row["elapsed_seconds"] for row in scheme_rows], 95),
            "mean_distance_mean": mean([row["mean_distance"] for row in scheme_rows]),
            "mean_distance_median": percentile([row["mean_distance"] for row in scheme_rows], 50),
            "mean_distance_p95": percentile([row["mean_distance"] for row in scheme_rows], 95),
            "max_distance_mean": mean([row["max_distance"] for row in scheme_rows]),
            "max_distance_p95": percentile([row["max_distance"] for row in scheme_rows], 95),
        }
    return summary


def mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


class ManoGenerator:
    def __init__(self, mano_root: str) -> None:
        patch_numpy_legacy_aliases()
        import smplx

        self.torch = __import__("torch")
        self.model = smplx.create(
            model_path=mano_root,
            model_type="mano",
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        )

    def create(self, *, seed: int, pose_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        hand_pose = torch.randn(1, 45, generator=generator) * pose_scale
        hand_pose[:, 2::3] = hand_pose[:, 2::3].abs()
        output = self.model(
            betas=torch.zeros(1, 10),
            global_orient=torch.zeros(1, 3),
            hand_pose=hand_pose,
            transl=torch.zeros(1, 3),
            return_verts=True,
            return_full_pose=True,
        )
        vertices = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
        joints16 = output.joints.detach().cpu().numpy()[0].astype(np.float64)
        keypoints = make_mano_21_joints(vertices, joints16)
        return vertices, np.asarray(self.model.faces, dtype=np.int64), keypoints


if __name__ == "__main__":
    main()
