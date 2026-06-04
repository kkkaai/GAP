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

from prosthetic_grasp.geometry import (
    ContactPatchTarget,
    RetargetOptimizationConfig,
    RobotSurfaceSamples,
    RobotSurfaceTopology,
    assign_patches_to_robot_surface,
    load_robot_surface_model,
    optimize_retarget_action,
    retarget_loss_terms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test contact retargeting with synthetic object targets built "
            "from previously mapped robot surface points."
        )
    )
    parser.add_argument(
        "--mapping-npz",
        default="outputs/contact_mapping_shadow_canonical/center_targets_to_shadow_hand_canonical_center.npz",
        help="NPZ from tools/test_contact_mapping_schemes.py with canonical soft_surface results.",
    )
    parser.add_argument(
        "--robot-surface-npz",
        default="outputs/robot_surface_samples/shadow_hand_zero_samples.npz",
        help="Robot surface NPZ from tools/visualize_robot_surface_samples.py.",
    )
    parser.add_argument("--robot-model-dir", default="hand/shadow_hand")
    parser.add_argument("--robot-format", default="urdf", choices=["mjcf", "xml", "urdf", "auto"])
    parser.add_argument("--wrist-link", default="robot0:palm")
    parser.add_argument("--assignment-top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--maxiter-stage1", type=int, default=30)
    parser.add_argument("--maxiter-stage2", type=int, default=30)
    parser.add_argument("--fc-weight", type=float, default=0.01)
    parser.add_argument("--output-json", default="outputs/contact_retargeting/synthetic_shadow_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_surface = load_robot_surface_npz(args.robot_surface_npz)
    patches = load_synthetic_patches(args.mapping_npz, robot_surface)
    assign_result = assign_patches_to_robot_surface(
        patches,
        robot_surface,
        top_k=args.assignment_top_k,
        temperature=args.temperature,
    )

    robot_model = load_robot_surface_model(
        args.robot_model_dir,
        model_format=args.robot_format,
        wrist_link=args.wrist_link,
    )
    robot_topology = RobotSurfaceTopology(
        local_points=robot_surface.local_points,
        local_normals=robot_surface.local_normals,
        link_names=robot_surface.link_names,
    )
    config = RetargetOptimizationConfig(
        maxiter_stage1=args.maxiter_stage1,
        maxiter_stage2=args.maxiter_stage2,
        fc_weight=args.fc_weight,
        object_center=np.mean([patch.object_point_target for patch in patches], axis=0),
    )
    initial_terms = retarget_loss_terms(
        robot_model.zero_action,
        robot_model,
        robot_topology,
        patches,
        config,
        include_fc=True,
    )
    result = optimize_retarget_action(
        robot_model,
        robot_topology,
        patches,
        config=config,
    )

    payload = {
        "num_patches": len(patches),
        "assignment": {
            "scheme": assign_result.scheme,
            "mean_distance": float(assign_result.metrics["mean_distance"]),
            "max_distance": float(assign_result.metrics["max_distance"]),
            "assigned_link_names": assign_result.assigned_link_names.astype(str).tolist()
            if assign_result.assigned_link_names is not None
            else [],
        },
        "initial_loss_terms": initial_terms,
        "best_index": int(result.best_index),
        "best_loss_terms": result.best.loss_terms,
        "best_success": bool(result.best.success),
        "best_iterations": int(result.best.iterations),
        "best_elapsed_seconds": float(result.best.elapsed_seconds),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def load_robot_surface_npz(path: str | Path) -> RobotSurfaceSamples:
    data = np.load(path, allow_pickle=True)
    return RobotSurfaceSamples(
        points=np.asarray(data["points"], dtype=np.float64),
        normals=np.asarray(data["normals"], dtype=np.float64),
        local_points=np.asarray(data["local_points"], dtype=np.float64),
        local_normals=np.asarray(data["local_normals"], dtype=np.float64),
        link_names=np.asarray(data["link_names"].astype(str), dtype=object),
    )


def load_synthetic_patches(path: str | Path, robot_surface: RobotSurfaceSamples) -> list[ContactPatchTarget]:
    data = np.load(path, allow_pickle=True)
    if "soft_surface_assigned_points" not in data or "soft_surface_assigned_indices" not in data:
        raise KeyError(f"{path} must contain soft_surface_assigned_points and soft_surface_assigned_indices.")
    target_points = np.asarray(data["soft_surface_assigned_points"], dtype=np.float64)
    assigned_indices = np.asarray(data["soft_surface_assigned_indices"], dtype=np.int64)
    object_normals = -robot_surface.normals[assigned_indices]
    patches: list[ContactPatchTarget] = []
    for i, (point, normal) in enumerate(zip(target_points, object_normals)):
        patches.append(
            ContactPatchTarget(
                cluster_label=i,
                patch_size=1,
                sample_indices=np.asarray([i], dtype=np.int64),
                representative_sample_index=i,
                mano_face_index=0,
                mano_barycentric=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                mano_point_posed=point.copy(),
                object_point_target=point.copy(),
                object_normal_target=normalize(normal),
                object_point_nearest=point.copy(),
                object_normal_nearest=normalize(normal),
                object_point_center=point.copy(),
                object_normal_center=normalize(normal),
                contact_distance=0.0,
                canonical_robot_target=point.copy(),
            )
        )
    return patches


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / norm


if __name__ == "__main__":
    main()
