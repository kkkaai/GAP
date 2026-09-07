#!/usr/bin/env python3
"""Run a Do-as-I-Do-style single-frame warmup retargeting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_ROOT = REPO_ROOT / "tools" / "vis"
RETARGET_SRC = REPO_ROOT / "src"

for path in (REPO_ROOT, VIS_ROOT, RETARGET_SRC, REPO_ROOT / "tools" / "retarget"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prosthetic_grasp.geometry import (  # noqa: E402
    DaiSingleFrameWeights,
    SingleFrameReference,
    classify_contact_stage,
    final_loss_terms,
    perturbation_robustness_score,
    robot_tip_from_link,
    warmup_loss_terms,
)


DEFAULT_SEQUENCE = REPO_ROOT / "extracted_dataset_sampled" / "ZY20210800004" / "H4" / "C4" / "N01" / "S55" / "s02" / "T1"
DEFAULT_FRAME = "00074"
DEFAULT_RGB = DEFAULT_SEQUENCE / "align_rgb" / "00074.jpg"
DEFAULT_CAD = REPO_ROOT / "extracted_dataset_sampled" / "models" / "kettle" / "obj_mesh" / "kettle.obj"
DEFAULT_HAND = DEFAULT_SEQUENCE / "handpose" / "00074.pkl"
DEFAULT_MANO_ROOT = REPO_ROOT / "mano" / "mano"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "runs"
MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
ROBOT_PROFILE_CHOICES = ["folding_hand_right", "inspire_hand", "shadow_hand"]


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()


def pose_from_absolute_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    return rotation_matrix_from_rotvec(params[:3]), params[3:6]


def rotvec_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def make_reference(
    mano: Any,
    object_vertices_scene: np.ndarray,
    *,
    contact_distance_threshold: float,
) -> SingleFrameReference:
    from prosthetic_grasp.geometry import build_hand_frame

    keypoints = np.asarray(mano.keypoints, dtype=np.float64)
    fingertips = keypoints[MANO_FINGERTIP_INDICES]
    palm_rotation = build_hand_frame(
        wrist=keypoints[0],
        index_mcp=keypoints[5],
        middle_mcp=keypoints[9],
        little_mcp=keypoints[17],
    )
    stage = classify_contact_stage(
        np.asarray(mano.vertices, dtype=np.float64),
        object_vertices_scene,
        threshold=contact_distance_threshold,
    )
    return SingleFrameReference(
        object_points=np.asarray(object_vertices_scene, dtype=np.float64),
        object_normals=_estimate_object_vertex_normals(object_vertices_scene),
        hand_points=np.asarray(mano.vertices, dtype=np.float64),
        fingertip_targets=fingertips,
        palm_position=keypoints[0],
        palm_rotation=palm_rotation,
        q_reference=np.zeros(1, dtype=np.float64),
        contact_stage=stage.stage,
        contact_distance_threshold=contact_distance_threshold,
    )


def _estimate_object_vertex_normals(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    center = points.mean(axis=0)
    normals = points - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(points), 1))
    return np.where(norms > 1e-12, normals / np.maximum(norms, 1e-12), fallback)


def downsample_reference(reference: SingleFrameReference, *, max_object_points: int, seed: int) -> SingleFrameReference:
    if len(reference.object_points) <= max_object_points:
        return reference
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(reference.object_points), size=max_object_points, replace=False)
    return SingleFrameReference(
        object_points=reference.object_points[indices],
        object_normals=reference.object_normals[indices],
        hand_points=reference.hand_points,
        fingertip_targets=reference.fingertip_targets,
        palm_position=reference.palm_position,
        palm_rotation=reference.palm_rotation,
        q_reference=reference.q_reference,
        contact_stage=reference.contact_stage,
        contact_distance_threshold=reference.contact_distance_threshold,
    )


def evaluate_state(
    x: np.ndarray,
    robot_model: Any,
    robot_topology: Any,
    fingertip_links: list[str],
    reference: SingleFrameReference,
    q_kinematic: np.ndarray,
    *,
    q_warmup: np.ndarray | None,
    weights: DaiSingleFrameWeights,
    stage: str,
) -> dict[str, float]:
    from position_force import transform_normals, transform_points

    rotation, translation = pose_from_absolute_params(x[:6])
    action = np.asarray(x[6:], dtype=np.float64)
    surface = robot_model.materialize_surface(robot_topology, action)
    robot_points = transform_points(surface.points, rotation, translation)
    robot_normals = transform_normals(surface.normals, rotation)
    fingertips = np.asarray(
        [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in fingertip_links],
        dtype=np.float64,
    )
    fingertips_scene = transform_points(fingertips, rotation, translation)
    palm_position = transform_points(np.zeros((1, 3), dtype=np.float64), rotation, translation)[0]
    palm_rotation = rotation
    if stage == "warmup":
        return warmup_loss_terms(
            robot_points=robot_points,
            robot_normals=robot_normals,
            fingertip_points=fingertips_scene,
            palm_position=palm_position,
            palm_rotation=palm_rotation,
            q=action,
            reference=reference,
            q_kinematic=q_kinematic,
            weights=weights,
        )
    if q_warmup is None:
        raise ValueError("q_warmup is required for final loss.")
    return final_loss_terms(
        robot_points=robot_points,
        robot_normals=robot_normals,
        fingertip_points=fingertips_scene,
        palm_position=palm_position,
        palm_rotation=palm_rotation,
        q=action,
        reference=reference,
        q_kinematic=q_kinematic,
        q_warmup=q_warmup,
        weights=weights,
    )


def optimize_stage(
    *,
    stage: str,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    robot_model: Any,
    robot_topology: Any,
    fingertip_links: list[str],
    reference: SingleFrameReference,
    q_kinematic: np.ndarray,
    q_warmup: np.ndarray | None,
    weights: DaiSingleFrameWeights,
    maxiter: int,
    optimizer: str,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    started = time.perf_counter()
    initial_terms = evaluate_state(
        x0,
        robot_model,
        robot_topology,
        fingertip_links,
        reference,
        q_kinematic,
        q_warmup=q_warmup,
        weights=weights,
        stage=stage,
    )
    result = minimize(
        lambda values: evaluate_state(
            values,
            robot_model,
            robot_topology,
            fingertip_links,
            reference,
            q_kinematic,
            q_warmup=q_warmup,
            weights=weights,
            stage=stage,
        )["total"],
        np.asarray(x0, dtype=np.float64),
        method=optimizer,
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    elapsed = time.perf_counter() - started
    x_best = np.asarray(result.x, dtype=np.float64)
    best_terms = evaluate_state(
        x_best,
        robot_model,
        robot_topology,
        fingertip_links,
        reference,
        q_kinematic,
        q_warmup=q_warmup,
        weights=weights,
        stage=stage,
    )
    return {
        "x": x_best,
        "action": x_best[6:],
        "pose_params": x_best[:6],
        "initial_loss_terms": initial_terms,
        "best_loss_terms": best_terms,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "elapsed_seconds": float(elapsed),
    }


def perturb_reference(reference: SingleFrameReference, rng: np.random.Generator, *, translation_scale: float) -> SingleFrameReference:
    delta = rng.normal(size=3) * float(translation_scale)
    return SingleFrameReference(
        object_points=reference.object_points + delta,
        object_normals=reference.object_normals,
        hand_points=reference.hand_points,
        fingertip_targets=reference.fingertip_targets,
        palm_position=reference.palm_position,
        palm_rotation=reference.palm_rotation,
        q_reference=reference.q_reference,
        contact_stage=reference.contact_stage,
        contact_distance_threshold=reference.contact_distance_threshold,
    )


def evaluate_robustness(
    x: np.ndarray,
    robot_model: Any,
    robot_topology: Any,
    fingertip_links: list[str],
    reference: SingleFrameReference,
    q_kinematic: np.ndarray,
    q_warmup: np.ndarray,
    weights: DaiSingleFrameWeights,
    *,
    num_trials: int,
    translation_scale: float,
    success_threshold: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(int(num_trials)):
        ref_i = perturb_reference(reference, rng, translation_scale=translation_scale)
        terms = evaluate_state(
            x,
            robot_model,
            robot_topology,
            fingertip_links,
            ref_i,
            q_kinematic,
            q_warmup=q_warmup,
            weights=weights,
            stage="final",
        )
        losses.append(float(terms["total"]))
    nominal = evaluate_state(
        x,
        robot_model,
        robot_topology,
        fingertip_links,
        reference,
        q_kinematic,
        q_warmup=q_warmup,
        weights=weights,
        stage="final",
    )["total"]
    score = perturbation_robustness_score(
        nominal_loss=float(nominal),
        perturbed_losses=np.asarray(losses, dtype=np.float64),
        success_threshold=success_threshold,
    )
    return {
        "score": _jsonable(score),
        "perturbed_losses": losses,
        "translation_scale_m": float(translation_scale),
        "success_threshold": float(success_threshold),
    }


def materialize_summary(
    x: np.ndarray,
    robot_model: Any,
    robot_topology: Any,
    fingertip_links: list[str],
) -> dict[str, Any]:
    from position_force import transform_points

    rotation, translation = pose_from_absolute_params(x[:6])
    action = np.asarray(x[6:], dtype=np.float64)
    surface = robot_model.materialize_surface(robot_topology, action)
    robot_points = transform_points(surface.points, rotation, translation)
    fingertips = np.asarray(
        [robot_tip_from_link(surface.points, surface.link_names.astype(str), link) for link in fingertip_links],
        dtype=np.float64,
    )
    fingertips_scene = transform_points(fingertips, rotation, translation)
    palm_scene = transform_points(np.zeros((1, 3), dtype=np.float64), rotation, translation)[0]
    return {
        "action": action.astype(float).tolist(),
        "pose_params": np.asarray(x[:6], dtype=np.float64).astype(float).tolist(),
        "robot_surface_points_scene": robot_points.astype(float).tolist(),
        "robot_fingertips_scene": fingertips_scene.astype(float).tolist(),
        "robot_palm_scene": palm_scene.astype(float).tolist(),
    }


def run_do_as_i_do_single_frame(
    sequence: Path,
    frame: str,
    cad: Path,
    hand_path: Path,
    mano_root: Path,
    robot_profile: str,
    *,
    position_restarts: int,
    position_max_nfev: int,
    num_robot_samples: int,
    num_object_points: int,
    warmup_maxiter: int,
    final_maxiter: int,
    contact_distance_threshold: float,
    perturb_trials: int,
    perturb_translation_scale: float,
    optimizer: str,
    seed: int,
) -> dict[str, Any]:
    from contact_surface import ROBOT_PROFILES, load_hoi4d_scene, load_robot_model
    from position import as_jsonable, run_position_retarget
    from position_force import PointSetRigidTransform

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
        raise RuntimeError(f"Kinematic reference failed: {position_result.status} {position_result.message}")

    _, mano, object_vertices_scene, _ = load_hoi4d_scene(sequence, frame, cad, hand_path, mano_root)
    reference = make_reference(
        mano,
        object_vertices_scene,
        contact_distance_threshold=contact_distance_threshold,
    )
    reference = downsample_reference(reference, max_object_points=num_object_points, seed=seed)

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
    x_kin = np.concatenate([rotvec_from_matrix(init_rotation), init_translation, np.asarray(position_result.action, dtype=np.float64)])
    q_kinematic = x_kin[6:]
    reference = SingleFrameReference(
        object_points=reference.object_points,
        object_normals=reference.object_normals,
        hand_points=reference.hand_points,
        fingertip_targets=reference.fingertip_targets,
        palm_position=reference.palm_position,
        palm_rotation=reference.palm_rotation,
        q_reference=q_kinematic,
        contact_stage=reference.contact_stage,
        contact_distance_threshold=reference.contact_distance_threshold,
    )

    lower, upper = robot_model.joint_bounds()
    pose_bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    bounds = pose_bounds + list(zip(lower, upper))
    x_warm0 = np.concatenate([x_kin[:6], np.asarray(robot_model.zero_action, dtype=np.float64)])
    weights = DaiSingleFrameWeights()

    warmup = optimize_stage(
        stage="warmup",
        x0=x_warm0,
        bounds=bounds,
        robot_model=robot_model,
        robot_topology=robot_topology,
        fingertip_links=profile["fingertip_links"],
        reference=reference,
        q_kinematic=q_kinematic,
        q_warmup=None,
        weights=weights,
        maxiter=warmup_maxiter,
        optimizer=optimizer,
    )
    final = optimize_stage(
        stage="final",
        x0=np.asarray(warmup["x"], dtype=np.float64),
        bounds=bounds,
        robot_model=robot_model,
        robot_topology=robot_topology,
        fingertip_links=profile["fingertip_links"],
        reference=reference,
        q_kinematic=q_kinematic,
        q_warmup=np.asarray(warmup["action"], dtype=np.float64),
        weights=weights,
        maxiter=final_maxiter,
        optimizer=optimizer,
    )
    success_threshold = float(final["best_loss_terms"]["total"] * 1.25 + 1e-9)
    robustness = evaluate_robustness(
        np.asarray(final["x"], dtype=np.float64),
        robot_model,
        robot_topology,
        profile["fingertip_links"],
        reference,
        q_kinematic,
        np.asarray(warmup["action"], dtype=np.float64),
        weights,
        num_trials=perturb_trials,
        translation_scale=perturb_translation_scale,
        success_threshold=success_threshold,
        seed=seed + 101,
    )

    return {
        "status": "ok",
        "route": "do_as_i_do_single_frame",
        "robot_profile": robot_profile,
        "action_names": list(robot_model.action_names),
        "action": np.asarray(final["action"], dtype=np.float64).astype(float).tolist(),
        "kinematic_reference": {
            "optimization_seconds": float(position_seconds),
            "status": position_result.status,
            "message": position_result.message,
            "result": as_jsonable(position_result),
        },
        "reference": {
            "contact_stage": reference.contact_stage,
            "contact_distance_threshold_m": float(reference.contact_distance_threshold),
            "num_object_points": int(len(reference.object_points)),
            "source": "single_frame_state_from_mano_object_scene",
        },
        "warmup": {
            **{key: value for key, value in warmup.items() if key != "x"},
            "summary": materialize_summary(np.asarray(warmup["x"], dtype=np.float64), robot_model, robot_topology, profile["fingertip_links"]),
        },
        "final": {
            **{key: value for key, value in final.items() if key != "x"},
            "summary": materialize_summary(np.asarray(final["x"], dtype=np.float64), robot_model, robot_topology, profile["fingertip_links"]),
        },
        "robustness": robustness,
        "config": {
            "optimizer": optimizer,
            "num_robot_samples": int(num_robot_samples),
            "num_object_points": int(num_object_points),
            "warmup_maxiter": int(warmup_maxiter),
            "final_maxiter": int(final_maxiter),
            "weights": _jsonable(weights),
            "do_as_i_do_source": str(REPO_ROOT / "external" / "do-as-i-do" / "retargeting"),
            "adaptation": "single_frame_warmup_plus_final_static_optimization",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--robot-profile", choices=ROBOT_PROFILE_CHOICES, default="folding_hand_right")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", default="do_as_i_do_single_frame")
    parser.add_argument("--position-restarts", type=int, default=4)
    parser.add_argument("--position-max-nfev", type=int, default=100)
    parser.add_argument("--num-robot-samples", type=int, default=600)
    parser.add_argument("--num-object-points", type=int, default=384)
    parser.add_argument("--warmup-maxiter", type=int, default=60)
    parser.add_argument("--final-maxiter", type=int, default=80)
    parser.add_argument("--contact-distance-threshold", type=float, default=0.03)
    parser.add_argument("--perturb-trials", type=int, default=8)
    parser.add_argument("--perturb-translation-scale", type=float, default=0.005)
    parser.add_argument("--optimizer", default="L-BFGS-B")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from position import prepare_shared_case

    case_dir = prepare_shared_case(
        args.sequence,
        args.frame,
        args.rgb,
        args.cad,
        args.hand,
        args.mano_root,
        args.output_root,
        args.case_id,
        False,
    )
    route_dir = case_dir / "do_as_i_do_single_frame"
    route_dir.mkdir(parents=True, exist_ok=True)
    payload = run_do_as_i_do_single_frame(
        args.sequence,
        args.frame,
        args.cad,
        args.hand,
        args.mano_root,
        args.robot_profile,
        position_restarts=args.position_restarts,
        position_max_nfev=args.position_max_nfev,
        num_robot_samples=args.num_robot_samples,
        num_object_points=args.num_object_points,
        warmup_maxiter=args.warmup_maxiter,
        final_maxiter=args.final_maxiter,
        contact_distance_threshold=args.contact_distance_threshold,
        perturb_trials=args.perturb_trials,
        perturb_translation_scale=args.perturb_translation_scale,
        optimizer=args.optimizer,
        seed=args.seed,
    )
    payload["metadata"] = {
        "sequence": str(args.sequence),
        "frame": str(args.frame),
        "cad": str(args.cad),
        "hand_pickle": str(args.hand),
        "mano_root": str(args.mano_root),
        "route": "do_as_i_do_single_frame",
    }
    json_path = route_dir / f"retargeted_{args.robot_profile}_do_as_i_do_single_frame.json"
    json_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
