from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from .contact_clustering import ManoObjectContactResult, closest_points_on_mesh
from .contact_mapping import ContactMappingResult, soft_surface_mapping
from .mano_surface import ManoSurfaceTopology, materialize_mano_surface_samples
from .robot_surface import RobotSurfaceSamples, RobotSurfaceTopology

try:
    from utils.force_closure.dexgraspnet_fc import dexgraspnet_force_closure_energy
    from utils.force_closure.strict_fc import ForceClosureResult, evaluate_force_closure
except ImportError:  # pragma: no cover - lets geometry import without optional utils path.
    dexgraspnet_force_closure_energy = None
    ForceClosureResult = Any
    evaluate_force_closure = None


@dataclass
class ContactPatchTarget:
    """One clustered MANO-object contact patch and its robot assignment."""

    cluster_label: int
    patch_size: int
    sample_indices: np.ndarray
    representative_sample_index: int
    mano_face_index: int
    mano_barycentric: np.ndarray
    mano_point_posed: np.ndarray
    object_point_target: np.ndarray
    object_normal_target: np.ndarray
    object_point_nearest: np.ndarray
    object_normal_nearest: np.ndarray
    object_point_center: np.ndarray
    object_normal_center: np.ndarray
    contact_distance: float
    mano_point_canonical: np.ndarray | None = None
    canonical_robot_target: np.ndarray | None = None
    robot_assignment_indices: np.ndarray | None = None
    robot_assignment_weights: np.ndarray | None = None
    robot_assignment_link_names: np.ndarray | None = None


@dataclass(frozen=True)
class ManoToRobotFrameMapping:
    mapped_points: np.ndarray
    scale: float
    mano_mid_length: float
    robot_mid_length: float
    robot_tip_links: tuple[str, str, str]


@dataclass(frozen=True)
class RetargetOptimizationConfig:
    """Weights and optimizer settings for contact retargeting."""

    contact_weight: float = 1.0
    normal_weight: float = 0.02
    fc_weight: float = 0.05
    regularization_weight: float = 1e-4
    maxiter_stage1: int = 80
    maxiter_stage2: int = 80
    num_random_restarts: int = 0
    random_start_scale: float = 0.15
    seed: int | None = 7
    optimizer_method: str = "L-BFGS-B"
    object_center: np.ndarray | None = None
    torque_scale: float = 1.0
    friction_coef: float = 0.5
    num_friction_edges: int = 8


@dataclass(frozen=True)
class RetargetCandidate:
    action: np.ndarray
    stage1_loss: float
    stage2_loss: float
    loss_terms: dict[str, float]
    success: bool
    message: str
    iterations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RetargetOptimizationResult:
    candidates: list[RetargetCandidate]
    best_index: int
    config: RetargetOptimizationConfig

    @property
    def best(self) -> RetargetCandidate:
        return self.candidates[self.best_index]


@dataclass(frozen=True)
class CandidateForceClosureScore:
    candidate_index: int
    force_closure: ForceClosureResult
    dex_fc_energy: float
    contact_points: np.ndarray
    object_contact_points: np.ndarray
    object_contact_normals: np.ndarray
    projection_distances: np.ndarray
    mean_projection_distance: float


def build_contact_patch_targets(
    contact_result: ManoObjectContactResult,
    *,
    representative: str = "center",
) -> list[ContactPatchTarget]:
    """Convert MANO-object contact clusters into patch-level retarget targets.

    ``object_point_target`` is the stable patch representative used by contact
    position optimization. The nearest and center object representatives are
    both retained so the caller can compare strategies without recomputing
    clustering.
    """

    if representative not in {"center", "nearest"}:
        raise ValueError(f"representative must be 'center' or 'nearest', got {representative!r}.")

    patches: list[ContactPatchTarget] = []
    samples = contact_result.mano_samples
    for cluster in contact_result.clusters:
        local_index = cluster.center_index if representative == "center" else cluster.nearest_index
        rep_sample_index = int(cluster.sample_indices[local_index])
        object_point_target = cluster.center_object_point if representative == "center" else cluster.nearest_object_point
        object_normal_target = (
            cluster.center_object_normal if representative == "center" else cluster.nearest_object_normal
        )
        patches.append(
            ContactPatchTarget(
                cluster_label=int(cluster.label),
                patch_size=int(len(cluster.sample_indices)),
                sample_indices=np.asarray(cluster.sample_indices, dtype=np.int64),
                representative_sample_index=rep_sample_index,
                mano_face_index=int(samples.face_indices[rep_sample_index]),
                mano_barycentric=np.asarray(samples.barycentric[rep_sample_index], dtype=np.float64),
                mano_point_posed=np.asarray(cluster.mano_points[local_index], dtype=np.float64),
                object_point_target=np.asarray(object_point_target, dtype=np.float64),
                object_normal_target=_normalize_one(object_normal_target),
                object_point_nearest=np.asarray(cluster.nearest_object_point, dtype=np.float64),
                object_normal_nearest=_normalize_one(cluster.nearest_object_normal),
                object_point_center=np.asarray(cluster.center_object_point, dtype=np.float64),
                object_normal_center=_normalize_one(cluster.center_object_normal),
                contact_distance=float(cluster.distances[local_index]),
            )
        )
    return patches


def canonicalize_patch_targets(
    patches: list[ContactPatchTarget],
    mano_zero_vertices: np.ndarray,
    mano_faces: np.ndarray,
) -> list[ContactPatchTarget]:
    """Recover each patch representative on the zero/canonical MANO mesh."""

    if not patches:
        return patches
    topology = ManoSurfaceTopology(
        face_indices=np.asarray([patch.mano_face_index for patch in patches], dtype=np.int64),
        barycentric=np.asarray([patch.mano_barycentric for patch in patches], dtype=np.float64),
    )
    samples = materialize_mano_surface_samples(mano_zero_vertices, mano_faces, topology)
    for patch, point in zip(patches, samples.points):
        patch.mano_point_canonical = np.asarray(point, dtype=np.float64)
    return patches


def map_canonical_mano_patches_to_robot_frame(
    patches: list[ContactPatchTarget],
    mano_zero_keypoints: np.ndarray,
    robot_zero_surface: RobotSurfaceSamples,
    *,
    index_tip_link: str = "robot0:ffdistal",
    middle_tip_link: str = "robot0:mfdistal",
    little_tip_link: str = "robot0:lfdistal",
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> ManoToRobotFrameMapping:
    """Map canonical MANO patch points into the robot wrist frame."""

    canonical_points = _patch_array(patches, "mano_point_canonical")
    mano_keypoints = _as_points("mano_zero_keypoints", mano_zero_keypoints)
    robot_points = _as_points("robot_zero_surface.points", robot_zero_surface.points)
    robot_links = np.asarray(robot_zero_surface.link_names).astype(str)

    mano_frame = build_hand_frame(
        wrist=mano_keypoints[0],
        index_mcp=mano_keypoints[5],
        middle_mcp=mano_keypoints[9],
        little_mcp=mano_keypoints[17],
    )
    robot_index = robot_tip_from_link(robot_points, robot_links, index_tip_link)
    robot_middle = robot_tip_from_link(robot_points, robot_links, middle_tip_link)
    robot_little = robot_tip_from_link(robot_points, robot_links, little_tip_link)
    robot_frame = build_hand_frame(
        wrist=np.zeros(3, dtype=np.float64),
        index_mcp=robot_index,
        middle_mcp=robot_middle,
        little_mcp=robot_little,
    )

    mano_local = (canonical_points - mano_keypoints[0]) @ mano_frame
    mano_mid = max(float(np.linalg.norm(mano_keypoints[12] - mano_keypoints[0])), 1e-8)
    robot_mid = max(float(np.linalg.norm(robot_middle)), 1e-8)
    scale = float(np.clip(robot_mid / mano_mid, min_scale, max_scale))
    mapped = (mano_local * scale) @ robot_frame.T
    for patch, point in zip(patches, mapped):
        patch.canonical_robot_target = np.asarray(point, dtype=np.float64)
    return ManoToRobotFrameMapping(
        mapped_points=mapped.astype(np.float64),
        scale=scale,
        mano_mid_length=mano_mid,
        robot_mid_length=robot_mid,
        robot_tip_links=(index_tip_link, middle_tip_link, little_tip_link),
    )


def assign_patches_to_robot_surface(
    patches: list[ContactPatchTarget],
    robot_zero_surface: RobotSurfaceSamples,
    *,
    method: str = "soft_surface",
    top_k: int = 32,
    temperature: float = 1e-4,
) -> ContactMappingResult:
    """Assign canonical patch targets to fixed robot surface samples.

    The default is ``soft_surface``. The full soft assignment is reduced to a
    top-k support per patch so optimization can repeatedly materialize only the
    relevant robot surface region.
    """

    if method != "soft_surface":
        raise ValueError("Only method='soft_surface' is implemented as the default retarget assignment.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    target_points = _patch_array(patches, "canonical_robot_target")
    result = soft_surface_mapping(
        target_points,
        robot_zero_surface.points,
        robot_link_names=robot_zero_surface.link_names,
        temperature=temperature,
    )
    if result.weights is None:
        raise RuntimeError("soft_surface_mapping did not return weights.")

    weights = np.asarray(result.weights, dtype=np.float64)
    k = min(int(top_k), weights.shape[1])
    link_names = np.asarray(robot_zero_surface.link_names).astype(str)
    for patch_index, patch in enumerate(patches):
        row = weights[patch_index]
        if k == len(row):
            indices = np.arange(len(row), dtype=np.int64)
        else:
            indices = np.argpartition(-row, kth=k - 1)[:k].astype(np.int64)
        order = np.argsort(-row[indices])
        indices = indices[order]
        selected_weights = row[indices]
        selected_weights = selected_weights / max(float(selected_weights.sum()), 1e-300)
        patch.robot_assignment_indices = indices.astype(np.int64)
        patch.robot_assignment_weights = selected_weights.astype(np.float64)
        patch.robot_assignment_link_names = link_names[indices].astype(object)
    return result


def materialize_assigned_robot_contacts(
    patches: list[ContactPatchTarget],
    robot_surface: RobotSurfaceSamples,
) -> tuple[np.ndarray, np.ndarray]:
    """Return soft-region robot contact points and normals for the current q."""

    if not patches:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    points = []
    normals = []
    for patch in patches:
        if patch.robot_assignment_indices is None or patch.robot_assignment_weights is None:
            raise ValueError("Patch has no robot assignment. Call assign_patches_to_robot_surface first.")
        indices = patch.robot_assignment_indices
        weights = patch.robot_assignment_weights[:, None]
        points.append(np.sum(robot_surface.points[indices] * weights, axis=0))
        normal = np.sum(robot_surface.normals[indices] * weights, axis=0)
        normals.append(_normalize_one(normal))
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def retarget_loss_terms(
    action: np.ndarray,
    robot_model: Any,
    robot_topology: RobotSurfaceTopology,
    patches: list[ContactPatchTarget],
    config: RetargetOptimizationConfig,
    *,
    include_fc: bool = False,
    reference_action: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate contact, normal, regularization, and optional DexGraspNet FC losses."""

    action = np.asarray(action, dtype=np.float64)
    reference = robot_model.zero_action if reference_action is None else np.asarray(reference_action, dtype=np.float64)
    robot_surface = robot_model.materialize_surface(robot_topology, action)
    contact_points, contact_normals = materialize_assigned_robot_contacts(patches, robot_surface)
    object_points = _patch_array(patches, "object_point_target")
    object_normals = _patch_array(patches, "object_normal_target")

    if len(contact_points) == 0:
        return {
            "contact": 0.0,
            "normal": 0.0,
            "regularization": 0.0,
            "dex_fc": 0.0,
            "total": 0.0,
        }

    contact_loss = float(np.mean(np.sum((contact_points - object_points) ** 2, axis=1)))
    normal_dot = np.sum(contact_normals * object_normals, axis=1)
    normal_loss = float(np.mean((normal_dot + 1.0) ** 2))
    lower, upper = robot_model.joint_bounds()
    span = np.maximum(upper - lower, 1e-8)
    regularization_loss = float(np.mean(((action - reference) / span) ** 2)) if len(action) else 0.0

    dex_fc_loss = 0.0
    if include_fc and config.fc_weight > 0.0:
        if dexgraspnet_force_closure_energy is None:
            raise ImportError("DexGraspNet force-closure energy is not importable from utils.force_closure.")
        center = config.object_center
        if center is None:
            center = object_points.mean(axis=0)
        dex_fc_loss = float(
            dexgraspnet_force_closure_energy(
                contact_points,
                -object_normals,
                object_center=center,
                torque_scale=config.torque_scale,
                reduction="mean",
            )
        )

    total = (
        config.contact_weight * contact_loss
        + config.normal_weight * normal_loss
        + config.regularization_weight * regularization_loss
        + (config.fc_weight * dex_fc_loss if include_fc else 0.0)
    )
    return {
        "contact": contact_loss,
        "normal": normal_loss,
        "regularization": regularization_loss,
        "dex_fc": dex_fc_loss,
        "total": float(total),
    }


def optimize_retarget_action(
    robot_model: Any,
    robot_topology: RobotSurfaceTopology,
    patches: list[ContactPatchTarget],
    *,
    config: RetargetOptimizationConfig | None = None,
    initial_action: np.ndarray | None = None,
) -> RetargetOptimizationResult:
    """Two-stage optimization: contact first, then contact + DexGraspNet E_fc."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("optimize_retarget_action requires scipy.") from exc

    config = config or RetargetOptimizationConfig()
    lower, upper = robot_model.joint_bounds()
    base = robot_model.zero_action if initial_action is None else np.asarray(initial_action, dtype=np.float64)
    base = np.clip(base, lower, upper)
    starts = [base]
    rng = np.random.default_rng(config.seed)
    for _ in range(config.num_random_restarts):
        span = np.maximum(upper - lower, 1e-8)
        starts.append(np.clip(base + rng.normal(size=len(base)) * span * config.random_start_scale, lower, upper))

    candidates: list[RetargetCandidate] = []
    for start in starts:
        elapsed_start = time.perf_counter()
        stage1 = minimize(
            lambda q: retarget_loss_terms(
                q,
                robot_model,
                robot_topology,
                patches,
                config,
                include_fc=False,
                reference_action=base,
            )["total"],
            start,
            method=config.optimizer_method,
            bounds=list(zip(lower, upper)),
            options={"maxiter": int(config.maxiter_stage1), "ftol": 1e-10},
        )
        stage1_action = np.asarray(stage1.x, dtype=np.float64)
        if config.fc_weight > 0.0 and config.maxiter_stage2 > 0:
            stage2 = minimize(
                lambda q: retarget_loss_terms(
                    q,
                    robot_model,
                    robot_topology,
                    patches,
                    config,
                    include_fc=True,
                    reference_action=base,
                )["total"],
                stage1_action,
                method=config.optimizer_method,
                bounds=list(zip(lower, upper)),
                options={"maxiter": int(config.maxiter_stage2), "ftol": 1e-10},
            )
        else:
            stage2 = stage1

        action = np.asarray(stage2.x, dtype=np.float64)
        terms = retarget_loss_terms(
            action,
            robot_model,
            robot_topology,
            patches,
            config,
            include_fc=config.fc_weight > 0.0,
            reference_action=base,
        )
        candidates.append(
            RetargetCandidate(
                action=action,
                stage1_loss=float(stage1.fun),
                stage2_loss=float(stage2.fun),
                loss_terms=terms,
                success=bool(stage1.success and stage2.success),
                message=f"stage1={stage1.message}; stage2={stage2.message}",
                iterations=int(getattr(stage1, "nit", 0) + getattr(stage2, "nit", 0)),
                elapsed_seconds=float(time.perf_counter() - elapsed_start),
            )
        )

    if not candidates:
        raise RuntimeError("No retarget candidates were generated.")
    best_index = int(np.argmin([candidate.loss_terms["total"] for candidate in candidates]))
    return RetargetOptimizationResult(candidates=candidates, best_index=best_index, config=config)


def rank_candidates_by_strict_force_closure(
    candidates: list[RetargetCandidate],
    robot_model: Any,
    robot_topology: RobotSurfaceTopology,
    patches: list[ContactPatchTarget],
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    config: RetargetOptimizationConfig | None = None,
) -> list[CandidateForceClosureScore]:
    """Project optimized robot contacts to the object mesh and rank by strict FC."""

    if evaluate_force_closure is None:
        raise ImportError("Strict force-closure evaluation is not importable from utils.force_closure.")
    config = config or RetargetOptimizationConfig()
    object_points_target = _patch_array(patches, "object_point_target")
    object_center = config.object_center
    if object_center is None:
        object_center = object_points_target.mean(axis=0) if len(object_points_target) else np.zeros(3)

    scores: list[CandidateForceClosureScore] = []
    for candidate_index, candidate in enumerate(candidates):
        robot_surface = robot_model.materialize_surface(robot_topology, candidate.action)
        contact_points, _ = materialize_assigned_robot_contacts(patches, robot_surface)
        projected_points, projected_normals, projection_distances = closest_points_on_mesh(
            object_vertices,
            object_faces,
            contact_points,
        )
        dex_fc_energy = 0.0
        if dexgraspnet_force_closure_energy is not None and len(projected_points):
            dex_fc_energy = float(
                dexgraspnet_force_closure_energy(
                    projected_points,
                    -projected_normals,
                    object_center=object_center,
                    torque_scale=config.torque_scale,
                    reduction="mean",
                )
            )
        fc = evaluate_force_closure(
            projected_points,
            projected_normals,
            friction_coef=config.friction_coef,
            num_edges=config.num_friction_edges,
            object_center=object_center,
            torque_scale=config.torque_scale,
        )
        scores.append(
            CandidateForceClosureScore(
                candidate_index=candidate_index,
                force_closure=fc,
                dex_fc_energy=dex_fc_energy,
                contact_points=contact_points,
                object_contact_points=projected_points,
                object_contact_normals=projected_normals,
                projection_distances=projection_distances,
                mean_projection_distance=float(np.mean(projection_distances)) if len(projection_distances) else 0.0,
            )
        )

    return sorted(
        scores,
        key=lambda score: (
            bool(score.force_closure.is_force_closure),
            float(score.force_closure.epsilon),
            -float(score.dex_fc_energy),
            -float(score.mean_projection_distance),
        ),
        reverse=True,
    )


def build_hand_frame(
    *,
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    little_mcp: np.ndarray,
) -> np.ndarray:
    forward = _normalize_one(np.asarray(middle_mcp, dtype=np.float64) - np.asarray(wrist, dtype=np.float64))
    lateral = _normalize_one(np.asarray(index_mcp, dtype=np.float64) - np.asarray(little_mcp, dtype=np.float64))
    normal = _normalize_one(np.cross(lateral, forward))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = _normalize_one(np.cross(forward, normal))
    return np.stack([lateral, forward, normal], axis=1)


def robot_tip_from_link(robot_points: np.ndarray, robot_link_names: np.ndarray, link_name: str) -> np.ndarray:
    links = np.asarray(robot_link_names).astype(str)
    mask = links == str(link_name)
    if not np.any(mask):
        raise ValueError(f"robot_link_names does not contain {link_name!r}.")
    points = _as_points("robot_points", robot_points)[mask]
    return points[np.argmax(np.linalg.norm(points, axis=1))]


def _patch_array(patches: list[ContactPatchTarget], field_name: str) -> np.ndarray:
    values = []
    for patch in patches:
        value = getattr(patch, field_name)
        if value is None:
            raise ValueError(f"Patch {patch.cluster_label} has no {field_name}.")
        values.append(np.asarray(value, dtype=np.float64))
    return _as_points(field_name, np.asarray(values, dtype=np.float64)) if values else np.zeros((0, 3))


def _normalize_one(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector
    return vector / norm


def _as_points(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}.")
    return array
