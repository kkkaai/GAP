from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(frozen=True)
class ContactMappingResult:
    """Mapping result from contact targets to robot hand surface samples."""

    scheme: str
    target_points: np.ndarray
    assigned_indices: np.ndarray
    assigned_points: np.ndarray
    assigned_link_names: np.ndarray | None
    distances: np.ndarray
    elapsed_seconds: float
    weights: np.ndarray | None = None
    expected_points: np.ndarray | None = None
    assigned_link_indices: np.ndarray | None = None
    link_names: np.ndarray | None = None
    link_scores: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def nearest_surface_mapping(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    *,
    robot_link_names: np.ndarray | None = None,
) -> ContactMappingResult:
    """Scheme 1: hard nearest robot surface point for each target."""

    start = time.perf_counter()
    targets = _as_points("target_points", target_points)
    robot_points = _as_points("robot_points", robot_points)
    indices, distances = _nearest_indices(robot_points, targets)
    elapsed = time.perf_counter() - start
    link_names = _optional_links(robot_link_names, len(robot_points))
    return ContactMappingResult(
        scheme="nearest_surface",
        target_points=targets,
        assigned_indices=indices,
        assigned_points=robot_points[indices],
        assigned_link_names=link_names[indices] if link_names is not None else None,
        distances=distances,
        elapsed_seconds=elapsed,
        metrics=_basic_metrics(distances),
    )


def soft_surface_mapping(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    *,
    robot_link_names: np.ndarray | None = None,
    temperature: float = 1e-4,
) -> ContactMappingResult:
    """Scheme 2: softmin assignment over all robot surface samples."""

    start = time.perf_counter()
    targets = _as_points("target_points", target_points)
    robot_points = _as_points("robot_points", robot_points)
    weights, dist_sq = _soft_weights(targets, robot_points, temperature)
    expected_points = weights @ robot_points
    assigned_indices = np.argmax(weights, axis=1).astype(np.int64)
    distances = np.sqrt(dist_sq[np.arange(len(targets)), assigned_indices])
    losses = -temperature * _logsumexp(-dist_sq / temperature, axis=1)
    elapsed = time.perf_counter() - start
    link_names = _optional_links(robot_link_names, len(robot_points))
    metrics = _basic_metrics(distances)
    metrics.update(
        {
            "soft_contact_loss_sum": float(losses.sum()),
            "soft_contact_loss_mean": float(losses.mean()) if len(losses) else 0.0,
            "temperature": float(temperature),
        }
    )
    return ContactMappingResult(
        scheme="soft_surface",
        target_points=targets,
        assigned_indices=assigned_indices,
        assigned_points=robot_points[assigned_indices],
        assigned_link_names=link_names[assigned_indices] if link_names is not None else None,
        distances=distances,
        elapsed_seconds=elapsed,
        weights=weights,
        expected_points=expected_points,
        metrics=metrics,
    )


def diverse_soft_surface_mapping(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    *,
    robot_link_names: np.ndarray | None = None,
    temperature: float = 1e-4,
    spread_sigma: float = 0.025,
    diversity_weight: float = 1.0,
) -> ContactMappingResult:
    """Scheme 3: soft assignment plus a spread metric/penalty.

    The returned ``assigned_indices`` are still the maximum soft weights. The
    diversity term is reported as a metric so an optimizer can combine it with
    contact loss later.
    """

    start = time.perf_counter()
    targets = _as_points("target_points", target_points)
    robot_points = _as_points("robot_points", robot_points)
    weights, dist_sq = _soft_weights(targets, robot_points, temperature)
    expected_points = weights @ robot_points
    assigned_indices = np.argmax(weights, axis=1).astype(np.int64)
    distances = np.sqrt(dist_sq[np.arange(len(targets)), assigned_indices])
    losses = -temperature * _logsumexp(-dist_sq / temperature, axis=1)
    spread_loss = _spread_loss(expected_points, spread_sigma)
    total_loss = float(losses.sum() + diversity_weight * spread_loss)
    elapsed = time.perf_counter() - start
    link_names = _optional_links(robot_link_names, len(robot_points))
    metrics = _basic_metrics(distances)
    metrics.update(
        {
            "soft_contact_loss_sum": float(losses.sum()),
            "soft_contact_loss_mean": float(losses.mean()) if len(losses) else 0.0,
            "spread_loss": float(spread_loss),
            "combined_loss": total_loss,
            "temperature": float(temperature),
            "spread_sigma": float(spread_sigma),
            "diversity_weight": float(diversity_weight),
        }
    )
    return ContactMappingResult(
        scheme="diverse_soft_surface",
        target_points=targets,
        assigned_indices=assigned_indices,
        assigned_points=robot_points[assigned_indices],
        assigned_link_names=link_names[assigned_indices] if link_names is not None else None,
        distances=distances,
        elapsed_seconds=elapsed,
        weights=weights,
        expected_points=expected_points,
        metrics=metrics,
    )


def link_soft_surface_mapping(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    robot_link_names: np.ndarray,
    *,
    temperature: float = 1e-4,
    unique_links: bool = False,
) -> ContactMappingResult:
    """Scheme 4: topology/link-grouped softmin assignment.

    This does not encode semantic finger names. It only groups robot samples by
    their MJCF/URDF link name, scores each link by softmin distance, and selects
    the best link/sample for each target.
    """

    start = time.perf_counter()
    targets = _as_points("target_points", target_points)
    robot_points = _as_points("robot_points", robot_points)
    sample_links = _optional_links(robot_link_names, len(robot_points))
    if sample_links is None:
        raise ValueError("robot_link_names is required for link_soft_surface_mapping.")

    unique_link_names = np.asarray(sorted(set(sample_links.astype(str))), dtype=object)
    link_scores = np.zeros((len(targets), len(unique_link_names)), dtype=np.float64)
    per_link_best_sample = np.zeros((len(targets), len(unique_link_names)), dtype=np.int64)

    for link_index, link_name in enumerate(unique_link_names):
        sample_indices = np.nonzero(sample_links.astype(str) == str(link_name))[0]
        if len(sample_indices) == 0:
            link_scores[:, link_index] = np.inf
            continue
        link_points = robot_points[sample_indices]
        _, dist_sq = _soft_weights(targets, link_points, temperature)
        link_scores[:, link_index] = -temperature * _logsumexp(-dist_sq / temperature, axis=1)
        per_link_best_sample[:, link_index] = sample_indices[np.argmin(dist_sq, axis=1)]

    if unique_links:
        assigned_link_indices = _greedy_unique_link_assignment(link_scores)
    else:
        assigned_link_indices = np.argmin(link_scores, axis=1).astype(np.int64)
    assigned_indices = per_link_best_sample[np.arange(len(targets)), assigned_link_indices]
    distances = np.linalg.norm(robot_points[assigned_indices] - targets, axis=1)
    elapsed = time.perf_counter() - start
    metrics = _basic_metrics(distances)
    metrics.update(
        {
            "link_soft_loss_sum": float(link_scores[np.arange(len(targets)), assigned_link_indices].sum()),
            "temperature": float(temperature),
            "unique_links": float(bool(unique_links)),
            "num_links": float(len(unique_link_names)),
        }
    )
    return ContactMappingResult(
        scheme="link_soft_surface_unique" if unique_links else "link_soft_surface",
        target_points=targets,
        assigned_indices=assigned_indices.astype(np.int64),
        assigned_points=robot_points[assigned_indices],
        assigned_link_names=sample_links[assigned_indices],
        distances=distances,
        elapsed_seconds=elapsed,
        assigned_link_indices=assigned_link_indices.astype(np.int64),
        link_names=unique_link_names,
        link_scores=link_scores,
        metrics=metrics,
    )


def run_all_contact_mapping_schemes(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    *,
    robot_link_names: np.ndarray | None = None,
    temperature: float = 1e-4,
    spread_sigma: float = 0.025,
    diversity_weight: float = 1.0,
    unique_links: bool = False,
) -> list[ContactMappingResult]:
    """Run the four mapping schemes with identical inputs."""

    results = [
        nearest_surface_mapping(target_points, robot_points, robot_link_names=robot_link_names),
        soft_surface_mapping(
            target_points,
            robot_points,
            robot_link_names=robot_link_names,
            temperature=temperature,
        ),
        diverse_soft_surface_mapping(
            target_points,
            robot_points,
            robot_link_names=robot_link_names,
            temperature=temperature,
            spread_sigma=spread_sigma,
            diversity_weight=diversity_weight,
        ),
    ]
    if robot_link_names is not None:
        results.append(
            link_soft_surface_mapping(
                target_points,
                robot_points,
                robot_link_names,
                temperature=temperature,
                unique_links=unique_links,
            )
        )
    return results


def mapping_result_to_json_dict(result: ContactMappingResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "scheme": result.scheme,
        "elapsed_seconds": float(result.elapsed_seconds),
        "assigned_indices": result.assigned_indices.astype(int).tolist(),
        "distances": result.distances.astype(float).tolist(),
        "metrics": {key: float(value) for key, value in result.metrics.items()},
    }
    if result.assigned_link_names is not None:
        payload["assigned_link_names"] = result.assigned_link_names.astype(str).tolist()
    if result.assigned_link_indices is not None:
        payload["assigned_link_indices"] = result.assigned_link_indices.astype(int).tolist()
    if result.link_names is not None:
        payload["link_names"] = result.link_names.astype(str).tolist()
    return payload


def _soft_weights(
    target_points: np.ndarray,
    robot_points: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    diff = target_points[:, None, :] - robot_points[None, :, :]
    dist_sq = np.einsum("kmi,kmi->km", diff, diff)
    logits = -dist_sq / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    return weights, dist_sq


def _nearest_indices(points: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) * len(queries) <= 10_000_000:
        diff = queries[:, None, :] - points[None, :, :]
        dist_sq = np.einsum("qpi,qpi->qp", diff, diff)
        indices = np.argmin(dist_sq, axis=1)
        return indices.astype(np.int64), np.sqrt(dist_sq[np.arange(len(queries)), indices]).astype(np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(points).query(queries, k=1)
        return indices.astype(np.int64), distances.astype(np.float64)
    except Exception:
        diff = queries[:, None, :] - points[None, :, :]
        dist_sq = np.einsum("qpi,qpi->qp", diff, diff)
        indices = np.argmin(dist_sq, axis=1)
        return indices.astype(np.int64), np.sqrt(dist_sq[np.arange(len(queries)), indices]).astype(np.float64)


def _spread_loss(points: np.ndarray, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if len(points) < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist_sq = np.einsum("kli,kli->kl", diff, diff)
    upper = np.triu_indices(len(points), k=1)
    return float(np.exp(-dist_sq[upper] / (sigma * sigma)).sum())


def _greedy_unique_link_assignment(link_scores: np.ndarray) -> np.ndarray:
    num_targets, num_links = link_scores.shape
    assigned = np.full(num_targets, -1, dtype=np.int64)
    used_links: set[int] = set()
    order = np.argsort(np.min(link_scores, axis=1))
    for target_index in order:
        candidate_order = np.argsort(link_scores[target_index])
        for link_index in candidate_order:
            if int(link_index) not in used_links:
                assigned[target_index] = int(link_index)
                used_links.add(int(link_index))
                break
        if assigned[target_index] < 0:
            assigned[target_index] = int(candidate_order[0])
    return assigned


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.maximum(np.exp(values - max_values).sum(axis=axis), 1e-300)
    )


def _basic_metrics(distances: np.ndarray) -> dict[str, float]:
    if len(distances) == 0:
        return {"mean_distance": 0.0, "max_distance": 0.0, "min_distance": 0.0}
    return {
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "min_distance": float(np.min(distances)),
    }


def _optional_links(robot_link_names: np.ndarray | None, expected_len: int) -> np.ndarray | None:
    if robot_link_names is None:
        return None
    links = np.asarray(robot_link_names)
    if links.shape != (expected_len,):
        raise ValueError(f"robot_link_names must have shape ({expected_len},), got {links.shape}.")
    return links.astype(object)


def _as_points(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}.")
    return array
