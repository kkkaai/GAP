"""Strict force-closure checks and Ferrari-Canny epsilon scores.

This module is intended for candidate evaluation/ranking after retargeting. It
uses a polyhedral friction-cone approximation, constructs primitive 6D wrenches,
then evaluates whether the origin lies inside their convex hull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceClosureResult:
    """Result of a strict force-closure evaluation."""

    is_force_closure: bool
    epsilon: float
    origin_margin: float
    rank: int
    num_wrenches: int


def _require_scipy():
    try:
        from scipy.optimize import linprog
        from scipy.spatial import ConvexHull, QhullError
    except ImportError as exc:
        raise ImportError(
            "strict force-closure evaluation requires scipy. Install the "
            "project with the flux-stage extra or run: pip install scipy"
        ) from exc
    return linprog, ConvexHull, QhullError


def _as_points(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}")
    return array


def _normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("Cannot normalize zero-length vector.")
    return vectors / norms


def tangent_basis_from_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal tangent basis for each normal."""

    normals = _normalize(_as_points("normals", normals))
    helper = np.zeros_like(normals)
    use_x = np.abs(normals[:, 0]) < 0.9
    helper[use_x] = np.array([1.0, 0.0, 0.0])
    helper[~use_x] = np.array([0.0, 1.0, 0.0])

    tangent_1 = _normalize(np.cross(normals, helper))
    tangent_2 = _normalize(np.cross(normals, tangent_1))
    return tangent_1, tangent_2


def friction_cone_directions(
    outward_normals: np.ndarray,
    *,
    friction_coef: float = 0.5,
    num_edges: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """Approximate each contact's friction cone with edge directions.

    Args:
        outward_normals: Object surface outward normals at contacts, shape
            ``(N, 3)``.
        friction_coef: Coulomb friction coefficient ``mu``.
        num_edges: Number of cone edge directions per contact.
        normalize: If true, normalize each edge direction to unit length.

    Returns:
        Array with shape ``(N, num_edges, 3)``. Directions are forces applied by
        the hand to the object, so they point around ``-outward_normal``.
    """

    if friction_coef < 0:
        raise ValueError("friction_coef must be non-negative")
    if num_edges < 3:
        raise ValueError("num_edges must be at least 3")

    normals = _normalize(_as_points("outward_normals", outward_normals))
    tangent_1, tangent_2 = tangent_basis_from_normals(normals)
    angles = np.linspace(0.0, 2.0 * np.pi, num_edges, endpoint=False)

    directions = []
    for angle in angles:
        tangent = np.cos(angle) * tangent_1 + np.sin(angle) * tangent_2
        directions.append(-normals + friction_coef * tangent)
    cone = np.stack(directions, axis=1)
    return _normalize(cone.reshape(-1, 3)).reshape(cone.shape) if normalize else cone


def primitive_wrenches(
    contact_points: np.ndarray,
    outward_normals: np.ndarray,
    *,
    friction_coef: float = 0.5,
    num_edges: int = 8,
    object_center: np.ndarray | None = None,
    torque_scale: float = 1.0,
    normalize_wrenches: bool = True,
) -> np.ndarray:
    """Construct primitive 6D wrenches from friction-cone edge directions.

    Returns an array of shape ``(N * num_edges, 6)`` with force components first
    and torque components second.
    """

    points = _as_points("contact_points", contact_points)
    normals = _as_points("outward_normals", outward_normals)
    if points.shape != normals.shape:
        raise ValueError(
            "contact_points and outward_normals must have the same shape; "
            f"got {points.shape} and {normals.shape}"
        )

    center = np.zeros(3, dtype=np.float64) if object_center is None else np.asarray(object_center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"object_center must have shape (3,), got {center.shape}")

    directions = friction_cone_directions(
        normals,
        friction_coef=friction_coef,
        num_edges=num_edges,
        normalize=True,
    )
    repeated_points = np.repeat(points, num_edges, axis=0)
    flat_directions = directions.reshape(-1, 3)
    moment_arms = repeated_points - center
    torques = np.cross(moment_arms, flat_directions) * torque_scale
    wrenches = np.concatenate([flat_directions, torques], axis=1)
    return _normalize(wrenches) if normalize_wrenches else wrenches


def origin_in_convex_hull_lp(wrenches: np.ndarray, *, tol: float = 1e-7) -> bool:
    """Check whether the origin is in the convex hull using linear programming."""

    linprog, _, _ = _require_scipy()
    wrenches = np.asarray(wrenches, dtype=np.float64)
    if wrenches.ndim != 2 or wrenches.shape[1] != 6:
        raise ValueError(f"wrenches must have shape (M, 6), got {wrenches.shape}")

    num_wrenches = wrenches.shape[0]
    equality = np.vstack([wrenches.T, np.ones((1, num_wrenches))])
    rhs = np.zeros(7, dtype=np.float64)
    rhs[-1] = 1.0
    result = linprog(
        c=np.zeros(num_wrenches, dtype=np.float64),
        A_eq=equality,
        b_eq=rhs,
        bounds=[(0.0, None)] * num_wrenches,
        method="highs",
    )
    if not result.success:
        return False
    residual = np.linalg.norm(equality @ result.x - rhs)
    return bool(residual <= tol)


def ferrari_canny_epsilon(wrenches: np.ndarray, *, qhull_options: str = "QJ") -> tuple[float, float]:
    """Compute Ferrari-Canny epsilon from a wrench convex hull.

    Returns:
        ``(epsilon, origin_margin)``. ``epsilon`` is positive only when the
        origin is inside the convex hull. ``origin_margin`` is the signed minimum
        inward distance from the origin to the hull facets; positive means inside.
    """

    _, ConvexHull, QhullError = _require_scipy()
    wrenches = np.asarray(wrenches, dtype=np.float64)
    if wrenches.ndim != 2 or wrenches.shape[1] != 6:
        raise ValueError(f"wrenches must have shape (M, 6), got {wrenches.shape}")
    if wrenches.shape[0] < 8:
        return 0.0, -np.inf

    rank = int(np.linalg.matrix_rank(wrenches, tol=1e-8))
    if rank < 6:
        return 0.0, -np.inf

    try:
        hull = ConvexHull(wrenches, qhull_options=qhull_options)
    except QhullError:
        return 0.0, -np.inf

    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    normal_norms = np.linalg.norm(normals, axis=1)
    valid = normal_norms > 1e-12
    if not np.any(valid):
        return 0.0, -np.inf

    # scipy facets satisfy normal dot x + offset <= 0 for points inside.
    # At x=0 this gives offset <= 0. The positive inward margin is -offset / ||normal||.
    margins = -offsets[valid] / normal_norms[valid]
    origin_margin = float(np.min(margins))
    epsilon = max(0.0, origin_margin)
    return float(epsilon), origin_margin


def evaluate_force_closure(
    contact_points: np.ndarray,
    outward_normals: np.ndarray,
    *,
    friction_coef: float = 0.5,
    num_edges: int = 8,
    object_center: np.ndarray | None = None,
    torque_scale: float = 1.0,
    normalize_wrenches: bool = True,
    tol: float = 1e-7,
) -> ForceClosureResult:
    """Evaluate strict force closure and Ferrari-Canny epsilon for one grasp."""

    wrenches = primitive_wrenches(
        contact_points,
        outward_normals,
        friction_coef=friction_coef,
        num_edges=num_edges,
        object_center=object_center,
        torque_scale=torque_scale,
        normalize_wrenches=normalize_wrenches,
    )
    rank = int(np.linalg.matrix_rank(wrenches, tol=1e-8))
    if rank < 6:
        return ForceClosureResult(
            is_force_closure=False,
            epsilon=0.0,
            origin_margin=-np.inf,
            rank=rank,
            num_wrenches=int(wrenches.shape[0]),
        )

    epsilon, origin_margin = ferrari_canny_epsilon(wrenches)
    inside_by_lp = origin_in_convex_hull_lp(wrenches, tol=tol)
    is_force_closure = bool(inside_by_lp and origin_margin > tol)
    return ForceClosureResult(
        is_force_closure=is_force_closure,
        epsilon=epsilon if is_force_closure else 0.0,
        origin_margin=origin_margin,
        rank=rank,
        num_wrenches=int(wrenches.shape[0]),
    )

