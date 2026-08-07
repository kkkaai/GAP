"""DexGraspNet-style differentiable force-closure energy.

This module implements the lightweight proxy used by DexGraspNet:

    E_fc = ||G c||^2

where each contact normal ``c_i`` is treated as a unit contact force and
``G`` maps contact forces to object wrenches. It is intentionally simple and
optimization-friendly. It does not perform a strict force-closure test.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Reduction = Literal["mean", "sum", "none"]


def _is_torch_tensor(value: object) -> bool:
    return value.__class__.__module__.startswith("torch")


def _normalize_vectors(vectors, eps: float):
    if _is_torch_tensor(vectors):
        import torch

        return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=eps)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), eps)


def _sum_last(values):
    if _is_torch_tensor(values):
        return values.sum(dim=-1)
    return np.sum(values, axis=-1)


def _reduce(values, reduction: Reduction):
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def dexgraspnet_force_closure_energy(
    contact_points,
    contact_normals,
    *,
    object_center=None,
    normalize_normals: bool = True,
    torque_scale: float = 1.0,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
):
    """Compute the DexGraspNet force-closure proxy energy.

    Args:
        contact_points: Contact positions with shape ``(..., N, 3)``.
        contact_normals: Contact force/normal vectors with shape ``(..., N, 3)``.
            Use inward normals if you already have them. If your mesh normals are
            outward normals, pass ``-outward_normals``.
        object_center: Object-frame wrench origin with shape ``(3,)`` or
            ``(..., 3)``. Defaults to the coordinate origin.
        normalize_normals: Normalize contact vectors before forming wrenches.
        torque_scale: Multiplier for torque components. Use this to balance force
            units and torque units, e.g. ``1 / object_radius``.
        reduction: ``"none"`` returns one energy per batch item, ``"mean"`` and
            ``"sum"`` reduce all batch energies.
        eps: Numerical epsilon for vector normalization.

    Returns:
        Scalar energy for reduced modes, otherwise shape ``(...)``. The return
        type follows the input type: ``torch.Tensor`` stays differentiable;
        numpy input returns ``np.ndarray``/scalar.
    """

    if _is_torch_tensor(contact_points) or _is_torch_tensor(contact_normals):
        import torch

        points = contact_points
        normals = contact_normals
        if not _is_torch_tensor(points):
            points = torch.as_tensor(points, dtype=normals.dtype, device=normals.device)
        if not _is_torch_tensor(normals):
            normals = torch.as_tensor(normals, dtype=points.dtype, device=points.device)

        if object_center is None:
            center = torch.zeros(3, dtype=points.dtype, device=points.device)
        elif _is_torch_tensor(object_center):
            center = object_center.to(dtype=points.dtype, device=points.device)
        else:
            center = torch.as_tensor(object_center, dtype=points.dtype, device=points.device)

        forces = _normalize_vectors(normals, eps) if normalize_normals else normals
        moment_arms = points - center[..., None, :] if center.ndim > 1 else points - center
        force_sum = forces.sum(dim=-2)
        torque_sum = torch.cross(moment_arms, forces, dim=-1).sum(dim=-2) * torque_scale
        wrench = torch.cat([force_sum, torque_sum], dim=-1)
        return _reduce(_sum_last(wrench * wrench), reduction)

    points = np.asarray(contact_points, dtype=np.float64)
    normals = np.asarray(contact_normals, dtype=np.float64)
    if points.shape[-1] != 3 or normals.shape[-1] != 3:
        raise ValueError("contact_points and contact_normals must have shape (..., N, 3)")
    if points.shape != normals.shape:
        raise ValueError(
            "contact_points and contact_normals must have the same shape; "
            f"got {points.shape} and {normals.shape}"
        )

    center = np.zeros(3, dtype=np.float64) if object_center is None else np.asarray(object_center, dtype=np.float64)
    forces = _normalize_vectors(normals, eps) if normalize_normals else normals
    moment_arms = points - center[..., None, :] if center.ndim > 1 else points - center
    force_sum = np.sum(forces, axis=-2)
    torque_sum = np.sum(np.cross(moment_arms, forces), axis=-2) * torque_scale
    wrench = np.concatenate([force_sum, torque_sum], axis=-1)
    return _reduce(np.sum(wrench * wrench, axis=-1), reduction)


def dexgraspnet_wrench(
    contact_points,
    contact_normals,
    *,
    object_center=None,
    normalize_normals: bool = True,
    torque_scale: float = 1.0,
    eps: float = 1e-8,
):
    """Return the 6D total wrench used inside ``E_fc``.

    The output shape is ``(..., 6)`` with force first and torque second.
    """

    if _is_torch_tensor(contact_points) or _is_torch_tensor(contact_normals):
        import torch

        points = contact_points
        normals = contact_normals
        if not _is_torch_tensor(points):
            points = torch.as_tensor(points, dtype=normals.dtype, device=normals.device)
        if not _is_torch_tensor(normals):
            normals = torch.as_tensor(normals, dtype=points.dtype, device=points.device)
        center = (
            torch.zeros(3, dtype=points.dtype, device=points.device)
            if object_center is None
            else torch.as_tensor(object_center, dtype=points.dtype, device=points.device)
        )
        forces = _normalize_vectors(normals, eps) if normalize_normals else normals
        moment_arms = points - center[..., None, :] if center.ndim > 1 else points - center
        return torch.cat(
            [
                forces.sum(dim=-2),
                torch.cross(moment_arms, forces, dim=-1).sum(dim=-2) * torque_scale,
            ],
            dim=-1,
        )

    points = np.asarray(contact_points, dtype=np.float64)
    normals = np.asarray(contact_normals, dtype=np.float64)
    center = np.zeros(3, dtype=np.float64) if object_center is None else np.asarray(object_center, dtype=np.float64)
    forces = _normalize_vectors(normals, eps) if normalize_normals else normals
    moment_arms = points - center[..., None, :] if center.ndim > 1 else points - center
    return np.concatenate(
        [
            np.sum(forces, axis=-2),
            np.sum(np.cross(moment_arms, forces), axis=-2) * torque_scale,
        ],
        axis=-1,
    )

