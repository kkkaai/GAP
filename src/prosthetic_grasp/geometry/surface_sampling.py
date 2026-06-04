from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshSurfaceTopology:
    face_indices: np.ndarray
    barycentric: np.ndarray


@dataclass(frozen=True)
class MeshSurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_points: int = 2000,
    seed: int | None = 7,
    use_farthest_point_sampling: bool = True,
    oversample_factor: int = 20,
) -> MeshSurfaceSamples:
    topology = sample_mesh_surface_topology(
        vertices,
        faces,
        num_points=num_points,
        seed=seed,
        use_farthest_point_sampling=use_farthest_point_sampling,
        oversample_factor=oversample_factor,
    )
    return materialize_mesh_surface_samples(vertices, faces, topology)


def sample_mesh_surface_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_points: int = 2000,
    seed: int | None = 7,
    use_farthest_point_sampling: bool = True,
    oversample_factor: int = 20,
) -> MeshSurfaceTopology:
    vertices = as_vertices(vertices)
    faces = as_faces(faces)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    if oversample_factor <= 0:
        raise ValueError(f"oversample_factor must be positive, got {oversample_factor}.")

    rng = np.random.default_rng(seed)
    num_candidates = num_points
    if use_farthest_point_sampling:
        num_candidates = max(num_points, int(num_points * oversample_factor))

    candidates = _sample_surface_candidates(vertices, faces, num_candidates, rng)
    if use_farthest_point_sampling and num_candidates > num_points:
        keep = farthest_point_sample(candidates.points, num_points, rng=rng)
        return MeshSurfaceTopology(
            face_indices=candidates.face_indices[keep],
            barycentric=candidates.barycentric[keep],
        )
    return MeshSurfaceTopology(
        face_indices=candidates.face_indices,
        barycentric=candidates.barycentric,
    )


def materialize_mesh_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology: MeshSurfaceTopology | MeshSurfaceSamples,
) -> MeshSurfaceSamples:
    vertices = as_vertices(vertices)
    faces = as_faces(faces)
    face_indices = np.asarray(topology.face_indices, dtype=np.int64)
    barycentric = np.asarray(topology.barycentric, dtype=np.float64)
    if face_indices.ndim != 1:
        raise ValueError(f"face_indices must have shape (N,), got {face_indices.shape}.")
    if barycentric.shape != (len(face_indices), 3):
        raise ValueError(
            "barycentric must have shape (N, 3) matching face_indices; "
            f"got {barycentric.shape} and {face_indices.shape}."
        )
    if np.any(face_indices < 0) or np.any(face_indices >= len(faces)):
        raise ValueError("face_indices contain values outside the faces array.")

    triangles = vertices[faces[face_indices]]
    points = np.einsum("ni,nij->nj", barycentric, triangles)
    normals_raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = normals_raw / np.maximum(np.linalg.norm(normals_raw, axis=1, keepdims=True), 1e-12)
    return MeshSurfaceSamples(
        points=points.astype(np.float64),
        normals=normals.astype(np.float64),
        face_indices=face_indices.astype(np.int64),
        barycentric=barycentric.astype(np.float64),
    )


def farthest_point_sample(
    points: np.ndarray,
    num_points: int,
    *,
    rng: np.random.Generator | None = None,
    start_index: int | None = None,
) -> np.ndarray:
    points = as_vertices(points)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    if num_points > len(points):
        raise ValueError(f"num_points cannot exceed available points; got {num_points} > {len(points)}.")

    rng = rng or np.random.default_rng()
    selected = np.empty(num_points, dtype=np.int64)
    next_index = int(rng.integers(0, len(points))) if start_index is None else int(start_index)
    if next_index < 0 or next_index >= len(points):
        raise ValueError(f"start_index out of range: {start_index}.")

    min_dist_sq = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(num_points):
        selected[i] = next_index
        diff = points - points[next_index]
        min_dist_sq = np.minimum(min_dist_sq, np.einsum("ij,ij->i", diff, diff))
        next_index = int(np.argmax(min_dist_sq))
    return selected


def triangle_double_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = as_vertices(vertices)
    faces = as_faces(faces)
    tri_vertices = vertices[faces]
    return np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )


def as_vertices(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {array.shape}.")
    return array


def as_faces(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected face shape (F, 3), got {array.shape}.")
    return array


def _sample_surface_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> MeshSurfaceSamples:
    tri_vertices = vertices[faces]
    double_areas = triangle_double_areas(vertices, faces)
    valid = double_areas > 1e-14
    if not np.any(valid):
        raise ValueError("Mesh has no non-degenerate triangles.")

    valid_faces = np.nonzero(valid)[0]
    probabilities = double_areas[valid] / double_areas[valid].sum()
    chosen_faces = valid_faces[rng.choice(len(valid_faces), size=num_points, replace=True, p=probabilities)]
    chosen_triangles = tri_vertices[chosen_faces]

    u = rng.random(num_points)
    v = rng.random(num_points)
    sqrt_u = np.sqrt(u)
    barycentric = np.stack([1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=1)
    points = np.einsum("ni,nij->nj", barycentric, chosen_triangles)
    normals_raw = np.cross(chosen_triangles[:, 1] - chosen_triangles[:, 0], chosen_triangles[:, 2] - chosen_triangles[:, 0])
    normals = normals_raw / np.maximum(np.linalg.norm(normals_raw, axis=1, keepdims=True), 1e-12)
    return MeshSurfaceSamples(
        points=points.astype(np.float64),
        normals=normals.astype(np.float64),
        face_indices=chosen_faces.astype(np.int64),
        barycentric=barycentric.astype(np.float64),
    )
