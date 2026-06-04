from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mano_surface import ManoSurfaceSamples, sample_mano_surface


@dataclass(frozen=True)
class ContactCluster:
    label: int
    sample_indices: np.ndarray
    mano_points: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray
    distances: np.ndarray
    nearest_index: int
    center_index: int

    @property
    def nearest_mano_point(self) -> np.ndarray:
        return self.mano_points[self.nearest_index]

    @property
    def nearest_object_point(self) -> np.ndarray:
        return self.object_points[self.nearest_index]

    @property
    def nearest_object_normal(self) -> np.ndarray:
        return self.object_normals[self.nearest_index]

    @property
    def center_mano_point(self) -> np.ndarray:
        return self.mano_points[self.center_index]

    @property
    def center_object_point(self) -> np.ndarray:
        return self.object_points[self.center_index]

    @property
    def center_object_normal(self) -> np.ndarray:
        return self.object_normals[self.center_index]


@dataclass(frozen=True)
class ManoObjectContactResult:
    mano_samples: ManoSurfaceSamples
    contact_mask: np.ndarray
    contact_sample_indices: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray
    distances: np.ndarray
    cluster_labels: np.ndarray
    clusters: list[ContactCluster]


def extract_mano_object_contact_clusters(
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    num_mano_samples: int = 2000,
    contact_threshold: float = 0.005,
    cluster_radius: float = 0.012,
    min_cluster_size: int = 3,
    cluster_space: str = "object",
    seed: int | None = 7,
    use_farthest_point_sampling: bool = True,
    oversample_factor: int = 20,
) -> ManoObjectContactResult:
    """Sample MANO, filter near-object contacts, and cluster contact patches.

    This intentionally does not use semantic hand regions. Contacts are grouped
    by spatial connected components, which is a fast DBSCAN-like choice for the
    current use case.
    """

    if contact_threshold <= 0:
        raise ValueError(f"contact_threshold must be positive, got {contact_threshold}.")
    if cluster_radius <= 0:
        raise ValueError(f"cluster_radius must be positive, got {cluster_radius}.")
    if min_cluster_size <= 0:
        raise ValueError(f"min_cluster_size must be positive, got {min_cluster_size}.")
    if cluster_space not in {"object", "mano"}:
        raise ValueError(f"cluster_space must be 'object' or 'mano', got {cluster_space!r}.")

    mano_samples = sample_mano_surface(
        mano_vertices,
        mano_faces,
        num_points=num_mano_samples,
        seed=seed,
        use_farthest_point_sampling=use_farthest_point_sampling,
        oversample_factor=oversample_factor,
    )
    object_points_all, object_normals_all, distances_all = closest_points_on_mesh(
        object_vertices,
        object_faces,
        mano_samples.points,
    )
    contact_mask = distances_all <= contact_threshold
    contact_sample_indices = np.nonzero(contact_mask)[0].astype(np.int64)

    if len(contact_sample_indices) == 0:
        return ManoObjectContactResult(
            mano_samples=mano_samples,
            contact_mask=contact_mask,
            contact_sample_indices=contact_sample_indices,
            object_points=np.zeros((0, 3), dtype=np.float64),
            object_normals=np.zeros((0, 3), dtype=np.float64),
            distances=np.zeros((0,), dtype=np.float64),
            cluster_labels=np.zeros((0,), dtype=np.int64),
            clusters=[],
        )

    contact_mano_points = mano_samples.points[contact_sample_indices]
    contact_object_points = object_points_all[contact_sample_indices]
    contact_object_normals = object_normals_all[contact_sample_indices]
    contact_distances = distances_all[contact_sample_indices]
    clustering_points = contact_object_points if cluster_space == "object" else contact_mano_points
    cluster_labels = radius_connected_components(
        clustering_points,
        radius=cluster_radius,
        min_cluster_size=min_cluster_size,
    )

    clusters: list[ContactCluster] = []
    for label in sorted(int(x) for x in np.unique(cluster_labels) if x >= 0):
        local_indices = np.nonzero(cluster_labels == label)[0]
        nearest_index = int(local_indices[np.argmin(contact_distances[local_indices])])
        center = clustering_points[local_indices].mean(axis=0)
        center_index = int(local_indices[np.argmin(np.linalg.norm(clustering_points[local_indices] - center, axis=1))])
        clusters.append(
            ContactCluster(
                label=label,
                sample_indices=contact_sample_indices[local_indices],
                mano_points=contact_mano_points[local_indices],
                object_points=contact_object_points[local_indices],
                object_normals=contact_object_normals[local_indices],
                distances=contact_distances[local_indices],
                nearest_index=int(np.where(local_indices == nearest_index)[0][0]),
                center_index=int(np.where(local_indices == center_index)[0][0]),
            )
        )

    return ManoObjectContactResult(
        mano_samples=mano_samples,
        contact_mask=contact_mask,
        contact_sample_indices=contact_sample_indices,
        object_points=contact_object_points,
        object_normals=contact_object_normals,
        distances=contact_distances,
        cluster_labels=cluster_labels,
        clusters=clusters,
    )


def radius_connected_components(
    points: np.ndarray,
    *,
    radius: float,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Fast spatial clustering by radius graph connected components.

    This is the useful part of DBSCAN for our contact patches: nearby points
    connected by edges of length <= ``radius`` become one cluster. It avoids a
    sklearn dependency and is fast with scipy's cKDTree.
    """

    points = _as_points(points)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int64)
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")

    parent = np.arange(len(points), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j in _radius_pairs(points, radius):
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(len(points))], dtype=np.int64)
    labels = np.full(len(points), -1, dtype=np.int64)
    next_label = 0
    for root in sorted(np.unique(roots)):
        members = np.nonzero(roots == root)[0]
        if len(members) < min_cluster_size:
            continue
        labels[members] = next_label
        next_label += 1
    return labels


def closest_points_on_mesh(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest mesh points, approximate normals, and distances."""

    vertices = _as_points(mesh_vertices)
    faces = np.asarray(mesh_faces, dtype=np.int64)
    query_points = _as_points(query_points)
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        closest, distances, face_indices = trimesh.proximity.closest_point(mesh, query_points)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_indices]
        return closest.astype(np.float64), normals.astype(np.float64), distances.astype(np.float64)
    except Exception:
        indices, distances = _nearest_vertex_indices(vertices, query_points)
        normals = _vertex_normals(vertices, faces)
        return vertices[indices].astype(np.float64), normals[indices].astype(np.float64), distances.astype(np.float64)


def _radius_pairs(points: np.ndarray, radius: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
        return np.asarray(pairs, dtype=np.int64)
    except Exception:
        pairs = []
        radius_sq = radius * radius
        for i in range(len(points)):
            diff = points[i + 1 :] - points[i]
            js = np.nonzero(np.einsum("ij,ij->i", diff, diff) <= radius_sq)[0] + i + 1
            pairs.extend((i, int(j)) for j in js)
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _nearest_vertex_indices(vertices: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(vertices).query(query_points, k=1)
        return indices.astype(np.int64), distances.astype(np.float64)
    except Exception:
        diff = query_points[:, None, :] - vertices[None, :, :]
        dist_sq = np.einsum("qvi,qvi->qv", diff, diff)
        indices = np.argmin(dist_sq, axis=1)
        return indices.astype(np.int64), np.sqrt(dist_sq[np.arange(len(query_points)), indices]).astype(np.float64)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for face, normal in zip(faces, face_normals):
        normals[face] += normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _as_points(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {array.shape}.")
    return array
