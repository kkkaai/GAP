from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ManoSurfaceTopology:
    """Fixed MANO surface anchors in mesh topology coordinates."""

    face_indices: np.ndarray
    barycentric: np.ndarray


@dataclass(frozen=True)
class ManoSurfaceSamples:
    """Area-aware MANO surface samples.

    ``points`` and ``normals`` are in the same coordinate frame as the input
    MANO vertices. ``face_indices`` and ``barycentric`` keep enough information
    to trace each sampled point back to the source triangle.
    """

    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray


def sample_mano_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_points: int = 2000,
    seed: int | None = 7,
    use_farthest_point_sampling: bool = True,
    oversample_factor: int = 20,
) -> ManoSurfaceSamples:
    """Uniformly sample and materialize points from a MANO mesh surface.

    This is a convenience wrapper around ``sample_mano_surface_topology`` and
    ``materialize_mano_surface_samples``. For optimization loops, prefer storing
    the returned ``face_indices`` and ``barycentric`` and re-materializing them
    on each updated MANO mesh.
    """

    topology = sample_mano_surface_topology(
        vertices,
        faces,
        num_points=num_points,
        seed=seed,
        use_farthest_point_sampling=use_farthest_point_sampling,
        oversample_factor=oversample_factor,
    )
    return materialize_mano_surface_samples(vertices, faces, topology)


def sample_mano_surface_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_points: int = 2000,
    seed: int | None = 7,
    use_farthest_point_sampling: bool = True,
    oversample_factor: int = 20,
) -> ManoSurfaceTopology:
    """Sample fixed topology anchors from a MANO mesh surface.

    The first stage samples triangles with probability proportional to surface
    area. When ``use_farthest_point_sampling`` is true, it then follows the
    DexGraspNet-style pattern: generate a dense surface cloud and downsample it
    with farthest point sampling for more even coverage.

    The output is only ``face_indices`` and ``barycentric``. Applying those
    anchors to any later MANO mesh with the same faces makes the sampled points
    move with the MANO deformation instead of being randomly re-sampled.
    """

    vertices = _as_vertices(vertices)
    faces = _as_faces(faces)
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
        return ManoSurfaceTopology(
            face_indices=candidates.face_indices[keep],
            barycentric=candidates.barycentric[keep],
        )
    return ManoSurfaceTopology(
        face_indices=candidates.face_indices,
        barycentric=candidates.barycentric,
    )


def materialize_mano_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology: ManoSurfaceTopology | ManoSurfaceSamples,
) -> ManoSurfaceSamples:
    """Compute current points/normals from fixed MANO topology anchors."""

    vertices = _as_vertices(vertices)
    faces = _as_faces(faces)
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
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    normals_raw = np.cross(edge_1, edge_2)
    normal_norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
    normals = normals_raw / np.maximum(normal_norms, 1e-12)
    return ManoSurfaceSamples(
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
    """Return indices selected by greedy farthest point sampling."""

    points = _as_vertices(points)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    if num_points > len(points):
        raise ValueError(
            f"num_points cannot exceed available points; got {num_points} > {len(points)}."
        )

    rng = rng or np.random.default_rng()
    selected = np.empty(num_points, dtype=np.int64)
    if start_index is None:
        next_index = int(rng.integers(0, len(points)))
    else:
        next_index = int(start_index)
        if next_index < 0 or next_index >= len(points):
            raise ValueError(f"start_index out of range: {start_index}.")

    min_dist_sq = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(num_points):
        selected[i] = next_index
        diff = points - points[next_index]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        next_index = int(np.argmax(min_dist_sq))
    return selected


def write_mano_surface_samples_html(
    output_path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    samples: ManoSurfaceSamples,
    *,
    keypoints: np.ndarray | None = None,
    title: str = "MANO Surface Samples",
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write an interactive Plotly HTML visualization."""

    fig = make_mano_surface_samples_figure(
        vertices=vertices,
        faces=faces,
        samples=samples,
        keypoints=keypoints,
        title=title,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=include_plotlyjs)
    return output_path


def make_mano_surface_samples_figure(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    samples: ManoSurfaceSamples,
    keypoints: np.ndarray | None = None,
    title: str = "MANO Surface Samples",
):
    """Build a Plotly figure for MANO mesh and sampled points."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("MANO surface sample HTML visualization requires plotly.") from exc

    vertices = _as_vertices(vertices)
    faces = _as_faces(faces)
    points = _as_vertices(samples.points)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="MANO mesh",
            color="rgba(40, 180, 80, 0.30)",
            opacity=0.30,
            flatshading=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name=f"surface samples ({len(points)})",
            marker=dict(size=2.2, color="rgb(220, 50, 45)"),
        )
    )
    if keypoints is not None:
        keypoints = _as_vertices(keypoints)
        fig.add_trace(
            go.Scatter3d(
                x=keypoints[:, 0],
                y=keypoints[:, 1],
                z=keypoints[:, 2],
                mode="markers+text",
                name="MANO keypoints",
                text=[str(i) for i in range(len(keypoints))],
                marker=dict(size=4, color="rgba(40,40,40,0.70)"),
                textfont=dict(size=10, color="rgba(40,40,40,0.80)"),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _sample_surface_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> ManoSurfaceSamples:
    tri_vertices = vertices[faces]
    edge_1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge_2 = tri_vertices[:, 2] - tri_vertices[:, 0]
    face_normals_raw = np.cross(edge_1, edge_2)
    double_areas = np.linalg.norm(face_normals_raw, axis=1)
    valid = double_areas > 1e-14
    if not np.any(valid):
        raise ValueError("MANO mesh has no non-degenerate triangles.")

    valid_faces = np.nonzero(valid)[0]
    probabilities = double_areas[valid] / double_areas[valid].sum()
    chosen_valid = rng.choice(len(valid_faces), size=num_points, replace=True, p=probabilities)
    chosen_faces = valid_faces[chosen_valid]
    chosen_triangles = tri_vertices[chosen_faces]

    u = rng.random(num_points)
    v = rng.random(num_points)
    sqrt_u = np.sqrt(u)
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v],
        axis=1,
    )
    points = np.einsum("ni,nij->nj", barycentric, chosen_triangles)
    normals = face_normals_raw[chosen_faces] / double_areas[chosen_faces, None]
    return ManoSurfaceSamples(
        points=points.astype(np.float64),
        normals=normals.astype(np.float64),
        face_indices=chosen_faces.astype(np.int64),
        barycentric=barycentric.astype(np.float64),
    )


def _as_vertices(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {array.shape}.")
    return array


def _as_faces(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected face shape (F, 3), got {array.shape}.")
    return array
