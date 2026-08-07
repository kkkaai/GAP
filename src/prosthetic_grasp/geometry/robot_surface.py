from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .surface_sampling import sample_mesh_surface, triangle_double_areas


@dataclass(frozen=True)
class RobotSurfaceTopology:
    local_points: np.ndarray
    local_normals: np.ndarray
    link_names: np.ndarray


@dataclass(frozen=True)
class RobotSurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    local_points: np.ndarray
    local_normals: np.ndarray
    link_names: np.ndarray


@dataclass(frozen=True)
class _GeomSpec:
    mesh_name: str
    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class _BodySpec:
    name: str
    parent: str | None
    pos: np.ndarray
    quat: np.ndarray
    joint_names: list[str]
    geoms: list[_GeomSpec]


@dataclass(frozen=True)
class _JointSpec:
    name: str
    body: str
    axis: np.ndarray
    range: tuple[float, float]


@dataclass(frozen=True)
class _ActuatorSpec:
    name: str
    range: tuple[float, float]
    joint: str | None = None
    tendon: str | None = None


class MjcfRobotSurfaceModel:
    """MJCF hand surface sampler with fixed link-local sample points."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        xml_path: str | Path | None = None,
        wrist_link: str = "base_link",
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = Path.cwd() / self.model_dir
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Robot model directory does not exist: {self.model_dir}")

        self.xml_path = self._resolve_xml_path(xml_path)
        self.wrist_link = wrist_link
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        compiler = self.root.find("compiler")
        meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
        self.mesh_dir = (self.xml_path.parent / meshdir).resolve()

        self.mesh_assets = self._parse_mesh_assets()
        self.bodies: dict[str, _BodySpec] = {}
        self.joints: dict[str, _JointSpec] = {}
        self._parse_bodies()
        self.tendons = self._parse_fixed_tendons()
        self.equalities = self._parse_equality_joints()
        self.actuators = self._parse_actuators()
        self.action_names = [actuator.name for actuator in self.actuators] if self.actuators else list(self.joints)
        self.zero_action = np.zeros(len(self.action_names), dtype=np.float64)
        if self.wrist_link not in self.bodies:
            raise ValueError(f"MJCF model has no wrist/base body {self.wrist_link!r}.")

    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.actuators:
            ranges = {actuator.name: actuator.range for actuator in self.actuators}
        else:
            ranges = {name: joint.range for name, joint in self.joints.items()}
        lower, upper = zip(*(ranges[name] for name in self.action_names))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def sample_surface_topology(
        self,
        *,
        num_points: int = 2000,
        seed: int | None = 7,
        include_links: set[str] | None = None,
        exclude_links: set[str] | None = None,
        use_farthest_point_sampling: bool = True,
        oversample_factor: int = 20,
    ) -> RobotSurfaceTopology:
        """Sample fixed body-local surface points across visual meshes."""

        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}.")
        rng = np.random.default_rng(seed)
        mesh_entries = self._mesh_entries(include_links=include_links, exclude_links=exclude_links)
        if not mesh_entries:
            raise ValueError("No mesh geoms found for the requested links.")

        areas = np.asarray([entry[3] for entry in mesh_entries], dtype=np.float64)
        counts = np.floor(areas / areas.sum() * num_points).astype(np.int64)
        counts[np.argmax(areas)] += num_points - int(counts.sum())

        all_points = []
        all_normals = []
        all_links = []
        for (link_name, vertices, faces, _area), count in zip(mesh_entries, counts):
            if count <= 0:
                continue
            samples = sample_mesh_surface(
                vertices,
                faces,
                num_points=int(count),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                use_farthest_point_sampling=use_farthest_point_sampling,
                oversample_factor=oversample_factor,
            )
            all_points.append(samples.points)
            all_normals.append(samples.normals)
            all_links.extend([link_name] * len(samples.points))

        return RobotSurfaceTopology(
            local_points=np.concatenate(all_points, axis=0).astype(np.float64),
            local_normals=np.concatenate(all_normals, axis=0).astype(np.float64),
            link_names=np.asarray(all_links, dtype=object),
        )

    def materialize_surface(
        self,
        topology: RobotSurfaceTopology,
        action: np.ndarray | None = None,
    ) -> RobotSurfaceSamples:
        """Move fixed link-local samples to the wrist frame for an action."""

        action = self.zero_action if action is None else np.asarray(action, dtype=np.float64)
        transforms = self._body_transforms(self._action_to_joint_values(action))
        world_to_wrist = np.linalg.inv(transforms[self.wrist_link])

        points = np.zeros_like(topology.local_points, dtype=np.float64)
        normals = np.zeros_like(topology.local_normals, dtype=np.float64)
        for link_name in np.unique(topology.link_names):
            mask = topology.link_names == link_name
            transform = world_to_wrist @ transforms[str(link_name)]
            points_h = np.concatenate([topology.local_points[mask], np.ones((int(mask.sum()), 1))], axis=1)
            points[mask] = (transform @ points_h.T).T[:, :3]
            normals[mask] = topology.local_normals[mask] @ transform[:3, :3].T
            normals[mask] /= np.maximum(np.linalg.norm(normals[mask], axis=1, keepdims=True), 1e-12)
        return RobotSurfaceSamples(
            points=points,
            normals=normals,
            local_points=topology.local_points,
            local_normals=topology.local_normals,
            link_names=topology.link_names,
        )

    def link_mesh(self, action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the full visual mesh in wrist frame for visualization."""

        import trimesh

        action = self.zero_action if action is None else np.asarray(action, dtype=np.float64)
        transforms = self._body_transforms(self._action_to_joint_values(action))
        world_to_wrist = np.linalg.inv(transforms[self.wrist_link])
        meshes = []
        for link_name, body in self.bodies.items():
            for geom in body.geoms:
                mesh_path = self.mesh_assets.get(geom.mesh_name)
                if mesh_path is None:
                    continue
                mesh = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                local = _make_transform(_quat_to_matrix(geom.quat), geom.pos)
                vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
                vertices_wrist = (world_to_wrist @ transforms[link_name] @ local @ vertices_h.T).T[:, :3]
                meshes.append(trimesh.Trimesh(vertices=vertices_wrist, faces=faces, process=False))
        if not meshes:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        combined = trimesh.util.concatenate(meshes)
        return np.asarray(combined.vertices, dtype=np.float64), np.asarray(combined.faces, dtype=np.int64)

    def _resolve_xml_path(self, xml_path: str | Path | None) -> Path:
        if xml_path is not None:
            path = Path(xml_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                path = self.model_dir / xml_path
            if not path.exists():
                raise FileNotFoundError(f"MJCF XML does not exist: {xml_path}")
            return path.resolve()

        preferred = self.model_dir / f"{self.model_dir.name}.xml"
        if preferred.exists():
            return preferred.resolve()
        xmls = sorted(self.model_dir.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"No MJCF XML found in {self.model_dir}")
        return xmls[0].resolve()

    def _mesh_entries(
        self,
        *,
        include_links: set[str] | None,
        exclude_links: set[str] | None,
    ) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        import trimesh

        entries = []
        exclude_links = exclude_links or set()
        for link_name, body in self.bodies.items():
            if include_links is not None and link_name not in include_links:
                continue
            if link_name in exclude_links:
                continue
            for geom in body.geoms:
                mesh_path = self.mesh_assets.get(geom.mesh_name)
                if mesh_path is None:
                    continue
                mesh = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                geom_transform = _make_transform(_quat_to_matrix(geom.quat), geom.pos)
                vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
                vertices_local = (geom_transform @ vertices_h.T).T[:, :3]
                area = 0.5 * float(triangle_double_areas(vertices_local, faces).sum())
                if area > 0:
                    entries.append((link_name, vertices_local, faces, area))
        return entries

    def _parse_mesh_assets(self) -> dict[str, Path]:
        assets = {}
        for mesh in self.root.findall("./asset/mesh"):
            name = mesh.get("name")
            filename = mesh.get("file")
            if name and filename:
                assets[name] = self.mesh_dir / filename
        return assets

    def _parse_bodies(self) -> None:
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"MJCF has no worldbody: {self.xml_path}")
        for body in worldbody.findall("body"):
            self._parse_body(body, None)

    def _parse_body(self, elem: ET.Element, parent: str | None) -> None:
        name = elem.get("name")
        if not name:
            return
        joint_names = []
        geoms = []
        for joint in elem.findall("joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            range_value = _parse_vec(joint.get("range"), (0.0, 0.0))
            self.joints[joint_name] = _JointSpec(
                name=joint_name,
                body=name,
                axis=_parse_vec(joint.get("axis"), (0.0, 0.0, 1.0)),
                range=(float(range_value[0]), float(range_value[1])),
            )
            joint_names.append(joint_name)
        for geom in elem.findall("geom"):
            if geom.get("type") == "mesh" and geom.get("mesh"):
                geoms.append(
                    _GeomSpec(
                        mesh_name=str(geom.get("mesh")),
                        pos=_parse_vec(geom.get("pos"), (0.0, 0.0, 0.0)),
                        quat=_parse_vec(geom.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                    )
                )
        self.bodies[name] = _BodySpec(
            name=name,
            parent=parent,
            pos=_parse_vec(elem.get("pos"), (0.0, 0.0, 0.0)),
            quat=_parse_vec(elem.get("quat"), (1.0, 0.0, 0.0, 0.0)),
            joint_names=joint_names,
            geoms=geoms,
        )
        for child in elem.findall("body"):
            self._parse_body(child, name)

    def _parse_fixed_tendons(self) -> dict[str, list[tuple[str, float]]]:
        tendons = {}
        for fixed in self.root.findall("./tendon/fixed"):
            name = fixed.get("name")
            if not name:
                continue
            tendons[name] = [
                (str(joint.get("joint")), float(joint.get("coef", "1.0")))
                for joint in fixed.findall("joint")
                if joint.get("joint")
            ]
        return tendons

    def _parse_equality_joints(self) -> list[tuple[str, str, np.ndarray]]:
        equalities = []
        for joint in self.root.findall("./equality/joint"):
            joint1 = joint.get("joint1")
            joint2 = joint.get("joint2")
            if joint1 and joint2:
                equalities.append((joint1, joint2, _parse_vec(joint.get("polycoef"), (0.0, 1.0, 0.0, 0.0, 0.0))))
        return equalities

    def _parse_actuators(self) -> list[_ActuatorSpec]:
        actuators = []
        for position in self.root.findall("./actuator/position"):
            name = position.get("name")
            if not name:
                continue
            range_value = _parse_vec(position.get("ctrlrange"), (0.0, 0.0))
            actuators.append(
                _ActuatorSpec(
                    name=name,
                    range=(float(range_value[0]), float(range_value[1])),
                    joint=position.get("joint"),
                    tendon=position.get("tendon"),
                )
            )
        return actuators

    def _action_to_joint_values(self, action: np.ndarray) -> dict[str, float]:
        joint_values = {name: 0.0 for name in self.joints}
        if self.actuators:
            actuator_map = {actuator.name: actuator for actuator in self.actuators}
            for name, value in zip(self.action_names, action):
                actuator = actuator_map[name]
                if actuator.joint:
                    joint_values[actuator.joint] = float(value)
                elif actuator.tendon:
                    for joint_name, coef in self.tendons.get(actuator.tendon, []):
                        joint_values[joint_name] = float(value) * coef
        else:
            for name, value in zip(self.action_names, action):
                joint_values[name] = float(value)
        for joint1, joint2, coeffs in self.equalities:
            value = joint_values.get(joint2, 0.0)
            joint_values[joint1] = float(sum(coeff * (value ** power) for power, coeff in enumerate(coeffs)))
        return joint_values

    def _body_transforms(self, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {}
        pending = list(self.bodies)
        while pending:
            progressed = False
            for name in pending[:]:
                body = self.bodies[name]
                if body.parent is not None and body.parent not in transforms:
                    continue
                parent_transform = transforms[body.parent] if body.parent is not None else np.eye(4)
                transform = parent_transform @ _make_transform(_quat_to_matrix(body.quat), body.pos)
                for joint_name in body.joint_names:
                    joint = self.joints[joint_name]
                    transform = transform @ _make_transform(
                        _axis_angle_matrix(joint.axis, joint_values.get(joint_name, 0.0)),
                        np.zeros(3, dtype=np.float64),
                    )
                transforms[name] = transform
                pending.remove(name)
                progressed = True
            if not progressed:
                raise ValueError(f"MJCF has cyclic or invalid body hierarchy: {self.xml_path}")
        return transforms


class UrdfRobotSurfaceModel:
    """URDF hand surface sampler with fixed link-local sample points."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        urdf_path: str | Path | None = None,
        wrist_link: str = "robot0:palm",
    ) -> None:
        from yourdfpy import URDF

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = Path.cwd() / self.model_dir
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Robot model directory does not exist: {self.model_dir}")
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.wrist_link = wrist_link
        self.robot = URDF.load(str(self.urdf_path))
        if self.wrist_link not in self.robot.link_map:
            raise ValueError(f"URDF model has no wrist/base link {self.wrist_link!r}.")
        self.action_names = list(self.robot.actuated_joint_names)
        self.zero_action = np.zeros(len(self.action_names), dtype=np.float64)

    @property
    def xml_path(self) -> Path:
        return self.urdf_path

    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in self.action_names:
            limit = self.robot.joint_map[name].limit
            lo = float(limit.lower) if limit is not None and limit.lower is not None else -np.pi
            hi = float(limit.upper) if limit is not None and limit.upper is not None else np.pi
            if hi <= lo:
                hi = lo + 1e-9
            lower.append(lo)
            upper.append(hi)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def sample_surface_topology(
        self,
        *,
        num_points: int = 2000,
        seed: int | None = 7,
        include_links: set[str] | None = None,
        exclude_links: set[str] | None = None,
        use_farthest_point_sampling: bool = True,
        oversample_factor: int = 20,
    ) -> RobotSurfaceTopology:
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}.")
        rng = np.random.default_rng(seed)
        mesh_entries = self._mesh_entries(include_links=include_links, exclude_links=exclude_links)
        if not mesh_entries:
            raise ValueError("No visual meshes found for the requested links.")
        areas = np.asarray([entry[3] for entry in mesh_entries], dtype=np.float64)
        counts = np.floor(areas / areas.sum() * num_points).astype(np.int64)
        counts[np.argmax(areas)] += num_points - int(counts.sum())

        all_points = []
        all_normals = []
        all_links = []
        for (link_name, vertices, faces, _area), count in zip(mesh_entries, counts):
            if count <= 0:
                continue
            samples = sample_mesh_surface(
                vertices,
                faces,
                num_points=int(count),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                use_farthest_point_sampling=use_farthest_point_sampling,
                oversample_factor=oversample_factor,
            )
            all_points.append(samples.points)
            all_normals.append(samples.normals)
            all_links.extend([link_name] * len(samples.points))

        return RobotSurfaceTopology(
            local_points=np.concatenate(all_points, axis=0).astype(np.float64),
            local_normals=np.concatenate(all_normals, axis=0).astype(np.float64),
            link_names=np.asarray(all_links, dtype=object),
        )

    def materialize_surface(
        self,
        topology: RobotSurfaceTopology,
        action: np.ndarray | None = None,
    ) -> RobotSurfaceSamples:
        action = self.zero_action if action is None else np.asarray(action, dtype=np.float64)
        self.robot.update_cfg(dict(zip(self.action_names, action)))
        world_to_wrist = np.linalg.inv(self.robot.get_transform(self.wrist_link))

        points = np.zeros_like(topology.local_points, dtype=np.float64)
        normals = np.zeros_like(topology.local_normals, dtype=np.float64)
        for link_name in np.unique(topology.link_names):
            mask = topology.link_names == link_name
            transform = world_to_wrist @ self.robot.get_transform(str(link_name))
            points_h = np.concatenate([topology.local_points[mask], np.ones((int(mask.sum()), 1))], axis=1)
            points[mask] = (transform @ points_h.T).T[:, :3]
            normals[mask] = topology.local_normals[mask] @ transform[:3, :3].T
            normals[mask] /= np.maximum(np.linalg.norm(normals[mask], axis=1, keepdims=True), 1e-12)
        return RobotSurfaceSamples(
            points=points,
            normals=normals,
            local_points=topology.local_points,
            local_normals=topology.local_normals,
            link_names=topology.link_names,
        )

    def link_mesh(self, action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        action = self.zero_action if action is None else np.asarray(action, dtype=np.float64)
        self.robot.update_cfg(dict(zip(self.action_names, action)))
        world_to_wrist = np.linalg.inv(self.robot.get_transform(self.wrist_link))
        meshes = []
        for link_name, link in self.robot.link_map.items():
            for visual in link.visuals:
                mesh = visual.geometry.mesh
                if mesh is None:
                    continue
                mesh_path = self._resolve_mesh_path(mesh.filename)
                loaded = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(loaded.vertices, dtype=np.float64)
                if mesh.scale is not None:
                    vertices = vertices * np.asarray(mesh.scale, dtype=np.float64)
                faces = np.asarray(loaded.faces, dtype=np.int64)
                vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
                vertices_wrist = (world_to_wrist @ self.robot.get_transform(link_name) @ visual.origin @ vertices_h.T).T[:, :3]
                meshes.append(trimesh.Trimesh(vertices=vertices_wrist, faces=faces, process=False))
        if not meshes:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        combined = trimesh.util.concatenate(meshes)
        return np.asarray(combined.vertices, dtype=np.float64), np.asarray(combined.faces, dtype=np.int64)

    def _mesh_entries(
        self,
        *,
        include_links: set[str] | None,
        exclude_links: set[str] | None,
    ) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        import trimesh

        exclude_links = exclude_links or set()
        entries = []
        for link_name, link in self.robot.link_map.items():
            if include_links is not None and link_name not in include_links:
                continue
            if link_name in exclude_links:
                continue
            for visual in link.visuals:
                mesh = visual.geometry.mesh
                if mesh is None:
                    continue
                mesh_path = self._resolve_mesh_path(mesh.filename)
                loaded = trimesh.load_mesh(str(mesh_path), process=False)
                vertices = np.asarray(loaded.vertices, dtype=np.float64)
                if mesh.scale is not None:
                    vertices = vertices * np.asarray(mesh.scale, dtype=np.float64)
                faces = np.asarray(loaded.faces, dtype=np.int64)
                vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
                vertices_link = (visual.origin @ vertices_h.T).T[:, :3]
                area = 0.5 * float(triangle_double_areas(vertices_link, faces).sum())
                if area > 0:
                    entries.append((link_name, vertices_link, faces, area))
        return entries

    def _resolve_urdf_path(self, urdf_path: str | Path | None) -> Path:
        if urdf_path is not None:
            path = Path(urdf_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                path = self.model_dir / urdf_path
            if not path.exists():
                raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
            return path.resolve()
        preferred = self.model_dir / "shadowhand.urdf"
        if preferred.exists():
            return preferred.resolve()
        urdfs = sorted(self.model_dir.glob("*.urdf"))
        if not urdfs:
            raise FileNotFoundError(f"No URDF found in {self.model_dir}")
        return urdfs[0].resolve()

    def _resolve_mesh_path(self, filename: str) -> Path:
        path = Path(filename)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                [
                    self.urdf_path.parent / path,
                    self.model_dir / path,
                    self.model_dir / path.name,
                    self.urdf_path.parent / path.name,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve mesh {filename!r} for {self.urdf_path}")


def load_robot_surface_model(
    model_dir: str | Path,
    *,
    model_format: str = "mjcf",
    xml_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    wrist_link: str = "base_link",
) -> MjcfRobotSurfaceModel | UrdfRobotSurfaceModel:
    fmt = model_format.lower()
    if fmt in {"mjcf", "xml"}:
        return MjcfRobotSurfaceModel(model_dir, xml_path=xml_path, wrist_link=wrist_link)
    if fmt == "urdf":
        return UrdfRobotSurfaceModel(model_dir, urdf_path=urdf_path, wrist_link=wrist_link)
    if fmt == "auto":
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_absolute():
            model_dir_path = Path.cwd() / model_dir_path
        if xml_path is not None or list(model_dir_path.glob("*.xml")):
            return MjcfRobotSurfaceModel(model_dir, xml_path=xml_path, wrist_link=wrist_link)
        return UrdfRobotSurfaceModel(model_dir, urdf_path=urdf_path, wrist_link=wrist_link)
    raise ValueError("model_format must be 'mjcf', 'xml', 'urdf', or 'auto'.")


def _parse_vec(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None or value.strip() == "":
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(part) for part in value.split()], dtype=np.float64)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform
