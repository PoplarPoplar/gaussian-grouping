from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import LinearRing, LineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, triangulate, unary_union

from lod2.config import LOD2Config
from lod2.footprint import _alpha_shape
from lod2.plane_extraction import PlaneSegment


@dataclass
class MeshBuildResult:
    mesh: trimesh.Trimesh
    base_z: float
    roof_plane_count: int


@dataclass
class RoofFacet:
    plane_id: int
    equation: np.ndarray
    points: np.ndarray


class MeshAccumulator:
    def __init__(self, digits: int) -> None:
        self._digits = digits
        self._vertex_map: dict[tuple[float, float, float], int] = {}
        self.vertices: list[list[float]] = []
        self.faces: list[list[int]] = []

    def _add_vertex(self, vertex: np.ndarray) -> int:
        key = tuple(np.round(vertex.astype(np.float64), self._digits))
        if key not in self._vertex_map:
            self._vertex_map[key] = len(self.vertices)
            self.vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
        return self._vertex_map[key]

    def add_face(self, vertices: np.ndarray, flip: bool = False) -> None:
        face = [self._add_vertex(vertex) for vertex in vertices]
        if flip:
            face = face[::-1]
        self.faces.append(face)

    def add_convex_polygon(self, vertices: np.ndarray, upward: bool) -> None:
        if vertices.shape[0] < 3:
            return

        base = vertices[0]
        for idx in range(1, vertices.shape[0] - 1):
            tri = np.vstack([base, vertices[idx], vertices[idx + 1]])
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal_z = normal[2]
            flip = normal_z < 0.0 if upward else normal_z > 0.0
            self.add_face(tri, flip=flip)

    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
            process=False,
        )
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
        mesh = _seal_boundary_loops(mesh, self._digits)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        return mesh


def _polygon_to_xy(polygon: Polygon) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    return coords


def _triangle_candidates(footprint: Polygon, config: LOD2Config) -> list[Polygon]:
    candidates: list[Polygon] = []
    for triangle in triangulate(footprint):
        clipped = triangle.intersection(footprint)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "Polygon":
            if clipped.area > config.polygon_area_epsilon:
                candidates.append(orient(clipped, sign=1.0))
        elif clipped.geom_type == "MultiPolygon":
            for geom in clipped.geoms:
                if geom.area > config.polygon_area_epsilon:
                    candidates.append(orient(geom, sign=1.0))
    return candidates


def _support_polygon_from_points(points_xy: np.ndarray, epsilon: float) -> Polygon:
    support = MultiPoint(points_xy).convex_hull
    if support.geom_type != "Polygon":
        support = support.buffer(max(epsilon, 1e-3))
    else:
        support = support.buffer(epsilon)
    support = support.buffer(0)
    if support.geom_type != "Polygon":
        raise ValueError("屋顶支持域构建失败。")
    return orient(support, sign=1.0)


def _clean_support_geometry(
    geometry: Polygon | MultiPolygon,
    simplify_tolerance: float,
    area_epsilon: float,
) -> Polygon | MultiPolygon:
    geom = geometry.buffer(0)
    if simplify_tolerance > 0.0:
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)
    geom = geom.buffer(0)
    if geom.is_empty:
        raise ValueError("屋顶支持域清洗后为空。")
    if geom.geom_type == "Polygon":
        return orient(geom, sign=1.0)
    geoms = [orient(g, sign=1.0) for g in geom.geoms if g.area > area_epsilon]
    if not geoms:
        raise ValueError("屋顶支持域面积过小。")
    return MultiPolygon(geoms)


def _plane_angle_deg(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    cosine = np.clip(np.dot(normal_a, normal_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _merge_roof_planes(roof_planes: list[PlaneSegment], config: LOD2Config) -> list[RoofFacet]:
    groups: list[dict[str, object]] = []

    for plane in sorted(roof_planes, key=lambda item: item.points.shape[0], reverse=True):
        matched_group = None
        for group in groups:
            ref_equation = group["equation"]
            angle = _plane_angle_deg(plane.equation[:3], ref_equation[:3])
            offset = abs(float(plane.equation[3] - ref_equation[3]))
            if angle <= config.roof_merge_angle_deg and offset <= config.roof_merge_offset:
                matched_group = group
                break

        if matched_group is None:
            groups.append(
                {
                    "equation": plane.equation.copy(),
                    "points": [plane.points],
                    "count": plane.points.shape[0],
                }
            )
        else:
            matched_group["points"].append(plane.points)
            matched_group["count"] += plane.points.shape[0]

    groups.sort(key=lambda group: int(group["count"]), reverse=True)
    merged: list[RoofFacet] = []
    for group_id, group in enumerate(groups[: config.roof_max_merged_planes]):
        merged.append(
            RoofFacet(
                plane_id=group_id,
                equation=np.asarray(group["equation"], dtype=np.float64),
                points=np.concatenate(group["points"], axis=0),
            )
        )
    return merged


def _build_roof_supports(
    roof_facets: list[RoofFacet],
    footprint: Polygon,
    config: LOD2Config,
) -> dict[int, Polygon | MultiPolygon]:
    supports: dict[int, Polygon | MultiPolygon] = {}
    for facet in roof_facets:
        support = _alpha_shape(facet.points[:, :2], config.roof_support_alpha)
        support = _clean_support_geometry(
            support,
            config.roof_support_simplify_tolerance,
            config.polygon_area_epsilon,
        )
        support = support.intersection(footprint).buffer(0)
        if support.is_empty:
            continue
        supports[facet.plane_id] = support
    return supports


def _plane_height_at_xy(plane: PlaneSegment, xy: np.ndarray) -> np.ndarray:
    a, b, c, d = plane.equation
    if np.abs(c) < 1e-9:
        raise ValueError(f"平面 {plane.plane_id} 的 C 分量过小，无法计算 z(x, y)。")
    return -(a * xy[:, 0] + b * xy[:, 1] + d) / c


def _polygon_area_2d(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _supported_roof_height_upper_envelope(
    roof_planes: list[RoofFacet],
    roof_supports: dict[int, Polygon | MultiPolygon],
    xy: np.ndarray,
) -> np.ndarray:
    heights = np.zeros(xy.shape[0], dtype=np.float64)
    for idx, point_xy in enumerate(xy):
        point = Point(point_xy)
        candidate_planes = [
            plane
            for plane in roof_planes
            if roof_supports[plane.plane_id].contains(point) or roof_supports[plane.plane_id].touches(point)
        ]
        if not candidate_planes:
            candidate_planes = [
                min(roof_planes, key=lambda plane: roof_supports[plane.plane_id].distance(point))
            ]

        values = [_plane_height_at_xy(plane, point_xy[None, :])[0] for plane in candidate_planes]
        heights[idx] = float(np.max(values))
    return heights


def _iter_polygons(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [orient(geometry, sign=1.0)]
    if hasattr(geometry, "geoms"):
        return [orient(geom, sign=1.0) for geom in geometry.geoms if geom.geom_type == "Polygon"]
    return []


def _seal_boundary_loops(mesh: trimesh.Trimesh, digits: int) -> trimesh.Trimesh:
    outline = mesh.outline()
    if outline is None or len(outline.entities) == 0:
        return mesh

    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    vertex_map = {tuple(np.round(vertex, digits)): idx for idx, vertex in enumerate(mesh.vertices)}

    def add_vertex(vertex: np.ndarray) -> int:
        key = tuple(np.round(vertex.astype(np.float64), digits))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
        return vertex_map[key]

    for entity in outline.entities:
        if len(entity.points) < 3:
            continue

        loop = np.asarray(outline.vertices[entity.points], dtype=np.float64)
        if loop.shape[0] < 3:
            continue
        if np.linalg.norm(loop[0] - loop[-1]) < 1e-8:
            loop = loop[:-1]
        if loop.shape[0] < 3:
            continue

        center = loop.mean(axis=0)
        centered = loop - center
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values.shape[0] < 2 or singular_values[1] < 1e-8:
            continue

        axis_u = vh[0]
        axis_v = vh[1]
        coords_2d = np.column_stack([centered @ axis_u, centered @ axis_v])
        polygon_2d = Polygon(coords_2d).buffer(0)
        if polygon_2d.is_empty or polygon_2d.area < 1e-8:
            continue

        triangles = triangulate(polygon_2d)
        for tri in triangles:
            tri_2d = np.asarray(tri.exterior.coords[:-1], dtype=np.float64)
            if tri_2d.shape[0] != 3:
                continue
            tri_indices: list[int] = []
            for point_2d in tri_2d:
                nearest = int(np.argmin(np.linalg.norm(coords_2d - point_2d[None, :], axis=1)))
                tri_indices.append(nearest)
            if len(set(tri_indices)) < 3:
                continue
            tri_vertices = loop[tri_indices]
            face = [add_vertex(vertex) for vertex in tri_vertices]
            faces.append(face)

    sealed = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    sealed.merge_vertices()
    sealed.remove_unreferenced_vertices()
    sealed.remove_duplicate_faces()
    return sealed


def _equal_height_line_segment(
    plane_a: RoofFacet,
    plane_b: RoofFacet,
    overlap_geom: Polygon | MultiPolygon,
) -> list[LineString]:
    """
    计算两个屋顶平面的等高线，并裁切到它们的重叠支持域内。

    只有把 roof_i = roof_j 的真实交线加入 2D 分区，两个屋顶面在 3D 中才会共边。
    否则虽然 2D 分区相邻，但两边抬升后的 z 不一致，会在屋顶上留下缝。
    """
    ai, bi, ci, di = plane_a.equation
    aj, bj, cj, dj = plane_b.equation
    coeff_x = (-ai / ci) + (aj / cj)
    coeff_y = (-bi / ci) + (bj / cj)
    bias = (-di / ci) + (dj / cj)

    normal_xy = np.array([coeff_x, coeff_y], dtype=np.float64)
    norm_xy = np.linalg.norm(normal_xy)
    if norm_xy < 1e-10:
        return []

    min_x, min_y, max_x, max_y = overlap_geom.bounds
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=np.float64)

    if abs(coeff_y) > abs(coeff_x):
        point_on_line = np.array([center[0], -(coeff_x * center[0] + bias) / coeff_y], dtype=np.float64)
    else:
        point_on_line = np.array([-(coeff_y * center[1] + bias) / coeff_x, center[1]], dtype=np.float64)

    direction = np.array([coeff_y, -coeff_x], dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    diag = max(np.hypot(max_x - min_x, max_y - min_y), 1.0)
    p0 = point_on_line - direction * diag * 4.0
    p1 = point_on_line + direction * diag * 4.0
    infinite_segment = LineString([tuple(p0), tuple(p1)])
    clipped = overlap_geom.intersection(infinite_segment)

    if clipped.is_empty:
        return []
    if clipped.geom_type == "LineString":
        return [clipped]
    if hasattr(clipped, "geoms"):
        return [geom for geom in clipped.geoms if geom.geom_type == "LineString" and geom.length > 1e-6]
    return []


def _point_on_segment(point_xy: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, epsilon: float) -> bool:
    seg = seg_end - seg_start
    rel = point_xy - seg_start
    cross = np.abs(seg[0] * rel[1] - seg[1] * rel[0])
    if cross > epsilon * max(1.0, np.linalg.norm(seg)):
        return False
    dot = np.dot(rel, seg)
    if dot < -epsilon:
        return False
    if dot - np.dot(seg, seg) > epsilon:
        return False
    return True


def _collect_boundary_splits(
    ring: LinearRing,
    roof_boundary_xy: np.ndarray,
    epsilon: float,
) -> list[np.ndarray]:
    ring_coords = np.asarray(ring.coords[:-1], dtype=np.float64)
    split_edges: list[np.ndarray] = []
    for start_idx in range(ring_coords.shape[0]):
        start = ring_coords[start_idx]
        end = ring_coords[(start_idx + 1) % ring_coords.shape[0]]
        edge = end - start
        edge_len_sq = float(np.dot(edge, edge))
        edge_points = [start, end]

        for point_xy in roof_boundary_xy:
            if _point_on_segment(point_xy, start, end, epsilon):
                edge_points.append(point_xy)

        unique_points = np.unique(np.round(np.asarray(edge_points), 8), axis=0)
        t_values = (
            np.dot(unique_points - start[None, :], edge[None, :].T).ravel()
            / max(edge_len_sq, 1e-12)
        )
        order = np.argsort(t_values)
        split_edges.append(unique_points[order])

    return split_edges


def _add_bottom_faces(
    accumulator: MeshAccumulator,
    footprint: Polygon,
    base_z: float,
    config: LOD2Config,
) -> None:
    for triangle in _triangle_candidates(footprint, config):
        xy = _polygon_to_xy(triangle)
        tri_vertices = np.column_stack([xy, np.full(xy.shape[0], base_z, dtype=np.float64)])
        accumulator.add_convex_polygon(tri_vertices, upward=False)


def _add_roof_faces(
    accumulator: MeshAccumulator,
    footprint: Polygon,
    roof_planes: list[RoofFacet],
    roof_supports: dict[int, Polygon | MultiPolygon],
    config: LOD2Config,
) -> dict[tuple[float, float], float]:
    roof_boundary_vertices: dict[tuple[float, float], float] = {}
    linework = [LineString(footprint.exterior.coords)]
    for interior in footprint.interiors:
        linework.append(LineString(interior.coords))

    if config.roof_partition_use_support_boundaries:
        for facet in roof_planes:
            if facet.plane_id not in roof_supports:
                continue
            if abs(float(facet.equation[2])) < config.roof_partition_support_abs_z_min:
                continue
            support_geom = roof_supports[facet.plane_id]
            for poly in _iter_polygons(support_geom):
                linework.append(LineString(poly.exterior.coords))
                for interior in poly.interiors:
                    linework.append(LineString(interior.coords))

    for idx, plane_a in enumerate(roof_planes):
        if plane_a.plane_id not in roof_supports:
            continue
        for plane_b in roof_planes[idx + 1 :]:
            if plane_b.plane_id not in roof_supports:
                continue
            overlap = roof_supports[plane_a.plane_id].intersection(roof_supports[plane_b.plane_id]).intersection(footprint)
            if overlap.is_empty:
                continue
            for overlap_poly in _iter_polygons(overlap):
                if overlap_poly.area < config.polygon_area_epsilon:
                    continue
                linework.extend(_equal_height_line_segment(plane_a, plane_b, overlap_poly))

    partition = polygonize(unary_union(linework))
    for cell in partition:
        clipped = cell.intersection(footprint).buffer(0)
        for cell_poly in _iter_polygons(clipped):
            if cell_poly.area < config.polygon_area_epsilon:
                continue

            rep_point = cell_poly.representative_point()
            active_candidates = [
                plane
                for plane in roof_planes
                if plane.plane_id in roof_supports
                and (
                    roof_supports[plane.plane_id].contains(rep_point)
                    or roof_supports[plane.plane_id].touches(rep_point)
                )
            ]
            if not active_candidates:
                active_candidates = [
                    min(
                        roof_planes,
                        key=lambda plane: roof_supports[plane.plane_id].distance(rep_point)
                        if plane.plane_id in roof_supports
                        else float("inf"),
                    )
                ]

            active_plane = max(
                active_candidates,
                key=lambda plane: _plane_height_at_xy(
                    plane, np.asarray([[rep_point.x, rep_point.y]], dtype=np.float64)
                )[0],
            )

            for roof_triangle in _triangle_candidates(cell_poly, config):
                tri_xy = _polygon_to_xy(roof_triangle)
                tri_z = _plane_height_at_xy(active_plane, tri_xy)
                boundary_mask = np.array(
                    [footprint.boundary.distance(Point(point_xy)) <= config.boundary_snap_epsilon for point_xy in tri_xy],
                    dtype=bool,
                )
                if np.any(boundary_mask):
                    tri_z[boundary_mask] = _supported_roof_height_upper_envelope(
                        roof_planes,
                        roof_supports,
                        tri_xy[boundary_mask],
                    )
                roof_vertices = np.column_stack([tri_xy, tri_z])
                accumulator.add_convex_polygon(roof_vertices, upward=True)

                for point_xy, point_z in zip(tri_xy, tri_z):
                    if footprint.boundary.distance(Point(point_xy)) <= config.boundary_snap_epsilon:
                        key = tuple(np.round(point_xy, 8))
                        roof_boundary_vertices[key] = max(
                            roof_boundary_vertices.get(key, -np.inf), float(point_z)
                        )

    return roof_boundary_vertices


def _add_wall_faces(
    accumulator: MeshAccumulator,
    footprint: Polygon,
    roof_planes: list[RoofFacet],
    roof_supports: dict[int, Polygon | MultiPolygon],
    base_z: float,
    roof_boundary_xyz: dict[tuple[float, float], float],
    config: LOD2Config,
) -> None:
    polygon = orient(footprint, sign=1.0)
    rings = [polygon.exterior, *polygon.interiors]
    roof_boundary_xy = (
        np.asarray(list(roof_boundary_xyz.keys()), dtype=np.float64)
        if roof_boundary_xyz
        else np.empty((0, 2), dtype=np.float64)
    )

    for ring in rings:
        split_edges = _collect_boundary_splits(ring, roof_boundary_xy, config.boundary_snap_epsilon)
        ring_coords = np.asarray(ring.coords[:-1], dtype=np.float64)
        ring_ccw = LinearRing(ring_coords).is_ccw

        for edge_points in split_edges:
            for idx in range(edge_points.shape[0] - 1):
                start_xy = edge_points[idx]
                end_xy = edge_points[idx + 1]
                if np.linalg.norm(end_xy - start_xy) < 1e-9:
                    continue

                start_key = tuple(np.round(start_xy, 8))
                end_key = tuple(np.round(end_xy, 8))
                top_z = np.array(
                    [
                        roof_boundary_xyz.get(
                            start_key,
                            _supported_roof_height_upper_envelope(
                                roof_planes, roof_supports, start_xy[None, :]
                            )[0],
                        ),
                        roof_boundary_xyz.get(
                            end_key,
                            _supported_roof_height_upper_envelope(
                                roof_planes, roof_supports, end_xy[None, :]
                            )[0],
                        ),
                    ],
                    dtype=np.float64,
                )
                quad = np.array(
                    [
                        [start_xy[0], start_xy[1], base_z],
                        [end_xy[0], end_xy[1], base_z],
                        [end_xy[0], end_xy[1], top_z[1]],
                        [start_xy[0], start_xy[1], top_z[0]],
                    ],
                    dtype=np.float64,
                )

                edge_dir = end_xy - start_xy
                outward_xy = (
                    np.array([edge_dir[1], -edge_dir[0]], dtype=np.float64)
                    if ring_ccw
                    else np.array([-edge_dir[1], edge_dir[0]], dtype=np.float64)
                )

                tri = quad[[0, 1, 2]]
                tri_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                flip = np.dot(tri_normal[:2], outward_xy) < 0.0
                if flip:
                    quad = quad[::-1]
                accumulator.add_face(quad[[0, 1, 2]], flip=False)
                accumulator.add_face(quad[[0, 2, 3]], flip=False)


def build_lod2_mesh(
    footprint: Polygon,
    roof_planes: list[PlaneSegment],
    all_points: np.ndarray,
    config: LOD2Config,
) -> MeshBuildResult:
    if not roof_planes:
        raise ValueError("没有可用的屋顶平面，无法生成 LOD2 封闭网格。")

    merged_roofs = _merge_roof_planes(roof_planes, config)
    roof_supports = _build_roof_supports(merged_roofs, footprint, config)
    active_roofs = [facet for facet in merged_roofs if facet.plane_id in roof_supports]
    if not active_roofs:
        raise ValueError("合并后的屋顶主平面为空，无法生成 LOD2。")

    base_z = float(np.percentile(all_points[:, 2], config.base_z_percentile))
    roof_probe_xy = np.asarray(footprint.exterior.coords[:-1], dtype=np.float64)
    roof_probe_z = _supported_roof_height_upper_envelope(active_roofs, roof_supports, roof_probe_xy)
    if float(np.min(roof_probe_z)) - base_z < config.min_building_height:
        base_z = float(np.min(roof_probe_z) - config.min_building_height)

    accumulator = MeshAccumulator(config.vertex_merge_digits)
    roof_boundary_xyz = _add_roof_faces(accumulator, footprint, active_roofs, roof_supports, config)
    _add_bottom_faces(accumulator, footprint, base_z, config)
    _add_wall_faces(accumulator, footprint, active_roofs, roof_supports, base_z, roof_boundary_xyz, config)

    mesh = accumulator.to_trimesh()
    return MeshBuildResult(mesh=mesh, base_z=base_z, roof_plane_count=len(active_roofs))
