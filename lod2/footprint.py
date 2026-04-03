from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union

from lod2.config import LOD2Config


@dataclass
class FootprintResult:
    polygon: Polygon
    sampled_wall_xy: np.ndarray
    sampled_all_xy: np.ndarray


def _sample_points(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    if points_xy.shape[0] <= max_points:
        return points_xy

    stride = int(np.ceil(points_xy.shape[0] / max_points))
    return points_xy[::stride]


def _alpha_shape(points_xy: np.ndarray, alpha: float) -> Polygon | MultiPolygon:
    if points_xy.shape[0] < 4 or alpha <= 0.0:
        return Polygon(points_xy).convex_hull

    delaunay = Delaunay(points_xy)
    simplices = delaunay.simplices
    triangles = points_xy[simplices]

    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    circum_radius = a * b * c / (4.0 * area + 1e-12)

    keep = circum_radius < (1.0 / alpha)
    if not np.any(keep):
        return Polygon(points_xy).convex_hull

    edge_counts: dict[tuple[int, int], int] = {}
    for simplex in simplices[keep]:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(simplex[start]), int(simplex[end]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return Polygon(points_xy).convex_hull

    linework = MultiLineString([(points_xy[i], points_xy[j]) for i, j in boundary_edges])
    polygons = list(polygonize(linework))
    if not polygons:
        return Polygon(points_xy).convex_hull

    return unary_union(polygons)


def _clean_polygon(
    geometry: Polygon | MultiPolygon, config: LOD2Config
) -> Polygon:
    if isinstance(geometry, MultiPolygon):
        geometry = max(geometry.geoms, key=lambda poly: poly.area)

    polygon = geometry.buffer(0)
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda poly: poly.area)

    if config.footprint_buffer != 0.0:
        polygon = polygon.buffer(config.footprint_buffer).buffer(-config.footprint_buffer)

    if config.footprint_simplify_tolerance > 0.0:
        polygon = polygon.simplify(config.footprint_simplify_tolerance, preserve_topology=True)

    polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon):
        raise ValueError("二维 footprint 提取失败，未能得到有效 Polygon。")

    if polygon.area < config.polygon_area_epsilon:
        raise ValueError("二维 footprint 面积过小，无法构建 LOD2。")

    return orient(polygon, sign=1.0)


def build_footprint(
    wall_points: np.ndarray,
    all_points: np.ndarray,
    config: LOD2Config,
) -> FootprintResult:
    """
    生成建筑底座轮廓。

    经验上，单纯依赖 RANSAC 提取到的“墙面大平面点”会漏掉很多边缘:
    比如檐口、转角、局部凸出体、没有被完整拟合成大平面的立面区域。
    因此这里采用混合策略：
    1. 墙面点 alpha-shape：保留“真正立面”提供的底座约束。
    2. 全部建筑点 alpha-shape：补回墙面 RANSAC 漏掉的边缘包络。
    3. 两者求并集后清洗，得到更贴合点云的 footprint。
    """
    if all_points.shape[0] < 10:
        raise ValueError("输入点太少，无法可靠估计建筑 footprint。")

    sampled_all_points = _sample_points(all_points[:, :2], config.footprint_max_points)
    polygon_all = _alpha_shape(sampled_all_points, config.all_points_alpha_shape_alpha)

    polygon = polygon_all
    sampled_wall_points = np.empty((0, 2), dtype=np.float64)
    if wall_points.shape[0] >= 10 and config.footprint_strategy in {"hybrid", "wall", "union"}:
        sampled_wall_points = _sample_points(wall_points[:, :2], config.footprint_max_points)
        polygon_wall = _alpha_shape(sampled_wall_points, config.wall_alpha_shape_alpha)
        if config.footprint_strategy == "wall":
            polygon = polygon_wall
        else:
            polygon = unary_union([polygon_wall, polygon_all])

    polygon = _clean_polygon(polygon, config)
    return FootprintResult(
        polygon=polygon,
        sampled_wall_xy=sampled_wall_points,
        sampled_all_xy=sampled_all_points,
    )
