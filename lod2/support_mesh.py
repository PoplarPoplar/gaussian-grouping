from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from lod2.config import LOD2Config
from lod2.io_utils import GaussianPointCloud


@dataclass
class SupportMeshResult:
    mesh_path: Path
    vertex_count: int
    triangle_count: int


def orient_gaussian_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    用建筑整体质心做一次全局定向，让法线尽量朝向建筑外侧。

    说明：
    1. 3DGS 推导得到的是“轴方向”，方向符号本身可能正反都对。
    2. 泊松重建要求法线符号整体一致，否则会在屋顶/墙面附近生成错误的封闭体。
    3. 这里用 `dot(point - centroid, normal)` 判断当前法线是否大致朝外。
       对单栋建筑这种单连通外壳，这个启发式通常足够稳定。
    """
    centroid = points.mean(axis=0, keepdims=True)
    radial = points - centroid
    alignment = np.einsum("ij,ij->i", radial, normals)
    oriented = normals.copy()
    oriented[alignment < 0.0] *= -1.0
    oriented /= np.linalg.norm(oriented, axis=1, keepdims=True) + 1e-12
    return oriented


def _make_open3d_point_cloud(gaussians: GaussianPointCloud, config: LOD2Config) -> o3d.geometry.PointCloud:
    points = gaussians.points
    normals = orient_gaussian_normals_outward(points, gaussians.normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if config.support_voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(config.support_voxel_size)

    if len(pcd.points) >= max(100, config.support_normal_k):
        pcd.orient_normals_consistent_tangent_plane(config.support_normal_k)

        # Open3D 的一致化只保证局部相对一致，不保证“整体向外”。
        # 因此这里再做一次基于质心的全局翻转，避免最终法线整体朝内。
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        normals = orient_gaussian_normals_outward(points, normals)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def _expand_aabb(
    bbox: o3d.geometry.AxisAlignedBoundingBox,
    padding_ratio: float,
) -> o3d.geometry.AxisAlignedBoundingBox:
    if padding_ratio <= 0.0:
        return bbox

    extent = bbox.get_extent()
    padding = np.maximum(extent * padding_ratio, 1e-3)
    return o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bbox.min_bound - padding,
        max_bound=bbox.max_bound + padding,
    )


def build_poisson_support_mesh(
    gaussians: GaussianPointCloud,
    config: LOD2Config,
    output_path: Path,
) -> SupportMeshResult:
    """
    生成给 COMPOD/COMPOSE 用的支撑网格。

    这里的泊松网格不是最终 LOD2 输出，而是作为 mesh-labeling 的几何参考：
    - PSDR/COMPOD 负责“平面 + 封闭拓扑”
    - 泊松网格负责告诉 partition 哪些 cell 在建筑内部
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = _make_open3d_point_cloud(gaussians, config)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=config.support_poisson_depth,
        scale=config.support_poisson_scale,
        linear_fit=True,
    )

    densities = np.asarray(densities, dtype=np.float64)
    if densities.size > 0 and 0.0 < config.support_density_quantile < 1.0:
        density_threshold = float(np.quantile(densities, config.support_density_quantile))
        remove_mask = densities < density_threshold
        if np.any(remove_mask):
            mesh.remove_vertices_by_mask(remove_mask)

    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()

    crop_box = _expand_aabb(
        pcd.get_axis_aligned_bounding_box(),
        config.support_crop_padding_ratio,
    )
    mesh = mesh.crop(crop_box)

    if config.support_target_triangles > 0 and len(mesh.triangles) > config.support_target_triangles:
        mesh = mesh.simplify_quadric_decimation(config.support_target_triangles)

    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("泊松支撑网格为空，无法继续进行 COMPOD mesh-labeling。")

    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise IOError(f"无法写出泊松支撑网格: {output_path}")

    return SupportMeshResult(
        mesh_path=output_path,
        vertex_count=len(mesh.vertices),
        triangle_count=len(mesh.triangles),
    )
