from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from lod2.config import LOD2Config
from lod2.io_utils import GaussianPointCloud


@dataclass
class PlaneSegment:
    plane_id: int
    equation: np.ndarray
    indices: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    plane_type: str

    @property
    def normal(self) -> np.ndarray:
        return self.equation[:3]


@dataclass
class PlaneExtractionResult:
    planes: list[PlaneSegment]
    remaining_indices: np.ndarray
    remaining_points: np.ndarray


def _normalize_plane(equation: np.ndarray) -> np.ndarray:
    equation = np.asarray(equation, dtype=np.float64)
    normal_norm = np.linalg.norm(equation[:3])
    if normal_norm < 1e-12:
        raise ValueError("检测到退化平面。")
    return equation / normal_norm


def _orient_plane(equation: np.ndarray, support_normals: np.ndarray) -> np.ndarray:
    """
    使用点法线平均方向来固定平面法向的正负号。

    Open3D 的 RANSAC 只给出一个数学上等价的平面，正负号是任意的。
    这里利用 Step 1 推导出来的点法线做参考，让平面法向尽量与局部法线一致，
    这样后续墙/顶分类和高度函数都更稳定。
    """
    eq = _normalize_plane(equation)
    mean_normal = support_normals.mean(axis=0)
    if np.linalg.norm(mean_normal) > 1e-9:
        mean_normal /= np.linalg.norm(mean_normal)
        if np.dot(eq[:3], mean_normal) < 0.0:
            eq = -eq
    return eq


def extract_planes(gaussians: GaussianPointCloud, config: LOD2Config) -> PlaneExtractionResult:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussians.points)
    pcd.normals = o3d.utility.Vector3dVector(gaussians.normals)

    remaining_indices = np.arange(gaussians.points.shape[0], dtype=np.int64)
    remaining_pcd = pcd
    planes: list[PlaneSegment] = []

    min_remaining_points = int(np.ceil(config.plane_remaining_ratio * gaussians.points.shape[0]))

    while len(remaining_indices) >= min_remaining_points and len(planes) < config.plane_max_count:
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=config.plane_distance_threshold,
            ransac_n=config.plane_ransac_n,
            num_iterations=config.plane_iterations,
        )

        inliers = np.asarray(inliers, dtype=np.int64)
        if inliers.size < config.plane_min_points:
            break

        original_indices = remaining_indices[inliers]
        support_points = gaussians.points[original_indices]
        support_normals = gaussians.normals[original_indices]
        equation = _orient_plane(np.asarray(plane_model, dtype=np.float64), support_normals)

        if np.abs(equation[2]) >= config.wall_z_threshold and equation[2] < 0.0:
            equation = -equation

        plane_type = "wall" if np.abs(equation[2]) < config.wall_z_threshold else "roof"

        planes.append(
            PlaneSegment(
                plane_id=len(planes),
                equation=equation,
                indices=original_indices,
                points=support_points,
                normals=support_normals,
                plane_type=plane_type,
            )
        )

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        keep_mask = np.ones(remaining_indices.shape[0], dtype=bool)
        keep_mask[inliers] = False
        remaining_indices = remaining_indices[keep_mask]

    return PlaneExtractionResult(
        planes=planes,
        remaining_indices=remaining_indices,
        remaining_points=gaussians.points[remaining_indices],
    )

