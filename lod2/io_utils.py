from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData


@dataclass
class GaussianPointCloud:
    points: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    field_names: tuple[str, ...]


def _require_fields(field_names: tuple[str, ...], required: list[str]) -> None:
    missing = [name for name in required if name not in field_names]
    if missing:
        raise KeyError(f"PLY 缺少必要字段: {missing}")


def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    将 3DGS 的四元数批量转成旋转矩阵。

    仓库中的 `utils.general_utils.build_rotation` 已经说明了字段顺序是
    (r, x, y, z)，也就是标量项在前。因此这里严格复现同一套公式，
    这样能与训练/导出的高斯朝向保持一致。
    """
    q = quaternions.astype(np.float64, copy=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - r * z)
    rot[:, 0, 2] = 2.0 * (x * z + r * y)
    rot[:, 1, 0] = 2.0 * (x * y + r * z)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - r * x)
    rot[:, 2, 0] = 2.0 * (x * z - r * y)
    rot[:, 2, 1] = 2.0 * (y * z + r * x)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def infer_normals_from_gaussians(scales_raw: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """
    由 3DGS 的尺度和旋转推导表面法线。

    数学含义：
    1. `scale_0~2` 在 3DGS 里是对数尺度，需要先做 `np.exp` 激活，得到真实轴长。
    2. 对一个局部表面贴片而言，最短轴通常垂直于表面，因此它就是局部法线方向。
    3. 四元数定义了局部坐标系相对世界坐标系的旋转。旋转矩阵的第 k 列，
       就是“局部第 k 个轴”在世界坐标系中的方向。
    4. 因此：找到最短尺度轴的索引 k，再取旋转矩阵第 k 列，即可得到高精度法线。
    """
    scales = np.exp(scales_raw.astype(np.float64))
    rot_mats = quaternion_to_rotation_matrix(rotations)

    shortest_axis = np.argmin(scales, axis=1)
    point_indices = np.arange(rot_mats.shape[0])
    normals = rot_mats[point_indices, :, shortest_axis]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals


def _extract_colors(vertex_data) -> np.ndarray:
    names = vertex_data.dtype.names
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.column_stack(
            [vertex_data["red"], vertex_data["green"], vertex_data["blue"]]
        ).astype(np.float64)
        if colors.max() > 1.0:
            colors /= 255.0
        return np.clip(colors, 0.0, 1.0)

    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        colors = np.column_stack(
            [vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"]]
        ).astype(np.float64)
        return np.clip(colors + 0.5, 0.0, 1.0)

    return np.full((len(vertex_data), 3), 0.7, dtype=np.float64)


def load_gaussian_ply(ply_path: str) -> GaussianPointCloud:
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data
    field_names = vertex.dtype.names

    _require_fields(
        field_names,
        [
            "x",
            "y",
            "z",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ],
    )

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    scales = np.column_stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]).astype(
        np.float64
    )
    rotations = np.column_stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]
    ).astype(np.float64)
    normals = infer_normals_from_gaussians(scales, rotations)
    colors = _extract_colors(vertex)

    return GaussianPointCloud(
        points=points,
        normals=normals,
        colors=colors,
        scales=np.exp(scales),
        rotations=rotations,
        field_names=field_names,
    )

