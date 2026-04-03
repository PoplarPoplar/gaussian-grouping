from __future__ import annotations

from pathlib import Path

import numpy as np

from lod2.config import LOD2Config
from lod2.footprint import build_footprint
from lod2.io_utils import load_gaussian_ply
from lod2.mesh_builder import build_lod2_mesh
from lod2.plane_extraction import PlaneSegment
from lod2.psdr_pipeline import _load_psdr_groups, _normalize_plane_parameter, _resolve_point_groups_path


def _groups_to_plane_segments(
    point_groups_path: Path,
    gaussians_normals: np.ndarray,
    config: LOD2Config,
) -> list[PlaneSegment]:
    groups = _load_psdr_groups(point_groups_path)
    planes: list[PlaneSegment] = []

    for plane_id, group in enumerate(groups):
        if group.points.shape[0] < config.psdr_envelope_min_points:
            continue

        equation = _normalize_plane_parameter(group.parameter)
        plane_type = "wall" if abs(float(equation[2])) < config.wall_z_threshold else "roof"

        # 这里不强行重新拟合平面，而是优先保留 PSDR 给出的 group parameter，
        # 因为它在 plane arrangement 语义上更稳定。
        planes.append(
            PlaneSegment(
                plane_id=len(planes),
                equation=equation,
                indices=np.empty((0,), dtype=np.int64),
                points=group.points.astype(np.float64),
                normals=np.zeros((group.points.shape[0], 3), dtype=np.float64)
                if gaussians_normals.size == 0
                else np.zeros((group.points.shape[0], 3), dtype=np.float64),
                plane_type=plane_type,
            )
        )
    return planes


def run_psdr_envelope_pipeline(config: LOD2Config) -> Path:
    print(f"[Envelope] 使用 PSDR 平面 + 单体 footprint + 精确屋顶包络: {config.psdr_dir}")
    gaussians = load_gaussian_ply(config.input_ply)
    point_groups_path = _resolve_point_groups_path(config)

    all_planes = _groups_to_plane_segments(point_groups_path, gaussians.normals, config)
    if not all_planes:
        raise ValueError("PSDR 平面分组为空，无法构建 envelope LOD2。")

    wall_planes = [plane for plane in all_planes if plane.plane_type == "wall"]
    roof_planes = [plane for plane in all_planes if plane.plane_type == "roof"]
    if not roof_planes:
        raise ValueError("没有可用屋顶平面。")

    base_limit = float(np.percentile(gaussians.points[:, 2], config.psdr_envelope_base_percentile))
    base_points = gaussians.points[gaussians.points[:, 2] <= base_limit]
    roof_median_z_threshold = float(
        np.percentile(gaussians.points[:, 2], config.psdr_envelope_roof_min_z_percentile)
    )
    wall_points = (
        np.concatenate([plane.points for plane in wall_planes], axis=0)
        if wall_planes
        else np.empty((0, 3), dtype=np.float64)
    )

    print(
        f"  PSDR 平面数: {len(all_planes)}，墙面 {len(wall_planes)}，屋顶 {len(roof_planes)}\n"
        f"  footprint 使用低处点阈值 z <= {base_limit:.3f}\n"
        f"  屋顶面最低中位高度阈值 z >= {roof_median_z_threshold:.3f}"
    )

    footprint = build_footprint(wall_points, base_points if len(base_points) >= 50 else gaussians.points, config)
    roof_probe_xy = np.asarray(footprint.polygon.exterior.coords[:-1], dtype=np.float64)

    # 过滤明显不合理的屋顶面：过低或近似垂直的“伪屋顶”不参与顶部包络。
    valid_roofs: list[PlaneSegment] = []
    for plane in roof_planes:
        if abs(float(plane.equation[2])) < config.psdr_envelope_roof_min_abs_z:
            continue
        if float(np.median(plane.points[:, 2])) < roof_median_z_threshold:
            continue
        a, b, c, d = plane.equation
        z_vals = -(a * roof_probe_xy[:, 0] + b * roof_probe_xy[:, 1] + d) / c
        if float(np.nanmax(z_vals)) - float(np.percentile(gaussians.points[:, 2], config.base_z_percentile)) < config.psdr_envelope_roof_min_height:
            continue
        valid_roofs.append(plane)

    if not valid_roofs:
        raise ValueError("过滤后没有合理的屋顶平面。")

    mesh_result = build_lod2_mesh(footprint.polygon, valid_roofs, gaussians.points, config)
    output_path = Path(config.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_result.mesh.export(output_path)
    print(
        f"  Envelope OBJ 已导出: {output_path}\n"
        f"  顶点数: {len(mesh_result.mesh.vertices)}, 面片数: {len(mesh_result.mesh.faces)}, "
        f"watertight: {mesh_result.mesh.is_watertight}"
    )
    return output_path
