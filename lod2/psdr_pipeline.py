from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from lod2.config import LOD2Config


@dataclass
class PSDRPlaneGroup:
    plane_id: int
    points: np.ndarray
    parameter: np.ndarray


def _load_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location("lod2_psdr_white_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {module_path} 加载 PSDR 白模模块。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_point_groups_path(config: LOD2Config) -> Path:
    psdr_dir = Path(config.psdr_dir)
    point_groups = psdr_dir / "point_groups.npz"
    if not point_groups.exists():
        raise FileNotFoundError(f"未找到 PSDR 平面分组结果: {point_groups}")
    return point_groups


def _load_psdr_groups(point_groups_path: Path) -> list[PSDRPlaneGroup]:
    data = np.load(point_groups_path)
    points = data["points"]
    group_num_points = data["group_num_points"]
    group_points = data["group_points"]
    group_parameters = data["group_parameters"]

    groups: list[PSDRPlaneGroup] = []
    offset = 0
    for plane_id, num_points in enumerate(group_num_points):
        indices = group_points[offset : offset + num_points]
        offset += num_points
        groups.append(
            PSDRPlaneGroup(
                plane_id=int(plane_id),
                points=points[indices],
                parameter=group_parameters[plane_id],
            )
        )
    return groups


def _normalize_plane_parameter(parameter: np.ndarray) -> np.ndarray:
    parameter = np.asarray(parameter, dtype=np.float64).copy()
    norm = np.linalg.norm(parameter[:3])
    if norm < 1e-12:
        raise ValueError("检测到退化 PSDR 平面参数。")
    parameter /= norm
    if parameter[2] < 0.0:
        parameter = -parameter
    return parameter


def _plane_angle_deg(parameter_a: np.ndarray, parameter_b: np.ndarray) -> float:
    cosine = np.clip(np.dot(parameter_a[:3], parameter_b[:3]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _merge_psdr_groups(groups: list[PSDRPlaneGroup], config: LOD2Config) -> list[PSDRPlaneGroup]:
    merged: list[dict[str, object]] = []
    for group in sorted(groups, key=lambda item: item.points.shape[0], reverse=True):
        if group.points.shape[0] < config.psdr_min_points:
            continue

        parameter = _normalize_plane_parameter(group.parameter)
        matched_group = None
        for candidate in merged:
            ref_parameter = candidate["parameter"]
            angle = _plane_angle_deg(parameter, ref_parameter)
            offset = abs(float(parameter[3] - ref_parameter[3]))
            if angle <= config.psdr_merge_angle_deg and offset <= config.psdr_merge_offset:
                matched_group = candidate
                break

        if matched_group is None:
            merged.append(
                {
                    "parameter": parameter,
                    "points": [group.points],
                }
            )
        else:
            matched_group["points"].append(group.points)

    merged_groups: list[PSDRPlaneGroup] = []
    for plane_id, group in enumerate(merged):
        merged_groups.append(
            PSDRPlaneGroup(
                plane_id=plane_id,
                points=np.concatenate(group["points"], axis=0),
                parameter=np.asarray(group["parameter"], dtype=np.float64),
            )
        )
    return merged_groups


def _generate_custom_psdr_mesh(config: LOD2Config, module, point_groups_path: Path) -> trimesh.Trimesh:
    merged_groups = _merge_psdr_groups(_load_psdr_groups(point_groups_path), config)
    if not merged_groups:
        raise ValueError("合并后没有可用的 PSDR 平面组。")

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    roof_count = 0
    wall_count = 0
    for group in merged_groups:
        normal, _, centroid = module.fit_plane(group.points)
        plane_type = module.classify_plane_type(group.points, normal)
        if plane_type == "roof":
            simplify_tolerance = config.psdr_simplify_tolerance * config.psdr_roof_simplify_factor
            roof_count += 1
        else:
            simplify_tolerance = config.psdr_simplify_tolerance * config.psdr_wall_simplify_factor
            wall_count += 1

        projected_points = module.project_points_to_plane(group.points, normal, centroid)
        hull_vertices, _ = module.compute_2d_convex_hull(projected_points, normal)
        if len(hull_vertices) < 3:
            continue

        simplified_vertices = module.simplify_polygon(
            hull_vertices,
            simplify_tolerance,
            preserve_corners=True,
        )
        if len(simplified_vertices) < 3:
            continue

        faces = module.triangulate_convex_polygon(simplified_vertices)
        if len(faces) == 0:
            continue

        all_vertices.append(simplified_vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(simplified_vertices)

    if not all_vertices:
        raise ValueError("自定义 PSDR LOD2 流水线没有生成任何面片。")

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    vertices, faces = module.remove_degenerate_and_duplicates(vertices, faces)
    vertices, faces = module.fill_boundary_with_fan(vertices, faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    if mesh.volume < 0:
        mesh.invert()

    print(
        f"  合并后 PSDR 主平面数: {len(merged_groups)}\n"
        f"  其中屋顶 {roof_count} 个，墙面 {wall_count} 个。"
    )
    return mesh


def run_psdr_white_pipeline(config: LOD2Config) -> Path:
    script_path = Path(config.psdr_white_script)
    if not script_path.exists():
        raise FileNotFoundError(f"未找到 PSDR 白模脚本: {script_path}")

    point_groups_path = _resolve_point_groups_path(config)
    output_path = Path(config.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module = _load_module_from_path(script_path)
    mesh = _generate_custom_psdr_mesh(config, module, point_groups_path)
    mesh.export(output_path)
    mesh = trimesh.load(output_path, force="mesh")
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    if mesh.volume < 0:
        mesh.invert()
    mesh.export(output_path)
    print(
        f"  PSDR-Lod2 OBJ 已导出: {output_path}\n"
        f"  顶点数: {len(mesh.vertices)}, 面片数: {len(mesh.faces)}, watertight: {mesh.is_watertight}"
    )
    return output_path
