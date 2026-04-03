from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from lod2.config import LOD2Config
from lod2.footprint import build_footprint
from lod2.hybrid_compod_pipeline import run_hybrid_compod_pipeline
from lod2.io_utils import load_gaussian_ply
from lod2.mesh_builder import build_lod2_mesh
from lod2.plane_extraction import extract_planes
from lod2.psdr_envelope_pipeline import run_psdr_envelope_pipeline
from lod2.psdr_pipeline import run_psdr_white_pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3DGS 单体建筑 -> LOD2 watertight OBJ")
    parser.add_argument("--mode", type=str, default=LOD2Config.reconstruction_mode)
    parser.add_argument("--input-ply", type=str, default=LOD2Config.input_ply)
    parser.add_argument("--output-obj", type=str, default=LOD2Config.output_obj)
    parser.add_argument("--psdr-dir", type=str, default=LOD2Config.psdr_dir)
    parser.add_argument("--psdr-white-script", type=str, default=LOD2Config.psdr_white_script)
    parser.add_argument("--psdr-min-points", type=int, default=LOD2Config.psdr_min_points)
    parser.add_argument(
        "--psdr-simplify-tolerance",
        type=float,
        default=LOD2Config.psdr_simplify_tolerance,
    )
    parser.add_argument("--psdr-merge-angle-deg", type=float, default=LOD2Config.psdr_merge_angle_deg)
    parser.add_argument("--psdr-merge-offset", type=float, default=LOD2Config.psdr_merge_offset)
    parser.add_argument(
        "--psdr-roof-simplify-factor",
        type=float,
        default=LOD2Config.psdr_roof_simplify_factor,
    )
    parser.add_argument(
        "--psdr-wall-simplify-factor",
        type=float,
        default=LOD2Config.psdr_wall_simplify_factor,
    )
    parser.add_argument("--psdr-envelope-min-points", type=int, default=LOD2Config.psdr_envelope_min_points)
    parser.add_argument(
        "--psdr-envelope-base-percentile",
        type=float,
        default=LOD2Config.psdr_envelope_base_percentile,
    )
    parser.add_argument(
        "--psdr-envelope-roof-min-abs-z",
        type=float,
        default=LOD2Config.psdr_envelope_roof_min_abs_z,
    )
    parser.add_argument(
        "--psdr-envelope-roof-min-height",
        type=float,
        default=LOD2Config.psdr_envelope_roof_min_height,
    )
    parser.add_argument(
        "--psdr-envelope-roof-min-z-percentile",
        type=float,
        default=LOD2Config.psdr_envelope_roof_min_z_percentile,
    )

    parser.add_argument("--plane-distance-threshold", type=float, default=LOD2Config.plane_distance_threshold)
    parser.add_argument("--plane-ransac-n", type=int, default=LOD2Config.plane_ransac_n)
    parser.add_argument("--plane-iterations", type=int, default=LOD2Config.plane_iterations)
    parser.add_argument("--plane-min-points", type=int, default=LOD2Config.plane_min_points)
    parser.add_argument("--plane-max-count", type=int, default=LOD2Config.plane_max_count)
    parser.add_argument("--plane-remaining-ratio", type=float, default=LOD2Config.plane_remaining_ratio)
    parser.add_argument("--wall-z-threshold", type=float, default=LOD2Config.wall_z_threshold)

    parser.add_argument("--roof-min-points", type=int, default=LOD2Config.roof_min_points)
    parser.add_argument("--roof-min-abs-z", type=float, default=LOD2Config.roof_min_abs_z)
    parser.add_argument("--roof-grid-resolution", type=float, default=LOD2Config.roof_grid_resolution)
    parser.add_argument("--roof-merge-angle-deg", type=float, default=LOD2Config.roof_merge_angle_deg)
    parser.add_argument("--roof-merge-offset", type=float, default=LOD2Config.roof_merge_offset)
    parser.add_argument("--roof-max-merged-planes", type=int, default=LOD2Config.roof_max_merged_planes)
    parser.add_argument("--roof-support-alpha", type=float, default=LOD2Config.roof_support_alpha)
    parser.add_argument(
        "--roof-support-simplify-tolerance",
        type=float,
        default=LOD2Config.roof_support_simplify_tolerance,
    )
    parser.add_argument(
        "--roof-partition-use-support-boundaries",
        action="store_true",
        default=LOD2Config.roof_partition_use_support_boundaries,
    )
    parser.add_argument(
        "--roof-partition-no-support-boundaries",
        action="store_false",
        dest="roof_partition_use_support_boundaries",
    )
    parser.add_argument(
        "--roof-partition-support-abs-z-min",
        type=float,
        default=LOD2Config.roof_partition_support_abs_z_min,
    )

    parser.add_argument("--wall-alpha-shape-alpha", type=float, default=LOD2Config.wall_alpha_shape_alpha)
    parser.add_argument(
        "--all-points-alpha-shape-alpha",
        type=float,
        default=LOD2Config.all_points_alpha_shape_alpha,
    )
    parser.add_argument("--footprint-strategy", type=str, default=LOD2Config.footprint_strategy)
    parser.add_argument("--footprint-max-points", type=int, default=LOD2Config.footprint_max_points)
    parser.add_argument(
        "--footprint-simplify-tolerance",
        type=float,
        default=LOD2Config.footprint_simplify_tolerance,
    )
    parser.add_argument("--footprint-buffer", type=float, default=LOD2Config.footprint_buffer)
    parser.add_argument("--base-z-percentile", type=float, default=LOD2Config.base_z_percentile)
    parser.add_argument("--min-building-height", type=float, default=LOD2Config.min_building_height)

    parser.add_argument("--compod-env-name", type=str, default=LOD2Config.compod_env_name)
    parser.add_argument("--compod-device", type=str, default=LOD2Config.compod_device)
    parser.add_argument("--compod-home-dir", type=str, default=LOD2Config.compod_home_dir)
    parser.add_argument("--compod-padding", type=float, default=LOD2Config.compod_padding)
    parser.add_argument("--compod-area-weight", type=float, default=LOD2Config.compod_area_weight)
    parser.add_argument("--compod-cc-weight", type=float, default=LOD2Config.compod_cc_weight)
    parser.add_argument(
        "--compod-min-component-area-ratio",
        type=float,
        default=LOD2Config.compod_min_component_area_ratio,
    )
    parser.add_argument("--compod-verbosity", type=int, default=LOD2Config.compod_verbosity)
    parser.add_argument("--compod-exact", action="store_true", default=LOD2Config.compod_exact)
    parser.add_argument("--compod-simplify-tree", action="store_true", default=LOD2Config.compod_simplify_tree)
    parser.add_argument("--compod-no-simplify-tree", action="store_false", dest="compod_simplify_tree")
    parser.add_argument("--compod-surface-mode", type=str, default=LOD2Config.compod_surface_mode)

    parser.add_argument("--support-mesh-dir", type=str, default=LOD2Config.support_mesh_dir)
    parser.add_argument("--support-mesh-name", type=str, default=LOD2Config.support_mesh_name)
    parser.add_argument("--support-poisson-depth", type=int, default=LOD2Config.support_poisson_depth)
    parser.add_argument("--support-poisson-scale", type=float, default=LOD2Config.support_poisson_scale)
    parser.add_argument(
        "--support-density-quantile",
        type=float,
        default=LOD2Config.support_density_quantile,
    )
    parser.add_argument(
        "--support-target-triangles",
        type=int,
        default=LOD2Config.support_target_triangles,
    )
    parser.add_argument(
        "--support-crop-padding-ratio",
        type=float,
        default=LOD2Config.support_crop_padding_ratio,
    )
    parser.add_argument("--support-normal-k", type=int, default=LOD2Config.support_normal_k)
    parser.add_argument("--support-voxel-size", type=float, default=LOD2Config.support_voxel_size)
    return parser


def _config_from_args(args: argparse.Namespace) -> LOD2Config:
    return LOD2Config(
        reconstruction_mode=args.mode,
        input_ply=args.input_ply,
        output_obj=args.output_obj,
        psdr_dir=args.psdr_dir,
        psdr_white_script=args.psdr_white_script,
        psdr_min_points=args.psdr_min_points,
        psdr_simplify_tolerance=args.psdr_simplify_tolerance,
        psdr_merge_angle_deg=args.psdr_merge_angle_deg,
        psdr_merge_offset=args.psdr_merge_offset,
        psdr_roof_simplify_factor=args.psdr_roof_simplify_factor,
        psdr_wall_simplify_factor=args.psdr_wall_simplify_factor,
        psdr_envelope_min_points=args.psdr_envelope_min_points,
        psdr_envelope_base_percentile=args.psdr_envelope_base_percentile,
        psdr_envelope_roof_min_abs_z=args.psdr_envelope_roof_min_abs_z,
        psdr_envelope_roof_min_height=args.psdr_envelope_roof_min_height,
        psdr_envelope_roof_min_z_percentile=args.psdr_envelope_roof_min_z_percentile,
        plane_distance_threshold=args.plane_distance_threshold,
        plane_ransac_n=args.plane_ransac_n,
        plane_iterations=args.plane_iterations,
        plane_min_points=args.plane_min_points,
        plane_max_count=args.plane_max_count,
        plane_remaining_ratio=args.plane_remaining_ratio,
        wall_z_threshold=args.wall_z_threshold,
        roof_min_points=args.roof_min_points,
        roof_min_abs_z=args.roof_min_abs_z,
        roof_grid_resolution=args.roof_grid_resolution,
        roof_merge_angle_deg=args.roof_merge_angle_deg,
        roof_merge_offset=args.roof_merge_offset,
        roof_max_merged_planes=args.roof_max_merged_planes,
        roof_support_alpha=args.roof_support_alpha,
        roof_support_simplify_tolerance=args.roof_support_simplify_tolerance,
        roof_partition_use_support_boundaries=args.roof_partition_use_support_boundaries,
        roof_partition_support_abs_z_min=args.roof_partition_support_abs_z_min,
        wall_alpha_shape_alpha=args.wall_alpha_shape_alpha,
        all_points_alpha_shape_alpha=args.all_points_alpha_shape_alpha,
        footprint_strategy=args.footprint_strategy,
        footprint_max_points=args.footprint_max_points,
        footprint_simplify_tolerance=args.footprint_simplify_tolerance,
        footprint_buffer=args.footprint_buffer,
        base_z_percentile=args.base_z_percentile,
        min_building_height=args.min_building_height,
        compod_env_name=args.compod_env_name,
        compod_device=args.compod_device,
        compod_home_dir=args.compod_home_dir,
        compod_padding=args.compod_padding,
        compod_area_weight=args.compod_area_weight,
        compod_cc_weight=args.compod_cc_weight,
        compod_min_component_area_ratio=args.compod_min_component_area_ratio,
        compod_verbosity=args.compod_verbosity,
        compod_exact=args.compod_exact,
        compod_simplify_tree=args.compod_simplify_tree,
        compod_surface_mode=args.compod_surface_mode,
        support_mesh_dir=args.support_mesh_dir,
        support_mesh_name=args.support_mesh_name,
        support_poisson_depth=args.support_poisson_depth,
        support_poisson_scale=args.support_poisson_scale,
        support_density_quantile=args.support_density_quantile,
        support_target_triangles=args.support_target_triangles,
        support_crop_padding_ratio=args.support_crop_padding_ratio,
        support_normal_k=args.support_normal_k,
        support_voxel_size=args.support_voxel_size,
    )


def run_legacy_pipeline(config: LOD2Config) -> Path:
    print(f"[Step 1] 读取 3DGS 点云并推导法线: {config.input_ply}")
    gaussians = load_gaussian_ply(config.input_ply)
    print(f"  点数: {gaussians.points.shape[0]}")
    print(f"  法线 |z| 分位数: {np.percentile(np.abs(gaussians.normals[:, 2]), [25, 50, 75])}")

    print("[Step 2] RANSAC 提取大平面...")
    extraction = extract_planes(gaussians, config)
    roof_planes = [
        plane
        for plane in extraction.planes
        if plane.plane_type == "roof"
        and plane.points.shape[0] >= config.roof_min_points
        and np.abs(plane.equation[2]) >= config.roof_min_abs_z
    ]
    wall_planes = [plane for plane in extraction.planes if plane.plane_type == "wall"]

    if not wall_planes:
        raise ValueError("没有检测到墙面平面，无法估计 footprint。")
    if not roof_planes:
        raise ValueError("没有检测到屋顶平面，无法构建屋顶。")

    wall_points = np.concatenate([plane.points for plane in wall_planes], axis=0)
    print(
        f"  共提取 {len(extraction.planes)} 个平面，其中墙面 {len(wall_planes)} 个，"
        f"屋顶 {len(roof_planes)} 个。"
    )

    print("[Step 3] 由墙面点生成二维 footprint，并准备墙体拉伸...")
    footprint = build_footprint(wall_points, gaussians.points, config)
    print(
        f"  footprint 面积: {footprint.polygon.area:.3f}, "
        f"外轮廓顶点数: {len(footprint.polygon.exterior.coords) - 1}"
    )

    print("[Step 4] 用屋顶平面裁剪拉伸体，生成 watertight OBJ...")
    mesh_result = build_lod2_mesh(footprint.polygon, roof_planes, gaussians.points, config)
    output_path = Path(config.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_result.mesh.export(output_path)

    print(
        f"  OBJ 已导出: {output_path}\n"
        f"  顶点数: {len(mesh_result.mesh.vertices)}, 面片数: {len(mesh_result.mesh.faces)}, "
        f"watertight: {mesh_result.mesh.is_watertight}"
    )
    return output_path


def run_pipeline(config: LOD2Config) -> Path:
    if config.reconstruction_mode == "psdr_envelope":
        return run_psdr_envelope_pipeline(config)
    if config.reconstruction_mode == "hybrid_compod":
        print(
            "[Hybrid] 使用 3DGS 法线约束的泊松支撑网格 + "
            f"PSDR 平面 + COMPOD/COMPOSE: {config.psdr_dir}"
        )
        return run_hybrid_compod_pipeline(config)
    if config.reconstruction_mode == "psdr_white":
        print(f"[PSDR] 使用已有平面分组结果生成 watertight LOD2: {config.psdr_dir}")
        return run_psdr_white_pipeline(config)
    if config.reconstruction_mode == "legacy_3dgs":
        return run_legacy_pipeline(config)
    raise ValueError(
        f"不支持的模式: {config.reconstruction_mode}。可选: psdr_envelope, hybrid_compod, psdr_white, legacy_3dgs"
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
