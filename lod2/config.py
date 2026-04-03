from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LOD2Config:
    """集中管理可调超参数，便于外部快速试验。"""

    reconstruction_mode: str = "psdr_envelope"
    input_ply: str = "/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings/instance_001.ply"
    output_obj: str = "lod2/1/default/building_lod2_default.obj"
    psdr_dir: str = "/media/liu/my_pssd/program/data_milo_run/paco/hj/psdr"
    psdr_white_script: str = "/home/liu/code/psdr/white/generate_lod2_white_model.py"
    psdr_min_points: int = 80
    psdr_simplify_tolerance: float = 0.50
    psdr_merge_angle_deg: float = 6.0
    psdr_merge_offset: float = 1.0
    psdr_roof_simplify_factor: float = 0.40
    psdr_wall_simplify_factor: float = 0.90
    psdr_envelope_min_points: int = 120
    psdr_envelope_base_percentile: float = 50.0
    psdr_envelope_roof_min_abs_z: float = 0.25
    psdr_envelope_roof_min_height: float = 1.5
    psdr_envelope_roof_min_z_percentile: float = 40.0


    plane_distance_threshold: float = 0.15
    plane_ransac_n: int = 3
    plane_iterations: int = 1500
    plane_min_points: int = 300
    plane_max_count: int = 20
    plane_remaining_ratio: float = 0.05
    wall_z_threshold: float = 0.25

    roof_min_points: int = 600
    roof_min_abs_z: float = 0.25
    roof_grid_resolution: float = 0.5
    roof_merge_angle_deg: float = 7.0
    roof_merge_offset: float = 1.5
    roof_max_merged_planes: int = 12
    roof_support_alpha: float = 0.18
    roof_support_simplify_tolerance: float = 0.35
    roof_partition_use_support_boundaries: bool = False
    roof_partition_support_abs_z_min: float = 0.995

    wall_alpha_shape_alpha: float = 0.20
    all_points_alpha_shape_alpha: float = 0.08
    footprint_strategy: str = "hybrid"
    footprint_max_points: int = 15000
    footprint_simplify_tolerance: float = 0.05
    footprint_buffer: float = 0.0

    base_z_percentile: float = 1.0
    min_building_height: float = 2.0
    vertex_merge_digits: int = 6
    boundary_snap_epsilon: float = 1e-4
    polygon_area_epsilon: float = 1e-6
    halfplane_epsilon: float = 1e-8

    compod_env_name: str = "compod"
    compod_device: str = "gpu"
    compod_home_dir: str = "/tmp"
    compod_padding: float = 0.0
    compod_area_weight: float = 0.8
    compod_cc_weight: float | None = None
    compod_min_component_area_ratio: float = 0.002
    compod_verbosity: int = 10
    compod_exact: bool = False
    compod_simplify_tree: bool = False
    compod_surface_mode: str = "simplified"

    support_mesh_dir: str = "lod2/1/intermediate"
    support_mesh_name: str = "support_poisson.ply"
    support_poisson_depth: int = 10
    support_poisson_scale: float = 1.1
    support_density_quantile: float = 0.03
    support_target_triangles: int = 30000
    support_crop_padding_ratio: float = 0.02
    support_normal_k: int = 24
    support_voxel_size: float = 0.0
