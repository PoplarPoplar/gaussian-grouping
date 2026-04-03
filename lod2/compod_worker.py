from __future__ import annotations

import argparse
import os
from pathlib import Path

import trimesh


def _force_runtime_dirs(home_dir: str) -> None:
    os.environ["HOME"] = home_dir
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(home_dir) / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(home_dir) / ".config" / "matplotlib"))
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")


def _postprocess_mesh(
    mesh_path: Path,
    min_component_area_ratio: float,
) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    components = list(mesh.split(only_watertight=False))
    if components:
        total_area = sum(component.area for component in components)
        if total_area > 0.0:
            kept = [
                component
                for component in components
                if component.area >= min_component_area_ratio * total_area
            ]
            if kept:
                mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]

    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    if mesh.is_winding_consistent and mesh.volume < 0:
        mesh.invert()

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def run_compod_worker(args: argparse.Namespace) -> Path:
    _force_runtime_dirs(args.home_dir)

    from pycompod import PolyhedralComplex, VertexGroup

    regularization: dict[str, float] | None = None
    if args.area_weight is not None or args.cc_weight is not None:
        regularization = {}
        if args.area_weight is not None:
            regularization["area"] = float(args.area_weight)
        if args.cc_weight is not None:
            regularization["cc"] = float(args.cc_weight)

    vg = VertexGroup(args.vg_file, verbosity=args.verbosity, debug_export=True)
    cc = PolyhedralComplex(
        vg,
        device=args.device,
        verbosity=args.verbosity,
        padding=args.padding,
    )

    cc.construct_partition()
    cc.add_bounding_box_planes()
    cc.label_partition(
        mode="mesh",
        mesh_file=args.mesh_file,
        regularization=regularization,
    )
    if args.simplify_tree:
        cc.simplify_partition_tree_based()

    output_path = Path(args.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # `raw` 导出的是 inside/outside 边界本身，通常更贴近平面 arrangement 的原始结构；
    # `simplified` 则会进一步把同平面区域合并成更少的面，更像传统 LOD2 白模。
    if args.surface_mode == "raw":
        cc.save_surface(
            out_file=str(output_path),
            backend="trimesh",
            triangulate=False,
        )
    else:
        cc.save_simplified_surface(
            out_file=str(output_path),
            triangulate=True,
            simplify_edges=True,
            backend="trimesh",
            exact=args.exact,
        )

    mesh = _postprocess_mesh(output_path, args.min_component_area_ratio)
    mesh.export(output_path)
    print(
        f"[COMPOD] OBJ 已导出: {output_path}\n"
        f"  顶点数: {len(mesh.vertices)}, 面片数: {len(mesh.faces)}, watertight: {mesh.is_watertight}"
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COMPOD/COMPOSE worker for watertight LOD2 export")
    parser.add_argument("--vg-file", type=str, required=True)
    parser.add_argument("--mesh-file", type=str, required=True)
    parser.add_argument("--output-obj", type=str, required=True)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--area-weight", type=float, default=0.8)
    parser.add_argument("--cc-weight", type=float, default=None)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.002)
    parser.add_argument("--verbosity", type=int, default=10)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--simplify-tree", action="store_true")
    parser.add_argument("--no-simplify-tree", action="store_false", dest="simplify_tree")
    parser.add_argument("--surface-mode", type=str, default="simplified", choices=["simplified", "raw"])
    parser.add_argument("--home-dir", type=str, default="/tmp")
    parser.set_defaults(simplify_tree=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_compod_worker(args)


if __name__ == "__main__":
    main()
