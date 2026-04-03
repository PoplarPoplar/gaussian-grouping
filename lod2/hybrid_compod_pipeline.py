from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from lod2.config import LOD2Config
from lod2.io_utils import load_gaussian_ply
from lod2.support_mesh import build_poisson_support_mesh


def _resolve_point_groups_path(config: LOD2Config) -> Path:
    point_groups_path = Path(config.psdr_dir) / "point_groups.npz"
    if not point_groups_path.exists():
        raise FileNotFoundError(f"未找到 PSDR 平面分组结果: {point_groups_path}")
    return point_groups_path


def _build_compod_command(
    config: LOD2Config,
    point_groups_path: Path,
    support_mesh_path: Path,
    output_path: Path,
) -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    quoted_repo_root = shlex.quote(str(repo_root))
    quoted_point_groups = shlex.quote(str(point_groups_path))
    quoted_support_mesh = shlex.quote(str(support_mesh_path))
    quoted_output = shlex.quote(str(output_path))
    quoted_device = shlex.quote(str(config.compod_device))
    quoted_home = shlex.quote(str(config.compod_home_dir))
    worker_command = (
        f"cd {quoted_repo_root} && "
        f"python -m lod2.compod_worker "
        f"--vg-file {quoted_point_groups} "
        f"--mesh-file {quoted_support_mesh} "
        f"--output-obj {quoted_output} "
        f"--device {quoted_device} "
        f"--padding {config.compod_padding} "
        f"--area-weight {config.compod_area_weight} "
        f"--min-component-area-ratio {config.compod_min_component_area_ratio} "
        f"--verbosity {config.compod_verbosity} "
        f"--home-dir {quoted_home}"
    )
    if config.compod_cc_weight is not None:
        worker_command += f" --cc-weight {config.compod_cc_weight}"
    if config.compod_exact:
        worker_command += " --exact"
    if config.compod_simplify_tree:
        worker_command += " --simplify-tree"
    else:
        worker_command += " --no-simplify-tree"
    worker_command += f" --surface-mode {shlex.quote(config.compod_surface_mode)}"

    return [
        "conda",
        "run",
        "-n",
        config.compod_env_name,
        "bash",
        "-lc",
        worker_command,
    ]


def run_hybrid_compod_pipeline(config: LOD2Config) -> Path:
    point_groups_path = _resolve_point_groups_path(config)
    output_path = Path(config.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    support_mesh_dir = Path(config.support_mesh_dir)
    support_mesh_dir.mkdir(parents=True, exist_ok=True)
    support_mesh_path = support_mesh_dir / f"{output_path.stem}_{config.support_mesh_name}"

    print(f"[Step 1] 读取 3DGS 点云并推导法线: {config.input_ply}")
    gaussians = load_gaussian_ply(config.input_ply)
    print(f"  点数: {gaussians.points.shape[0]}")

    print("[Step 2] 生成泊松支撑网格，用于约束封闭体 inside/outside...")
    support_mesh = build_poisson_support_mesh(gaussians, config, support_mesh_path)
    print(
        f"  支撑网格: {support_mesh.mesh_path}\n"
        f"  顶点数: {support_mesh.vertex_count}, 三角面数: {support_mesh.triangle_count}"
    )

    print("[Step 3] 读取 PSDR 平面分组，并交给 COMPOD/COMPOSE 做 plane arrangement...")
    print(f"  平面分组: {point_groups_path}")

    print("[Step 4] 用 mesh-labeling 导出 watertight、棱角化的 LOD2 OBJ...")
    env = os.environ.copy()
    env["HOME"] = config.compod_home_dir
    env["XDG_CACHE_HOME"] = str(Path(config.compod_home_dir) / ".cache")
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = (
        f"{repo_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )

    command = _build_compod_command(config, point_groups_path, support_mesh_path, output_path)
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip())
        raise RuntimeError(f"COMPOD/COMPOSE 调用失败，退出码 {result.returncode}")

    if not output_path.exists():
        raise FileNotFoundError(f"COMPOD/COMPOSE 未生成输出文件: {output_path}")
    return output_path
