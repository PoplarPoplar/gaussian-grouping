import argparse
import os
import sys
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

# Import SH utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.sh_utils import eval_sh, C0, C1, C2, C3, C4
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gaussian_render


def read_gaussian_xyz(ply_path):
    """读取高斯PLY，返回xyz和原始SH系数（不转换为RGB）"""
    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"].data
    
    # 读取xyz
    xyz = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)
    
    # 尝试读取f_dc系数（DC分量）
    max_sh_degree = 0
    try:
        f_dc = np.zeros((len(vertices), 3), dtype=np.float32)
        f_dc[:, 0] = vertices["f_dc_0"]
        f_dc[:, 1] = vertices["f_dc_1"]
        f_dc[:, 2] = vertices["f_dc_2"]
    except (ValueError, KeyError):
        # 降级到red/green/blue
        try:
            rgb_raw = np.stack(
                [vertices["red"], vertices["green"], vertices["blue"]],
                axis=1,
            ).astype(np.float32) / 255.0
            f_dc = (rgb_raw - 0.5) / C0  # 反向转换为"f_dc"
        except (ValueError, KeyError):
            f_dc = np.ones((len(vertices), 3), dtype=np.float32) * 0.5
    
    # 尝试读取高阶SH系数
    f_rest = None
    try:
        field_names = list(vertices.dtype.names)
        # 找出最高的SH度数
        max_idx = -1
        for name in field_names:
            if name.startswith('f_rest_'):
                idx = int(name.split('_')[-1])
                max_idx = max(max_idx, idx)
        
        if max_idx >= 0:
            # 计算SH度数：f_rest_0 到 f_rest_ (n_coeff-2)，共 (max_sh_degree+1)^2 - 1 个系数
            n_total_coeff = max_idx // 3 + 1 + 1  # +1 because DC is separate
            max_sh_degree = int(np.sqrt(n_total_coeff + 1)) - 1
            max_sh_degree = min(max_sh_degree, 3)  # Cap at degree 3
            
            n_coeff = (max_sh_degree + 1) ** 2
            f_rest = np.zeros((len(vertices), 3, n_coeff - 1), dtype=np.float32)
            
            for c in range(3):
                for i in range(n_coeff - 1):
                    field_name = f"f_rest_{c * (n_coeff - 1) + i}"
                    try:
                        f_rest[:, c, i] = vertices[field_name]
                    except (ValueError, KeyError):
                        pass
    except Exception as e:
        print(f"[Warning] Failed to read f_rest: {e}, using DC only")
        f_rest = None
    
    return xyz, f_dc, f_rest, max_sh_degree


def write_sparse_ply(xyz, out_path):
    """写入COLMAP sparse模型（仅位置和颜色）"""
    normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate([xyz, normals, rgb], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v, 0.0
    return v / n, n


def get_world2cam(eye, target, up_hint):
    """计算世界到相机坐标系的变换"""
    forward, _ = normalize(target - eye)

    right = np.cross(up_hint, forward)
    right, right_norm = normalize(right)
    if right_norm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(fallback, forward)
        right, _ = normalize(right)

    cam_up = np.cross(forward, right)
    cam_up, _ = normalize(cam_up)

    # COLMAP / Gaussian-Grouping 约定：相机坐标 y 轴向下
    down = -cam_up
    R_c2w = np.column_stack([right, down, forward])
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ eye

    quat_xyzw = Rotation.from_matrix(R_w2c).as_quat()
    q_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return q_wxyz, t_w2c.astype(np.float32), R_w2c.astype(np.float32)


def write_colmap_cameras(out_dir, width, height, fx, fy, cx, cy, views):
    """写入COLMAP相机元数据"""
    sparse_dir = Path(out_dir) / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(views)}, mean observations per image: 0\n")
        for image_id, view in enumerate(views, start=1):
            q = view["q"]
            t = view["t"]
            name = view["name"]
            f.write(
                f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {name}\n\n"
            )

    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def project_points(xyz, f_dc, f_rest, max_sh_degree, camera_center, R_w2c, t_w2c, fx, fy, cx, cy, width, height):
    """
    投影点到图像，使用SH系数基于观察方向计算颜色
    
    Args:
        xyz: (N, 3) 点的世界坐标
        f_dc: (N, 3) DC分量的SH系数
        f_rest: (N, 3, n_coeff-1) 高阶SH系数（如果存在）
        max_sh_degree: 最高SH度数
        camera_center: (3,) 相机中心世界坐标
        R_w2c: (3, 3) 世界到相机的旋转矩阵
        t_w2c: (3,) 世界到相机的平移向量
        fx, fy, cx, cy: 相机内参
        width, height: 图像尺寸
    
    Returns:
        rgb_image: (height, width, 3) RGB图像 [0, 1]
    """
    # 变换到相机坐标系
    pts_cam = (R_w2c @ xyz.T).T + t_w2c[None, :]
    
    # 计算观察方向（从点到摄像机）
    dir_pp = camera_center - xyz  # world coords
    dir_pp_norm = np.linalg.norm(dir_pp, axis=1, keepdims=True)
    dir_pp_normalized = dir_pp / (dir_pp_norm + 1e-6)
    
    # 检查有效范围（在摄像机前方）
    valid = pts_cam[:, 2] > 1e-6
    
    pts_cam_valid = pts_cam[valid]
    f_dc_valid = f_dc[valid]
    dir_pp_normalized_valid = dir_pp_normalized[valid]
    if f_rest is not None:
        f_rest_valid = f_rest[valid]
    else:
        f_rest_valid = None
    
    # 初始化输出
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    if pts_cam_valid.shape[0] == 0:
        return rgb_image
    
    # 计算RGB: 使用eval_sh基于观察方向
    if f_rest_valid is not None and max_sh_degree > 0:
        # 有高阶SH系数
        n_coeff = (max_sh_degree + 1) ** 2
        shs = np.zeros((f_dc_valid.shape[0], 3, n_coeff), dtype=np.float32)
        shs[:, :, 0] = f_dc_valid  # DC分量
        shs[:, :, 1:] = f_rest_valid
        rgb_valid = eval_sh(max_sh_degree, shs, dir_pp_normalized_valid)
    else:
        # 只有DC分量 - 使用 eval_sh 评估
        shs = np.zeros((f_dc_valid.shape[0], 3, 1), dtype=np.float32)
        shs[:, :, 0] = f_dc_valid
        rgb_valid = eval_sh(0, shs, dir_pp_normalized_valid)
    
    # 应用 SH2RGB 变换: result + 0.5
    # (注意 eval_sh 已经乘以了 C0)
    rgb_valid = np.clip(rgb_valid + 0.5, 0, 1)
    
    # 投影到图像平面
    u = fx * (pts_cam_valid[:, 0] / pts_cam_valid[:, 2]) + cx
    v = fy * (pts_cam_valid[:, 1] / pts_cam_valid[:, 2]) + cy
    
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    
    # 检查像素是否在图像范围内
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = ui[inside]
    vi = vi[inside]
    rgb_valid = rgb_valid[inside]
    
    # 填充图像
    rgb_image[vi, ui] = rgb_valid
    return rgb_image


def save_virtual_view(out_dir, view_name, rgb):
    """保存虚拟视角图像（不生成mask）"""
    image_dir = Path(out_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_dir / view_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def choose_up_hint(up_axis):
    """选择向上方向hint"""
    if up_axis.upper() == "Z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def render_view_with_gaussian_renderer(gaussians, eye, center, up_hint, width, height, fov_deg, bg_color, pipe):
    q, t_w2c, R_w2c = get_world2cam(eye, center, up_hint)

    R_c2w = R_w2c.T
    world_view_transform = torch.tensor(
        getWorld2View2(R_c2w, t_w2c), dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=math.radians(fov_deg),
        fovY=math.radians(fov_deg),
    ).transpose(0, 1).to("cuda")
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)

    camera = MiniCam(
        width=width,
        height=height,
        fovy=math.radians(fov_deg),
        fovx=math.radians(fov_deg),
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )

    with torch.no_grad():
        results = gaussian_render(camera, gaussians, pipe, bg_color)
        rgb = results["render"].permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy()

    return q, t_w2c, rgb


def generate_virtual_colmap(
    ply_path,
    out_dir,
    num_orbit=24,
    num_side_up=8,
    num_top=1,
    width=1024,
    height=1024,
    fov_deg=60.0,
    zoom_factor=1.2,
    up_axis="Z",
    orbit_elevation_deg=0.0,
    side_up_elevation_deg=35.0,
    save_sparse_ply=True,
):
    """生成虚拟COLMAP数据集"""
    ply_path = Path(ply_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取PLY: {ply_path}")
    xyz, f_dc, f_rest, max_sh_degree = read_gaussian_xyz(ply_path)
    print(f"[INFO] 点数: {xyz.shape[0]}, SH度数: {max_sh_degree}")

    gaussians = None
    use_native_renderer = False
    try:
        gaussians = GaussianModel(3)
        gaussians.load_ply(str(ply_path))
        gaussians.active_sh_degree = gaussians.max_sh_degree
        use_native_renderer = torch.cuda.is_available()
        if use_native_renderer:
            print("[INFO] 使用原生 gaussian_renderer 渲染虚拟图像")
        else:
            print("[WARN] CUDA 不可用，回退到点投影渲染")
    except Exception as e:
        print(f"[WARN] 原生渲染器初始化失败，回退到点投影渲染: {e}")

    pipe = SimpleNamespace(
        debug=False,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    )
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda") if torch.cuda.is_available() else None
    
    # 计算边界框
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min

    # 计算相机距离
    radius_xy = max(extent[0], extent[1]) * zoom_factor
    radius_xy = max(radius_xy, 1e-3)
    height_offset = extent[2] * 0.25
    up_hint = choose_up_hint(up_axis)
    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32) if up_axis.upper() == "Z" else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 相机内参
    fx = fy = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx = width * 0.5
    cy = height * 0.5

    views = []

    # 1) Z轴水平环绕（默认仰角0度）
    print(f"[INFO] 生成 {num_orbit} 个环绕视角...")
    orbit_elevation = np.deg2rad(orbit_elevation_deg)
    for i in range(num_orbit):
        angle = i * (2.0 * np.pi / num_orbit)
        if up_axis.upper() == "Z":
            eye = np.array([
                center[0] + radius_xy * np.cos(angle) * np.cos(orbit_elevation),
                center[1] + radius_xy * np.sin(angle) * np.cos(orbit_elevation),
                center[2] + radius_xy * np.sin(orbit_elevation) + height_offset,
            ], dtype=np.float32)
        else:
            eye = np.array([
                center[0] + radius_xy * np.cos(angle) * np.cos(orbit_elevation),
                center[1] + radius_xy * np.sin(orbit_elevation) + height_offset,
                center[2] + radius_xy * np.sin(angle) * np.cos(orbit_elevation),
            ], dtype=np.float32)

        if use_native_renderer and gaussians is not None:
            q, t, rgb_image = render_view_with_gaussian_renderer(
                gaussians, eye, center, up_hint, width, height, fov_deg, bg_color, pipe
            )
        else:
            q, t, R_w2c = get_world2cam(eye, center, up_hint)
            rgb_image = project_points(xyz, f_dc, f_rest, max_sh_degree, eye, R_w2c, t, fx, fy, cx, cy, width, height)
        rgb_display = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        name = f"render_orbit_{i:04d}.png"
        save_virtual_view(out_dir, name, rgb_display)
        views.append({"name": name, "q": q, "t": t})

    # 2) 侧上方环绕（固定较高仰角）
    print(f"[INFO] 生成 {num_side_up} 个侧上方视角...")
    side_up_elevation = np.deg2rad(side_up_elevation_deg)
    for i in range(num_side_up):
        angle = i * (2.0 * np.pi / max(num_side_up, 1))
        if up_axis.upper() == "Z":
            eye = np.array([
                center[0] + radius_xy * np.cos(angle) * np.cos(side_up_elevation),
                center[1] + radius_xy * np.sin(angle) * np.cos(side_up_elevation),
                center[2] + radius_xy * np.sin(side_up_elevation) + height_offset,
            ], dtype=np.float32)
        else:
            eye = np.array([
                center[0] + radius_xy * np.cos(angle) * np.cos(side_up_elevation),
                center[1] + radius_xy * np.sin(side_up_elevation) + height_offset,
                center[2] + radius_xy * np.sin(angle) * np.cos(side_up_elevation),
            ], dtype=np.float32)

        if use_native_renderer and gaussians is not None:
            q, t, rgb_image = render_view_with_gaussian_renderer(
                gaussians, eye, center, up_hint, width, height, fov_deg, bg_color, pipe
            )
        else:
            q, t, R_w2c = get_world2cam(eye, center, up_hint)
            rgb_image = project_points(xyz, f_dc, f_rest, max_sh_degree, eye, R_w2c, t, fx, fy, cx, cy, width, height)
        rgb_display = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        name = f"render_sideup_{i:04d}.png"
        save_virtual_view(out_dir, name, rgb_display)
        views.append({"name": name, "q": q, "t": t})

    # 3) 正上方俯视
    print(f"[INFO] 生成 {num_top} 个正上方视角...")
    top_distance = max(extent.max() * 1.6, radius_xy)
    for i in range(num_top):
        eye = center + up_vec * top_distance
        if use_native_renderer and gaussians is not None:
            q, t, rgb_image = render_view_with_gaussian_renderer(
                gaussians, eye, center, up_hint, width, height, fov_deg, bg_color, pipe
            )
        else:
            q, t, R_w2c = get_world2cam(eye, center, up_hint)
            rgb_image = project_points(xyz, f_dc, f_rest, max_sh_degree, eye, R_w2c, t, fx, fy, cx, cy, width, height)
        rgb_display = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        name = f"render_topdown_{i:04d}.png"
        save_virtual_view(out_dir, name, rgb_display)
        views.append({"name": name, "q": q, "t": t})

    # 写入COLMAP元数据
    write_colmap_cameras(out_dir, width, height, fx, fy, cx, cy, views)

    # 写入sparse PLY
    if save_sparse_ply:
        sparse_ply = out_dir / "sparse" / "0" / "points3D.ply"
        sparse_ply.parent.mkdir(parents=True, exist_ok=True)
        write_sparse_ply(xyz, sparse_ply)

    print(f"[OK] 已生成虚拟 COLMAP 数据集: {out_dir}")
    print(f"[OK] 图像目录: {out_dir / 'images'}")
    print(f"[OK] 已跳过 mask 生成")
    print(f"[OK] Sparse 目录: {out_dir / 'sparse' / '0'}")


def find_default_ply(input_dir):
    """在目录中查找默认的PLY文件"""
    input_dir = Path(input_dir)
    preferred = input_dir / "instance_001.ply"
    if preferred.exists():
        return preferred

    candidates = sorted(input_dir.glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到任何 .ply 文件")
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate virtual COLMAP data for frozen Gaussian-Grouping training")
    parser.add_argument("--input_dir", type=str, default="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings", help="包含建筑 ply 的文件夹")
    parser.add_argument("--ply", type=str, default="", help="直接指定输入 ply 文件，优先级高于 input_dir")
    parser.add_argument("--out_dir", type=str, required=True, help="输出虚拟数据集目录")
    parser.add_argument("--num_orbit", type=int, default=24, help="环绕视角数量")
    parser.add_argument("--num_side_up", type=int, default=8, help="侧上方环绕视角数量")
    parser.add_argument("--num_top", type=int, default=1, help="正上方俯视视角数量")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--zoom_factor", type=float, default=1.2)
    parser.add_argument("--up_axis", type=str, default="Z", choices=["Z", "Y"])
    parser.add_argument("--orbit_elevation_deg", type=float, default=0.0, help="水平环绕仰角（度）")
    parser.add_argument("--side_up_elevation_deg", type=float, default=35.0, help="侧上方环绕仰角（度）")
    parser.add_argument("--no_sparse_ply", action="store_true", help="不写入 sparse/0/points3D.ply")
    args = parser.parse_args()

    ply_path = Path(args.ply) if args.ply else find_default_ply(args.input_dir)
    print(f"[INFO] 使用输入 PLY: {ply_path}")
    generate_virtual_colmap(
        ply_path=ply_path,
        out_dir=args.out_dir,
        num_orbit=args.num_orbit,
        num_side_up=args.num_side_up,
        num_top=args.num_top,
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
        zoom_factor=args.zoom_factor,
        up_axis=args.up_axis,
        orbit_elevation_deg=args.orbit_elevation_deg,
        side_up_elevation_deg=args.side_up_elevation_deg,
        save_sparse_ply=not args.no_sparse_ply,
    )
