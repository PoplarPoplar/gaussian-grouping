import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


# Spherical Harmonics constants for SH to RGB conversion
C0 = 0.28209479177387814


def sh2rgb(sh_coeff):
    """Convert SH coefficients to RGB using the DC component formula."""
    return sh_coeff * C0 + 0.5


def read_gaussian_xyz(ply_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"].data
    xyz = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)
    
    # 尝试读取RGB颜色：优先用 f_dc_0/1/2（需要SH转换），否则用 red/green/blue
    try:
        sh_coeff = np.zeros((len(vertices), 3), dtype=np.float32)
        sh_coeff[:, 0] = vertices["f_dc_0"]
        sh_coeff[:, 1] = vertices["f_dc_1"]
        sh_coeff[:, 2] = vertices["f_dc_2"]
        # Apply SH to RGB conversion
        rgb = sh2rgb(sh_coeff)
    except (ValueError, KeyError):
        try:
            rgb = np.stack(
                [vertices["red"], vertices["green"], vertices["blue"]],
                axis=1,
            ).astype(np.float32) / 255.0
        except (ValueError, KeyError):
            rgb = np.ones((len(vertices), 3), dtype=np.float32) * 0.5
    
    return xyz, vertices, rgb


def write_sparse_ply(xyz, out_path):
    normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate([xyz, normals, rgb.astype(np.float32)], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v, 0.0
    return v / n, n


def get_world2cam(eye, target, up_hint):
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

    # 对当前流程并不强依赖 points3D，但写一个空文件更稳妥
    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def project_points(xyz, rgb, R_w2c, t_w2c, fx, fy, cx, cy, width, height):
    pts_cam = (R_w2c @ xyz.T).T + t_w2c[None, :]
    valid = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[valid]
    xyz_valid = xyz[valid]
    rgb_valid = rgb[valid] if rgb is not None else np.ones((pts_cam.shape[0], 3), dtype=np.float32)
    
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if pts_cam.shape[0] == 0:
        return rgb_image, mask

    u = fx * (pts_cam[:, 0] / pts_cam[:, 2]) + cx
    v = fy * (pts_cam[:, 1] / pts_cam[:, 2]) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = ui[inside]
    vi = vi[inside]
    rgb_valid = rgb_valid[inside]
    
    rgb_image[vi, ui] = np.clip(rgb_valid, 0, 1)
    mask[vi, ui] = 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return rgb_image, mask


def save_virtual_view(out_dir, view_name, rgb, mask):
    image_dir = Path(out_dir) / "images"
    mask_dir = Path(out_dir) / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_dir / view_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_dir / view_name), mask)


def choose_up_hint(up_axis):
    if up_axis.upper() == "Z":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def generate_virtual_colmap(
    ply_path,
    out_dir,
    num_orbit=24,
    num_top=4,
    width=1024,
    height=1024,
    fov_deg=60.0,
    zoom_factor=1.2,
    up_axis="Z",
    save_sparse_ply=True,
):
    ply_path = Path(ply_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz, vertices, rgb = read_gaussian_xyz(ply_path)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min

    radius_xy = max(extent[0], extent[1]) * zoom_factor
    radius_xy = max(radius_xy, 1e-3)
    height_offset = extent[2] * 0.25
    up_hint = choose_up_hint(up_axis)
    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32) if up_axis.upper() == "Z" else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    fx = fy = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx = width * 0.5
    cy = height * 0.5

    views = []

    orbit_elevation = np.deg2rad(15.0)
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

        q, t, R_w2c = get_world2cam(eye, center, up_hint)
        rgb_image, mask = project_points(xyz, rgb, R_w2c, t, fx, fy, cx, cy, width, height)
        rgb_display = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        name = f"render_orbit_{i:04d}.png"
        save_virtual_view(out_dir, name, rgb_display, mask)
        views.append({"name": name, "q": q, "t": t})

    top_distance = max(extent.max() * 1.6, radius_xy)
    for i in range(num_top):
        jitter = np.array([0.05 * extent[0], 0.05 * extent[1], 0.0], dtype=np.float32)
        offset = np.random.normal(0.0, 1.0, 3).astype(np.float32) * jitter
        eye = center + up_vec * top_distance + offset
        q, t, R_w2c = get_world2cam(eye, center, up_hint)
        rgb_image, mask = project_points(xyz, rgb, R_w2c, t, fx, fy, cx, cy, width, height)
        rgb_display = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
        name = f"render_topdown_{i:04d}.png"
        save_virtual_view(out_dir, name, rgb_display, mask)
        views.append({"name": name, "q": q, "t": t})

    write_colmap_cameras(out_dir, width, height, fx, fy, cx, cy, views)

    if save_sparse_ply:
        sparse_ply = out_dir / "sparse" / "0" / "points3D.ply"
        write_sparse_ply(xyz, sparse_ply)

    print(f"[OK] 已生成虚拟 COLMAP 数据集: {out_dir}")
    print(f"[OK] 图像目录: {out_dir / 'images'}")
    print(f"[OK] Mask 目录: {out_dir / 'mask'}")
    print(f"[OK] Sparse 目录: {out_dir / 'sparse' / '0'}")


def find_default_ply(input_dir):
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
    parser.add_argument("--num_top", type=int, default=4, help="俯视视角数量")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--zoom_factor", type=float, default=1.2)
    parser.add_argument("--up_axis", type=str, default="Z", choices=["Z", "Y"])
    parser.add_argument("--no_sparse_ply", action="store_true", help="不写入 sparse/0/points3D.ply")
    args = parser.parse_args()

    ply_path = Path(args.ply) if args.ply else find_default_ply(args.input_dir)
    print(f"[INFO] 使用输入 PLY: {ply_path}")
    generate_virtual_colmap(
        ply_path=ply_path,
        out_dir=args.out_dir,
        num_orbit=args.num_orbit,
        num_top=args.num_top,
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
        zoom_factor=args.zoom_factor,
        up_axis=args.up_axis,
        save_sparse_ply=not args.no_sparse_ply,
    )