import open3d as o3d
import numpy as np
from plyfile import PlyData
import argparse

def prepare_for_psdr(input_ply, output_ply, voxel_size):
    print(f"[*] 正在加载清洗后的建筑点云: {input_ply}")
    
    # 1. 使用 plyfile 安全读取，提取纯粹的 XYZ 坐标
    # 这一步会自动丢弃高斯的颜色、透明度、缩放和 16维特征，给数据“疯狂瘦身”
    plydata = PlyData.read(input_ply)
    vertices = plydata['vertex'].data
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print(f"[*] 初始点云数量: {len(pcd.points)}")

    # ---------------------------------------------------------
    # 核心步骤 A：体素下采样 (强制点云均匀分布)
    # ---------------------------------------------------------
    print(f"[*] 正在进行体素下采样 (Voxel Size: {voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[*] 下采样后点云数量: {len(pcd_down.points)}")

    # ---------------------------------------------------------
    # 核心步骤 B：计算平滑表面法线 (PSDR 找平面的唯一依据)
    # ---------------------------------------------------------
    print("[*] 正在计算几何表面法线...")
    # 搜索半径设为体素的 2~3 倍，保证能找到足够的邻居来拟合平面
    radius = voxel_size * 3.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    # ---------------------------------------------------------
    # 核心步骤 C：法线朝向一致化 (无人机航拍神技)
    # ---------------------------------------------------------
    print("[*] 正在统一法线朝向 (强制朝向建筑外部/上方)...")
    # PSDR 极其讨厌内部和外部法线颠倒。
    # 既然是无人机数据，我们在万米高空设一个虚拟相机，让所有法线尽量朝向天空。
    # 这样可以保证墙面和屋顶的法线都是朝外的，绝不会朝向屋里。
    camera_location = np.array([0., 0., 10000.])
    pcd_down.orient_normals_towards_camera_location(camera_location)

    # ---------------------------------------------------------
    # 导出
    # ---------------------------------------------------------
    print(f"[*] 正在保存为 PSDR 标准输入点云: {output_ply}")
    # 导出为标准的二进制 PLY (只含 x, y, z, nx, ny, nz)
    o3d.io.write_point_cloud(output_ply, pcd_down, write_ascii=False)
    print("[*] 完美搞定！点云已准备就绪。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gaussian Point Cloud for PSDR/CompOD")
    parser.add_argument("--input", type=str, required=True, help="路径: 之前生成的 cleaned_buildings.ply")
    parser.add_argument("--output", type=str, default="psdr_input.ply", help="输出的标准点云路径")
    parser.add_argument("--voxel_size", type=float, default=0.2, help="体素大小 (决定点云稀疏程度，极其重要)")
    args = parser.parse_args()
    
    prepare_for_psdr(args.input, args.output, args.voxel_size)