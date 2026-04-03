import numpy as np
from plyfile import PlyData
import open3d as o3d

def visualize_gaussian_features(ply_path):
    print(f"正在读取 {ply_path} ...")
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']

    # 1. 提取坐标
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    points = np.vstack((x, y, z)).T

    # 2. 提取实例特征 (ins_feat_r, g, b)
    r = vertex_data['ins_feat_r']
    g = vertex_data['ins_feat_g']
    b = vertex_data['ins_feat_b']
    colors = np.vstack((r, g, b)).T

    # 归一化颜色到 0-1 之间 (Open3D 的颜色要求)
    # 如果你的特征值本身就是特征向量而不是颜色，这里可能会显示出随机但有区分度的色块
    color_min = colors.min(axis=0)
    color_max = colors.max(axis=0)
    # 避免除以0
    color_range = np.where((color_max - color_min) == 0, 1, color_max - color_min)
    colors_normalized = (colors - color_min) / color_range

    # 3. 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # 顺便检查一下自带的 nx, ny, nz 是否有效
    nx = vertex_data['nx']
    ny = vertex_data['ny']
    nz = vertex_data['nz']
    # 如果法线不是全0，我们就把它赋给点云
    if np.any(nx) or np.any(ny) or np.any(nz):
        normals = np.vstack((nx, ny, nz)).T
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print("检测到自带法线信息！")
    else:
        print("自带的 nx, ny, nz 似乎为空 (全0)。后续可能需要通过四元数推导法线。")

    print(f"成功加载 {len(points)} 个点，正在打开可视化窗口...")
    
    # 4. 可视化
    # 提示：在弹出的窗口中，按 'n' 键可以显示法线，按 '+/-' 调整法线长度
    o3d.visualization.draw_geometries([pcd], window_name="3DGS Instance Features")

if __name__ == "__main__":
    # 请将这里的路径替换为你的 ply 文件路径
    visualize_gaussian_features("/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings/instance_001.ply")