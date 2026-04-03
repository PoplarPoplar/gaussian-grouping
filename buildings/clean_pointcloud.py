import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
import argparse

def clean_buildings(input_ply, output_ply):
    print(f"[*] 正在加载点云: {input_ply}")
    # 1. 用 plyfile 读取所有属性（保证高斯参数不丢失）
    plydata = PlyData.read(input_ply)
    vertices = plydata['vertex'].data
    
    # 提取 XYZ 坐标给 Open3D 处理
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    print(f"[*] 原始点云数量: {len(xyz)}")

    # ---------------------------------------------------------
    # 过滤阶段 1：SOR (Statistical Outlier Removal) 移除稀疏的“薄雾”噪点
    # ---------------------------------------------------------
    print("[*] 正在执行 SOR 统计滤波...")
    # nb_neighbors: 考察周围50个点; std_ratio: 距离标准差乘数，越小过滤越狠
    cl, sor_indices = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd_sor = pcd.select_by_index(sor_indices)
    
    print(f"[*] SOR 过滤后剩余点数: {len(sor_indices)}")

    bbox = pcd_sor.get_axis_aligned_bounding_box()
    print(f"[*] 点云边界框尺寸 (X, Y, Z): {bbox.get_extent()}")
    # ---------------------------------------------------------
    # 过滤阶段 2：DBSCAN 聚类移除悬空的“孤岛”块
    # ---------------------------------------------------------
    print("[*] 正在执行 DBSCAN 物理聚类 (请耐心等待)...")
    # eps: 聚类物理距离(米/坐标单位), min_points: 成为一个实体的最少点数
    # 注意：如果 COLMAP 的尺度比较小，可能需要微调 eps (比如 0.5, 1.0, 2.0)
    labels = np.array(pcd_sor.cluster_dbscan(eps=0.5, min_points=100, print_progress=True))#1.5
    
    # 统计每个聚类簇的点数
    max_label = labels.max()
    print(f"[*] 共发现 {max_label + 1} 个独立的物理建筑/块")
    
    # 我们只保留那些点数超级多的大簇（真正的建筑物），丢弃孤立的小漂浮块
    valid_dbscan_indices = []
    # 假设一栋楼至少要有 5000 个高斯点（您可以根据实际稠密度调大或调小）
    MIN_POINTS_PER_BUILDING = 800 
    
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) >= MIN_POINTS_PER_BUILDING:
            valid_dbscan_indices.extend(cluster_indices)
        else:
            # 打印被杀掉的漂浮块大小
            # print(f"    - 删除了一个包含 {len(cluster_indices)} 个点的漂浮噪块")
            pass

    print(f"[*] 聚类过滤后，保留了 {len(valid_dbscan_indices)} 个核心建筑物点")

    # ---------------------------------------------------------
    # 还原并保存
    # ---------------------------------------------------------
    # 找到最终存活的点在原始数组中的绝对索引
    final_indices = np.array(sor_indices)[valid_dbscan_indices]
    
    print(f"[*] 正在保存绝对纯净的建筑物点云至: {output_ply}")
    clean_vertices = vertices[final_indices]
    
    new_element = PlyElement.describe(clean_vertices, 'vertex')
    PlyData([new_element], text=False, byte_order='<').write(output_ply)
    print("[*] 清洗完成！现在去 MeshLab 里看看震撼的效果吧！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="pure_buildings.ply", help="输入的带有噪点的建筑物点云")
    parser.add_argument("--output", type=str, default="cleaned_buildings.ply", help="输出的绝对干净点云")
    args = parser.parse_args()
    
    clean_buildings(args.input, args.output)