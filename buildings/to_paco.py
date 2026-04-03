import numpy as np
import open3d as o3d
import os
import json

def process_single_cluster(points, normals, target_points, distance_threshold, min_points_ratio):
    """处理单个建筑团块的逻辑 (RANSAC 平面提取 + 归一化)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    plane_ids = np.full(len(points), -1, dtype=np.float32)
    remaining_pcd = pcd
    remaining_indices = np.arange(len(points))
    # 这里从 1 开始编号，避免出现 0 类；适合“已经没有背景”的建筑数据
    plane_id = 1
    
    # RANSAC
    while len(remaining_pcd.points) > min_points_ratio * len(points):
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
        if len(inliers) < 10:
            break
        original_inliers = remaining_indices[inliers]
        plane_ids[original_inliers] = plane_id
        plane_id += 1
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_indices = np.delete(remaining_indices, inliers)
        
    if len(remaining_indices) > 0:
        plane_ids[remaining_indices] = plane_id

    # 拼接并采样
    data = np.hstack((points, normals, plane_ids[:, None]))
    if len(data) > target_points:
        idx = np.random.choice(len(data), target_points, replace=False)
    else:
        idx = np.random.choice(len(data), target_points, replace=True)
    data = data[idx]
    
    # 计算归一化参数（重点：为了后续合并，必须返回这些参数）
    xyz = data[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_distance = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    # 防止单个极小块导致除以 0
    max_distance = max_distance if max_distance > 0 else 1.0 
    xyz = xyz / max_distance
    
    data[:, :3] = xyz
    return data.astype(np.float32), centroid.tolist(), float(max_distance)

def prepare_multi_buildings(input_ply_path, output_dir, target_points=2048):
    print(f"开始处理多建筑场景点云: {input_ply_path}")
    os.makedirs(output_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(input_ply_path)
    
    # 估算法线
    print("正在估算法线...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # DBSCAN 聚类切割单体建筑
    print("正在使用 DBSCAN 切割单体建筑...")
    # eps: 聚类距离阈值(米)，需要根据你的 UAV 数据实际比例调整
    # min_points: 构成一个独立建筑的最少点数
    labels = np.array(pcd.cluster_dbscan(eps=1.5, min_points=500, print_progress=True))
    
    max_label = labels.max()
    print(f"聚类完成，共发现 {max_label + 1} 个建筑团块。")
    
    metadata = {}
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    valid_clusters = 0
    for i in range(max_label + 1):
        cluster_idx = np.where(labels == i)[0]
        # 过滤掉太小的碎块
        if len(cluster_idx) < 1000: 
            continue
            
        cluster_points = points[cluster_idx]
        cluster_normals = normals[cluster_idx]
        
        print(f"处理建筑 {valid_clusters} (点数: {len(cluster_points)})...")
        data, centroid, scale = process_single_cluster(
            cluster_points, cluster_normals, target_points, 
            distance_threshold=0.1, min_points_ratio=0.05
        )
        
        # 保存单个建筑的输入张量
        file_name = f"building_{valid_clusters}.npy"
        np.save(os.path.join(output_dir, file_name), data)
        
        # 记录反向恢复坐标所需的参数
        metadata[file_name] = {
            "centroid": centroid,
            "scale": scale
        }
        valid_clusters += 1
        
    # 保存元数据
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"全部预处理完成！有效建筑 {valid_clusters} 个，已保存至 {output_dir}")

if __name__ == "__main__":
    input_cloud = "/media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/cleaned_buildings.ply"
    # 创建一个专属文件夹存放切分后的数据
    output_dir = "/media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/paco_chunks"
    
    prepare_multi_buildings(input_cloud, output_dir)