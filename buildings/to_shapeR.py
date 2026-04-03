import os
import pickle
import torch
import numpy as np
import cv2
import pycolmap
import trimesh
from pathlib import Path

def read_colmap_data(sparse_path):
    """读取 COLMAP 稀疏重建的位姿和内参"""
    reconstruction = pycolmap.Reconstruction(sparse_path)
    colmap_data = {'images': [], 'poses': [], 'intrinsics': []}
    
    for _, image in reconstruction.images.items():
        # 获取位姿对象：注意这里加了括号 () 把它当作方法调用
        try:
            # 尝试作为方法调用并获取矩阵
            cam_from_world = image.cam_from_world() 
            T_w_c = cam_from_world.matrix()
        except TypeError:
            # 如果它不是方法而是属性，则直接使用
            cam_from_world = image.cam_from_world
            T_w_c = cam_from_world.matrix()
        except AttributeError:
            # 最后的保底方案：手动从位姿对象提取
            pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            rotation_matrix = pose.rotation.matrix()
            translation_vector = pose.translation
            T_w_c = np.eye(4)
            T_w_c[:3, :3] = rotation_matrix
            T_w_c[:3, 3] = translation_vector

        # 获取相机内参 K
        camera = reconstruction.cameras[image.camera_id]
        # ShapeR 预期标准的 3x3 内参矩阵
        K = np.array([
            [camera.params[0], 0, camera.params[2]],
            [0, camera.params[1], camera.params[3]],
            [0, 0, 1]
        ])
        
        colmap_data['images'].append(image.name)
        colmap_data['poses'].append(T_w_c)
        colmap_data['intrinsics'].append(K)
    return colmap_data

def load_gaussian_ply(ply_path):
    """加载 PLY 格式的高斯点云并提取坐标"""
    # 使用 trimesh 加载点云
    pc = trimesh.load(ply_path)
    # 提取 XYZ 坐标，ShapeR 编码器预期输入为 [N, 3]
    points = np.array(pc.vertices).astype(np.float32)
    return points

def process_buildings(base_path, output_dir):
    """处理每个建筑的点云并封装为 ShapeR 的 .pkl 格式"""
    base_path = Path(base_path)
    building_dir = base_path / "building"
    image_dir = base_path / "images"
    sparse_path = base_path / "sparse/0"
    
    os.makedirs(output_dir, exist_ok=True)
    colmap_info = read_colmap_data(str(sparse_path))

    # 遍历 building 文件夹下的所有 .ply 文件
    for ply_file in building_dir.glob("*.ply"):
        building_name = ply_file.stem
        print(f"正在处理: {building_name}")
        
        # 1. 提取点云几何坐标
        points = load_gaussian_ply(ply_file)
        points_tensor = torch.from_numpy(points)
        
        # 2. 筛选 16 个视角以适配 ShapeR 的 quality 预设
        num_views = 16
        indices = np.linspace(0, len(colmap_info['images']) - 1, num_views, dtype=int)
        
        imgs, poses, Ks = [], [], []
        for idx in indices:
            img_path = image_dir / colmap_info['images'][idx]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) # 调整至模型输入分辨率
            img = img.transpose(2, 0, 1) / 255.0 # 归一化并调整通道顺序
            
            imgs.append(img)
            poses.append(colmap_info['poses'][idx])
            Ks.append(colmap_info['intrinsics'][idx])
            
        # 3. 构造数据包
        # ShapeR 会在 InferenceDataset 中自动处理这些数据
        data = {
            "name": building_name,
            "semi_dense_points": points_tensor, # 用于 VecSet 潜变量条件
            "images": torch.from_numpy(np.array(imgs)).float(),
            "poses": torch.from_numpy(np.array(poses)).float(),
            "K": torch.from_numpy(np.array(Ks)).float(),
            "caption": f"A 3D reconstruction of a building called {building_name}",
            "masks_ingest": torch.ones((num_views, 224, 224)), # 默认无遮罩
            "index": 0 
        }
        
        output_path = Path(output_dir) / f"{building_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"已生成: {output_path}")

if __name__ == "__main__":
    SOURCE_PATH = "/media/liu/my_pssd/program/data_milo_run/paco/hj"
    TARGET_DIR = "/media/liu/my_pssd/program/data_milo_run/paco/hj/shapeR_data" # 输出到 ShapeR 的数据存放目录
    process_buildings(SOURCE_PATH, TARGET_DIR)