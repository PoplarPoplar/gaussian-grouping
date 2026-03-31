import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import argparse

def extract_buildings(ply_path, classifier_path, out_path):
    print(f"[*] 加载轻量级分类器: {classifier_path}")
    # 读取分类器权重 (映射 16维 -> 2分类)
    classifier_state = torch.load(classifier_path, map_location='cpu')
    
    # 动态匹配权重和偏置的键名（Gaussian-Grouping 默认为单层 Conv2d 权重）
    weight_key = [k for k in classifier_state.keys() if 'weight' in k][0]
    bias_key = [k for k in classifier_state.keys() if 'bias' in k][0]
    
    # 转换形状为 (2, 16) 和 (2,)
    weight = classifier_state[weight_key].squeeze().numpy()
    bias = classifier_state[bias_key].squeeze().numpy()
    
    print(f"[*] 读取 3D 高斯点云 (可能需要几分钟，请耐心等待): {ply_path}")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data
    
    # 提取 16维的分组特征
    print("[*] 正在提取 3D 语义特征并进行推理...")
    num_points = len(vertices)
    f_obj_features = np.zeros((num_points, 16), dtype=np.float32)
    for i in range(16):
            f_obj_features[:, i] = vertices[f'obj_dc_{i}']
        
    # 计算 Logits (矩阵乘法) 并获取预测类别
    
    # 原来的代码
    logits = np.dot(f_obj_features, weight.T) + bias
    preds = np.argmax(logits, axis=1)
    building_mask = (preds == 1)
    num_buildings = np.sum(building_mask)
    print(f"[!] 发现 {num_buildings} 个建筑物高斯点 (总点数: {num_points})")
    
    print(f"[*] 正在剥离非建筑区域，另存为: {out_path}")
    building_vertices = vertices[building_mask]
    
    # 构建并保存新的 PLY
    new_element = PlyElement.describe(building_vertices, 'vertex')
    PlyData([new_element], text=False, byte_order='<').write(out_path)
    print("[*] 建筑物提取完美完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract buildings from trained Gaussian Grouping PLY")
    parser.add_argument("--ply", type=str, required=True, help="Path to the trained point_cloud.ply")
    parser.add_argument("--classifier", type=str, required=True, help="Path to the classifier.pth")
    parser.add_argument("--out", type=str, default="pure_buildings.ply", help="Output path for the pure buildings PLY")
    
    args = parser.parse_args()
    extract_buildings(args.ply, args.classifier, args.out)