import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import math
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
    # # 修改后的代码：加入置信度过滤
    # logits = np.dot(f_obj_features, weight.T) + bias
    # # 1. 计算 Softmax 概率
    # exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # 减去max防溢出
    # probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # # 2. 提取类别为 1 (建筑) 的概率
    # building_probs = probs[:, 1]
    
    # # 3. 设定严格的阈值 (您可以尝试 0.8, 0.9, 0.95 等)
    # THRESHOLD = 0.90
    # building_mask = (building_probs > THRESHOLD)
    # 原来的代码
    # 计算 Logits (矩阵乘法) 
    logits = np.dot(f_obj_features, weight.T) + bias
    
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    building_probs = probs[:, 1]
    
    # ---------------------------------------------------------
    # 修正 1：温和的置信度阈值 (保全阴影和边界区域)
    # ---------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.55  # 只要偏向建筑就保留，不要0.85那么苛刻
    building_mask = (building_probs > CONFIDENCE_THRESHOLD)

    # ---------------------------------------------------------
    # 修正 2：防误杀的物理属性过滤 (只杀极端的漂浮雾气和巨无霸)
    # ---------------------------------------------------------
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    opacities = sigmoid(vertices['opacity'])
    # 降低透明度门槛，保留建筑边缘和细小结构 (如栏杆/天线)
    opacity_mask = opacities > 0.15  
    
    scale_0 = np.exp(vertices['scale_0'])
    scale_1 = np.exp(vertices['scale_1'])
    scale_2 = np.exp(vertices['scale_2'])
    max_scales = np.maximum(scale_0, np.maximum(scale_1, scale_2))
    
    # 极其重要：把 95 改成 99.5！
    # 我们只删掉那万分之五的、真正巨大的“天空背景球”，绝对不碰正常的平铺墙面球
    scale_thresh = np.percentile(max_scales[building_mask], 99.5) 
    scale_mask = max_scales < scale_thresh

    # 终极 Mask 融合
    final_mask = building_mask & opacity_mask & scale_mask
    num_buildings = np.sum(final_mask)
    print(f"[!] 优化提取后，共发现 {num_buildings} 个高保真建筑物点")
    
    building_vertices = vertices[final_mask].copy() 

    # ---------------------------------------------------------
    # 修正 3：彻底删除“动态几何微缩 (Shrinking)” 代码
    # 绝对保持原始高斯的 scale 原封不动，防止出现裂缝和密度降低
    # (即：不要再对 building_vertices['scale_0'] 执行减法操作了)
    # ---------------------------------------------------------

    # 构建并保存新的 PLY
    print(f"[*] 正在保存高密度建筑点云至: {out_path}")
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