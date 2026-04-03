#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from numpy.lib import recfunctions as rfn


def read_object_features(vertices, num_objects=16):
    names = vertices.dtype.names
    required = [f"obj_dc_{i}" for i in range(num_objects)]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"PLY 中缺少对象特征字段: {missing}\n"
            "请先确认这是训练过 grouping 的 point_cloud.ply（应包含 obj_dc_0..obj_dc_15）。"
        )

    features = np.stack([vertices[f"obj_dc_{i}"] for i in range(num_objects)], axis=1).astype(np.float32)
    return features


def load_classifier(classifier_path, num_objects=16):
    state = torch.load(classifier_path, map_location="cpu")
    if not isinstance(state, dict) or "weight" not in state:
        raise ValueError("无法从 classifier.pth 读取权重（缺少 key: weight）。")

    weight = state["weight"].detach().cpu().numpy()
    bias = state.get("bias", None)
    if bias is None:
        bias = np.zeros((weight.shape[0],), dtype=np.float32)
    else:
        bias = bias.detach().cpu().numpy().astype(np.float32)

    if weight.ndim != 4 or weight.shape[2:] != (1, 1):
        raise ValueError(f"不支持的 classifier 权重形状: {weight.shape}，期望 [C, {num_objects}, 1, 1]")

    num_classes = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    if in_channels != num_objects:
        raise ValueError(f"classifier 输入通道数是 {in_channels}，但脚本按 {num_objects} 维 obj 特征读取。")

    weight = weight[:, :, 0, 0].astype(np.float32)  # [C, 16]
    return weight, bias, num_classes


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="给 grouping 点云导出每点语义类别（0/1/2...）")
    parser.add_argument("--ply", required=True, type=str, help="输入 point_cloud.ply")
    parser.add_argument("--classifier", required=True, type=str, help="输入 classifier.pth")
    parser.add_argument("--out_ply", required=True, type=str, help="输出带标签的 ply")
    parser.add_argument("--num_objects", default=16, type=int, help="obj 特征维度，默认16")
    args = parser.parse_args()

    ply = PlyData.read(args.ply)
    vertices = ply["vertex"].data

    features = read_object_features(vertices, num_objects=args.num_objects)  # [N, 16]
    weight, bias, num_classes = load_classifier(args.classifier, num_objects=args.num_objects)

    logits = features @ weight.T + bias[None, :]  # [N, C]
    probs = softmax(logits)
    # Training uses 0/1 internally for wall/roof; export as 1/2 for readability.
    pred = np.argmax(probs, axis=1).astype(np.int32) + 1
    conf = np.max(probs, axis=1).astype(np.float32)

    # 统计类别分布
    unique, counts = np.unique(pred, return_counts=True)
    print("预测类别统计:")
    for cls_id, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  class {cls_id}: {cnt}")
    print(f"总点数: {pred.shape[0]}, 类别数: {num_classes}")
    print("[*] 导出标签已整体 +1：1=wall, 2=roof")

    # 追加字段到原始 vertex 结构
    new_vertices = rfn.append_fields(
        vertices,
        ["pred_class", "pred_conf"],
        [pred, conf],
        dtypes=["i4", "f4"],
        usemask=False,
        asrecarray=False,
    )

    el = PlyElement.describe(new_vertices, "vertex")
    PlyData([el], text=False, byte_order="<").write(args.out_ply)
    print(f"已写出: {args.out_ply}")


if __name__ == "__main__":
    main()