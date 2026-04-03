#!/usr/bin/env python3
import argparse
import colorsys
import numpy as np
from plyfile import PlyData, PlyElement
from numpy.lib import recfunctions as rfn


DEFAULT_PALETTE = {
    0: (128, 128, 128),  # 背景：灰
    1: (255, 64, 64),    # 类1：红
    2: (64, 200, 255),   # 类2：蓝青
}


def class_to_color(class_ids: np.ndarray, use_default_palette: bool = True) -> np.ndarray:
    colors = np.zeros((class_ids.shape[0], 3), dtype=np.uint8)

    unique_ids = np.unique(class_ids)
    for cls_id in unique_ids:
        if use_default_palette and int(cls_id) in DEFAULT_PALETTE:
            color = DEFAULT_PALETTE[int(cls_id)]
        else:
            hue = (int(cls_id) * 0.61803398875) % 1.0
            sat = 0.75
            val = 0.95
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            color = tuple(int(c * 255) for c in rgb)

        mask = class_ids == cls_id
        colors[mask] = color

    return colors


def main():
    parser = argparse.ArgumentParser(description="按 pred_class 给点云上色并导出 PLY")
    parser.add_argument("--in_ply", required=True, type=str, help="输入带 pred_class 的 PLY")
    parser.add_argument("--out_ply", required=True, type=str, help="输出上色后的 PLY")
    parser.add_argument("--overwrite_rgb", action="store_true", help="若已有 red/green/blue，直接覆盖")
    parser.add_argument("--no_default_palette", action="store_true", help="不用固定0/1/2配色，所有类别都自动配色")
    args = parser.parse_args()

    ply = PlyData.read(args.in_ply)
    vertices = ply["vertex"].data
    names = vertices.dtype.names

    if "pred_class" not in names:
        raise ValueError(
            "输入 PLY 不包含 pred_class 字段。\n"
            "请先运行 prepare/export_point_labels.py 生成带语义标签的点云。"
        )

    pred_class = np.asarray(vertices["pred_class"], dtype=np.int32)
    rgb = class_to_color(pred_class, use_default_palette=not args.no_default_palette)

    has_rgb = all(ch in names for ch in ("red", "green", "blue"))

    new_vertices = vertices
    if has_rgb and args.overwrite_rgb:
        new_vertices["red"] = rgb[:, 0]
        new_vertices["green"] = rgb[:, 1]
        new_vertices["blue"] = rgb[:, 2]
    elif has_rgb and not args.overwrite_rgb:
        new_vertices = rfn.append_fields(
            vertices,
            ["vis_red", "vis_green", "vis_blue"],
            [rgb[:, 0], rgb[:, 1], rgb[:, 2]],
            dtypes=["u1", "u1", "u1"],
            usemask=False,
            asrecarray=False,
        )
    else:
        new_vertices = rfn.append_fields(
            vertices,
            ["red", "green", "blue"],
            [rgb[:, 0], rgb[:, 1], rgb[:, 2]],
            dtypes=["u1", "u1", "u1"],
            usemask=False,
            asrecarray=False,
        )

    out_el = PlyElement.describe(new_vertices, "vertex")
    PlyData([out_el], text=False, byte_order="<").write(args.out_ply)

    unique_ids, counts = np.unique(pred_class, return_counts=True)
    print("可视化类别统计：")
    for cls_id, cnt in zip(unique_ids.tolist(), counts.tolist()):
        print(f"  class {cls_id}: {cnt}")
    print(f"已写出上色点云: {args.out_ply}")


if __name__ == "__main__":
    main()
