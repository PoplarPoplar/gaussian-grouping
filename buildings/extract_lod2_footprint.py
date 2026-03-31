import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon
import argparse

def process_building(ply_path, output_image="footprint.png"):
    print(f"[*] 读取单栋建筑点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    # ---------------------------------------------------------
    # 步骤 1：计算高斯点云的表面法线 (Normals)
    # ---------------------------------------------------------
    print("[*] 正在估算几何表面法线...")
    # 使用 KDTree 搜索周边点来拟合局部小平面，计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    # 强制让法线朝上，统一方向
    pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))

    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # ---------------------------------------------------------
    # 步骤 2：利用法线 Z 轴分量，剥离屋顶与墙面
    # ---------------------------------------------------------
    # 法线是一个单位向量 (x, y, z)。如果面是水平的(屋顶)，z 会接近 1 或 -1。
    # 如果面是垂直的(墙体)，z 会接近 0。
    # 我们设定与垂直 Z 轴夹角较小的点为屋顶（阈值 0.75 大约对应 41 度）
    z_threshold = 0.75
    roof_mask = np.abs(normals[:, 2]) > z_threshold

    roof_indices = np.where(roof_mask)[0]
    wall_indices = np.where(~roof_mask)[0]

    roof_pcd = pcd.select_by_index(roof_indices)
    wall_pcd = pcd.select_by_index(wall_indices)

    # 可视化：把屋顶涂成耀眼的红色，墙体涂成蓝色
    roof_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    wall_pcd.paint_uniform_color([0.0, 0.5, 1.0])

    print(f"[*] 切分成功：找到 {len(roof_indices)} 个屋顶点，{len(wall_indices)} 个墙体/侧面点。")
    print("[*] 正在弹出演示窗口，请用鼠标拖拽旋转查看分类效果！")
    print("[*] (查看完毕后，请关闭 3D 窗口，代码将继续提取 2D 轮廓...)")
    
    # 弹出 3D 窗口
    o3d.visualization.draw_geometries([roof_pcd, wall_pcd], window_name="Roof (Red) vs Wall (Blue)")

    # ---------------------------------------------------------
    # 步骤 3：降维投影，提取 2D 建筑物底座轮廓 (Alpha Shape)
    # ---------------------------------------------------------
    print("[*] 正在将建筑投影至 2D 平面，计算物理轮廓...")
    # 只提取 X 和 Y 坐标，相当于从正上方把建筑拍扁
    xy_points = points[:, :2]

    # 计算 Alpha Shape（凹包）。
    # alpha 值决定了轮廓像皮筋一样“勒”得多紧。
    # alpha=0 是粗糙的凸包；alpha 越大，越能贴合建筑的凹角和细节（如 L 型楼、U 型楼）。
    alpha_value = 1.5 
    
    try:
        # 生成 Shapely 多边形对象
        footprint = alphashape.alphashape(xy_points, alpha_value)
        
        # 使用 matplotlib 画出点云和提取到的红线边界
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='lightgray', label='Projected 3D Points')
        
        if isinstance(footprint, Polygon):
            x, y = footprint.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, label='LOD2 Footprint')
        else: # 处理建筑物断裂形成多个多边形的情况 (MultiPolygon)
            for i, geom in enumerate(footprint.geoms):
                x, y = geom.exterior.xy
                ax.plot(x, y, color='red', linewidth=2)
        
        plt.legend()
        plt.title("2D Building Footprint Extraction (Alpha Shape)")
        plt.axis('equal') # 保证 X 和 Y 轴比例一致，房子不会变形
        
        plt.savefig(output_image, dpi=300)
        print(f"[*] 二维轮廓图已保存至: {output_image}")
        plt.show() # 弹出 2D 轮廓图
        
    except Exception as e:
        print(f"[!] 提取轮廓失败，可能是 alpha_value 设置得太大导致计算崩溃: {e}")
        print("[!] 请尝试将 alpha_value 调小（比如 0.5 或 1.0）后重试。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 替换成您的单体测试楼的 ply 路径
    parser.add_argument("--ply", type=str, default="single_building.ply", help="单体建筑的 ply 文件路径")
    args = parser.parse_args()
    
    process_building(args.ply)