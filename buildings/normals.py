import numpy as np
from plyfile import PlyData
import open3d as o3d

def visualize_gaussian_normal_fixed(ply_path):
    # ===================== 1. 安全读取二进制PLY文件 =====================
    print(f"正在读取文件: {ply_path}")
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        print(f"成功读取顶点数: {len(vertex)}")
    except Exception as e:
        print(f"PLY文件读取失败: {e}")
        return

    # ===================== 2. 提取核心数据，做合法性校验 =====================
    # 提取坐标
    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    print(f"坐标范围: X[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] | Y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] | Z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    
    # 提取并处理法向量（核心修复：归一化，避免光照错误）
    try:
        normals = np.vstack([vertex['nx'], vertex['ny'], vertex['nz']]).T
        # 法向量归一化（光照计算的必须步骤）
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        print(f"法向量读取成功，已归一化")
    except Exception as e:
        print(f"法向量读取失败: {e}")
        normals = np.zeros_like(xyz)

    # ===================== 3. 核心修复：正确读取3D高斯的颜色 =====================
    # 优先从球谐DC分量读取颜色（3D高斯的标准颜色存储）
    try:
        # f_dc是线性空间颜色，转换到0-1的sRGB范围
        dc_r = vertex['f_dc_0']
        dc_g = vertex['f_dc_1']
        dc_b = vertex['f_dc_2']
        colors = np.vstack([dc_r, dc_g, dc_b]).T
        colors = np.clip(colors + 0.5, 0.0, 1.0)  # 3D高斯DC分量的标准偏移处理
        print("成功从球谐DC分量读取颜色")
    except Exception as e:
        print(f"DC分量颜色读取失败，尝试读取RGB通道: {e}")
        # 兜底：读取red/green/blue通道
        if 'red' in vertex.data.dtype.names:
            colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0
        else:
            colors = np.ones_like(xyz) * 0.6  # 兜底灰色，彻底避免全黑

    # ===================== 4. 创建Open3D点云对象 =====================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # ===================== 5. 法向量可视化（优化采样，避免遮挡） =====================
    arrow_list = []
    sample_rate = 200  # 数值越大，箭头越少，避免画面混乱
    arrow_length = 0.3  # 箭头长度，可根据你的点云坐标范围调整
    
    # 法向量方向彩色映射：X=红, Y=绿, Z=蓝，直观判断方向
    normal_color = np.abs(normals)

    for i in range(0, len(xyz), sample_rate):
        start_pt = xyz[i]
        end_pt = start_pt + normals[i] * arrow_length
        
        # 创建法向量线段
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start_pt, end_pt])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([normal_color[i]])
        arrow_list.append(line)

    # ===================== 6. 可视化，核心渲染参数修复 =====================
    print("\n===== 可视化窗口已打开 =====")
    print("操作说明：左键旋转 | 滚轮缩放 | 右键平移")
    print("法向量颜色：偏红=X轴 | 偏绿=Y轴 | 偏蓝=Z轴")
    print("如果法向量全部朝内，添加 normals = -normals 即可统一翻转方向")

    # 自定义可视化窗口，精准控制渲染参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="高斯点云法向量可视化", width=1280, height=720)
    
    # 添加几何体
    vis.add_geometry(pcd)
    for arrow in arrow_list:
        vis.add_geometry(arrow)
    
    # 核心渲染修复
    opt = vis.get_render_option()
    opt.point_size = 2.0  # 调大点大小，避免过度绘制和噪点感
    opt.light_on = False  # 关闭光照，彻底规避法向量导致的全黑问题
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.show_coordinate_frame = True  # 显示世界坐标系，辅助判断方向
    
    # 自动适配相机视角，避免点云超出裁剪面
    vis.reset_view_point(True)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 这里改成你的PLY文件路径
    PLY_FILE_PATH = "/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings/instance_001.ply"
    visualize_gaussian_normal_fixed(PLY_FILE_PATH)