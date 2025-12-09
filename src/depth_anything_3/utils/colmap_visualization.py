import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from colmap_change import read_images_binary
import os

import open3d as o3d
import numpy as np
import os
import struct


# ==========================================
# 1. 必要的 COLMAP 读取函数 (复用之前的代码)
# ==========================================
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = fid.read(64)
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack(
                "<idddddddi", binary_image_properties)
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            # 跳过点数据读取，加快速度
            fid.seek(num_points2D * (16 + 8), 1)
            images[image_id] = Image(image_id, np.array([qw, qx, qy, qz]),
                                     np.array([tx, ty, tz]), camera_id, image_name, None, None)
    return images


# ==========================================
# 2. Open3D 可视化核心逻辑
# ==========================================

def get_camera_frustum(img, color=[0, 1, 0], scale=1.0):
    """
    创建一个代表相机的线框金字塔 (Frustum)
    """
    R = qvec2rotmat(img.qvec)
    t = img.tvec

    # 1. 计算相机中心 (C = -R^T * t)
    center = -np.dot(R.T, t)

    # 2. 定义局部坐标系下的金字塔形状 (假设看向 Z 轴正向或负向)
    # COLMAP/OpenCV: +Z 是前方, +X 右, +Y 下
    w, h = scale, scale * 0.75
    z = scale * 1.5  # 视锥体的深度

    # 局部坐标系下的 5 个顶点：[中心, 左上, 右上, 右下, 左下]
    local_points = np.array([
        [0, 0, 0],
        [-w, -h, z],
        [w, -h, z],
        [w, h, z],
        [-w, h, z]
    ])

    # 3. 将局部顶点转换到世界坐标系
    # P_world = R^T * P_local + Center
    # 这里的 R^T 是从相机转世界的旋转
    world_points = np.dot(R.T, local_points.T).T + center

    # 4. 定义连接线 (绘制金字塔的棱)
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从中心射出的四条棱
        [1, 2], [2, 3], [3, 4], [4, 1]  # 底面的四条边
    ]

    # 5. 创建 Open3D LineSet 对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    return line_set


def visualize_colmap_open3d(colmap_path, scale_factor=0.5):
    images_bin_path = os.path.join(colmap_path, "images.bin")
    if not os.path.exists(images_bin_path):
        print(f"找不到 {images_bin_path}")
        return

    print("读取数据中...")
    images = read_images_binary(images_bin_path)

    geometries = []

    # 添加一个坐标轴原点 (RGB = XYZ)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(axis)

    sorted_ids = sorted(images.keys(), key=lambda k: images[k].name)

    # 自动计算一个合适的显示比例
    # 采样前几个相机计算平均距离，用于确定金字塔画多大
    # if len(sorted_ids) > 1:
    #     c1 = -np.dot(qvec2rotmat(images[sorted_ids[0]].qvec).T, images[sorted_ids[0]].tvec)
    #     c2 = -np.dot(qvec2rotmat(images[sorted_ids[1]].qvec).T, images[sorted_ids[1]].tvec)
    #     dist = np.linalg.norm(c1 - c2)
    #     if dist > 0:
    #         scale_factor = dist * 0.5  # 金字塔大小设为相机间距的一半
    #         print(f"自动调整相机图标大小为: {scale_factor:.2f}")
    print(f"相机图标大小设置为: {scale_factor:.2f} 米")

    print("生成几何体...")
    for i, img_id in enumerate(sorted_ids):
        img = images[img_id]

        # 第一张图用红色，其他的用绿色
        if i == 0:
            color = [1, 0, 0]  # Red
        else:
            color = [0, 1, 0]  # Green

        frustum = get_camera_frustum(img, color=color, scale=scale_factor)
        geometries.append(frustum)

    print("打开可视化窗口... (按 'H' 查看帮助，左键旋转，右键平移，滚轮缩放)")
    o3d.visualization.draw_geometries(geometries, window_name="COLMAP Visualization")


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 替换成你的 sparse 路径
    path = r"/home/woshihg/PycharmProjects/Depth-Anything-3/data/mydata/images_undistorted/sparse/1"
    visualize_colmap_open3d(path)