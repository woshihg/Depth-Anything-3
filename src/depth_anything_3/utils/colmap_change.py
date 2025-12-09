import os
import struct
import numpy as np


# ==========================================
# Part 1: 数学工具与 COLMAP IO 辅助函数
# ==========================================

def qvec2rotmat(qvec):
    """四元数转旋转矩阵"""
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


def rotmat2qvec(R):
    """旋转矩阵转四元数"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if qvec[0] < 0: qvec *= -1
    return qvec


class Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


class Point3D:
    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs


# ==========================================
# 修复后的读取函数 (添加了 '<')
# ==========================================

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        # 使用 <Q 强制小端序无填充
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = fid.read(64)
            # 关键修改： "<idddddddi"
            # < : 小端序, 无填充
            # i : 4 bytes
            # d * 7 : 56 bytes
            # i : 4 bytes
            # 总计 : 64 bytes (与 fid.read(64) 匹配)
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack(
                "<idddddddi", binary_image_properties)

            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]

            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            data = np.fromfile(fid, dtype=np.float64, count=num_points2D * 2).reshape((-1, 2))
            point3D_ids = np.fromfile(fid, dtype=np.uint64, count=num_points2D)
            images[image_id] = Image(image_id, np.array([qw, qx, qy, qz]),
                                     np.array([tx, ty, tz]), camera_id, image_name, data, point3D_ids)
    return images


def write_images_binary(images, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for _, img in images.items():
            img_char = img.name.encode('utf-8') + b'\x00'
            # 同样写入时也要加 <
            fid.write(struct.pack("<idddddddi", img.id, *img.qvec, *img.tvec, img.camera_id))
            fid.write(img_char)
            fid.write(struct.pack("<Q", len(img.point3D_ids)))
            img.xys.tofile(fid)
            img.point3D_ids.tofile(fid)


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_properties = fid.read(43)
            # 关键修改： "<QdddBBBd"
            point3D_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", binary_point_properties)

            track_len = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_len, fid.read(8 * track_len))
            image_ids = np.array(tuple(track_elems[0::2]))
            point2D_idxs = np.array(tuple(track_elems[1::2]))
            points3D[point3D_id] = Point3D(point3D_id, np.array([x, y, z]),
                                           np.array([r, g, b]), error, image_ids, point2D_idxs)
    return points3D


def write_points3D_binary(points3D, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for _, pt in points3D.items():
            # 写入时也要加 <
            fid.write(struct.pack("<QdddBBBd", pt.id, *pt.xyz, *pt.rgb, pt.error))
            track_len = len(pt.image_ids)
            fid.write(struct.pack("<Q", track_len))
            track = np.zeros(track_len * 2, dtype=np.int32)
            track[0::2] = pt.image_ids
            track[1::2] = pt.point2D_idxs
            track.tofile(fid)


# ==========================================
# Part 2: 核心处理逻辑
# ==========================================

def recenter_colmap_model(input_path):
    """
    读取COLMAP数据，按图片名排序，将第一张图片设为原点，并保存回原路径。
    注意：这会覆盖原始文件，建议先备份。
    """
    images_bin_path = os.path.join(input_path, "images.bin")
    points_bin_path = os.path.join(input_path, "points3D.bin")

    if not os.path.exists(images_bin_path) or not os.path.exists(points_bin_path):
        print(f"Error: {input_path} 下未找到 images.bin 或 points3D.bin")
        return

    print("正在读取数据...")
    images = read_images_binary(images_bin_path)
    points3D = read_points3D_binary(points_bin_path)

    # 1. 按照图片名称排序
    sorted_img_ids = sorted(images.keys(), key=lambda k: images[k].name)
    first_img_id = sorted_img_ids[0]
    ref_img = images[first_img_id]

    print(f"参考图片 (原点): {ref_img.name}")

    # 打印所有图片变换前的外参矩阵
    print("\n--- 原始外参 (World-to-Camera) ---")
    for img_id in sorted_img_ids:
        img = images[img_id]
        R_before = qvec2rotmat(img.qvec)
        T_before = img.tvec
        ext_w2c_before = np.eye(4)
        ext_w2c_before[:3, :3] = R_before
        ext_w2c_before[:3, 3] = T_before.flatten()
        print(f"Image: {img.name}")
        print(ext_w2c_before)
    print("------------------------------------------")

    # 2. 获取参考图片的变换矩阵 (World -> Camera Ref)
    # P_ref = R_ref * P_world + T_ref
    # 我们要将 P_world 变换到 P_new_world，且让 P_new_world = P_ref
    # 也就是说，我们把参考图片的相机坐标系当作新的世界坐标系。

    R_ref = qvec2rotmat(ref_img.qvec)
    T_ref = ref_img.tvec

    # 3. 变换 Points3D
    # 旧世界坐标 P_w -> 新世界坐标 P_new (即 P_ref)
    # P_new = R_ref * P_w + T_ref
    print("正在转换 3D 点云...")
    for pt_id in points3D:
        pt = points3D[pt_id]
        # 应用变换
        pt.xyz = np.dot(R_ref, pt.xyz) + T_ref

    # 4. 变换 Cameras (Images)
    # 原公式: P_c = R_i * P_w + T_i
    # 逆变换: P_w = R_ref.T * (P_new - T_ref)
    # 代入原公式: P_c = R_i * (R_ref.T * P_new - R_ref.T * T_ref) + T_i
    #             P_c = (R_i * R_ref.T) * P_new + (T_i - R_i * R_ref.T * T_ref)
    # 所以:
    # R_new = R_i * R_ref.T
    # T_new = T_i - R_new * T_ref

    print("正在转换相机外参...")
    R_ref_T = R_ref.T

    for img_id in images:
        img = images[img_id]

        R_i = qvec2rotmat(img.qvec)
        T_i = img.tvec

        # 计算新旋转
        R_new = np.dot(R_i, R_ref_T)
        # 计算新平移
        T_new = T_i - np.dot(R_new, T_ref)

        # 更新数据
        img.qvec = rotmat2qvec(R_new)
        img.tvec = T_new

    # 打印所有图片变换后的外参矩阵
    print("\n--- 变换后外参 (新世界坐标系) ---")
    for img_id in sorted_img_ids:
        img_after = images[img_id]
        R_after = qvec2rotmat(img_after.qvec)
        T_after = img_after.tvec
        ext_w2c_after = np.eye(4)
        ext_w2c_after[:3, :3] = R_after
        ext_w2c_after[:3, 3] = T_after.flatten()
        print(f"Image: {img_after.name}")
        print(ext_w2c_after)
    print("------------------------------------------\n")

    # 5. 写回数据
    print("正在保存数据...")
    # 为了安全起见，这里可以改为写入一个新的文件夹，这里根据你的要求写回原位
    # 建议先备份原始文件！
    write_images_binary(images, images_bin_path)
    write_points3D_binary(points3D, points_bin_path)

    print("处理完成！第一张图片现在位于原点 (Identity Pose)。")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 修改为你的 sparse 文件夹路径 (包含 images.bin, points3D.bin 的文件夹)
    colmap_sparse_path = r"/home/woshihg/PycharmProjects/Depth-Anything-3/data/mydata/images_undistorted/sparse/1"

    # 运行函数
    recenter_colmap_model(colmap_sparse_path)