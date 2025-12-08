import os
import struct
import numpy as np
import collections
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 基础定义与二进制读写工具
# ==========================================

# 定义 Image 结构，必须包含 points_raw 以保留特征点
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "points_raw",
                                         "num_points2D"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            # 读取并保留原始特征点数据
            points_raw = fid.read(num_points2D * 24)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None,
                                     points_raw=points_raw, num_points2D=num_points2D)
    return images


def write_images_binary(images_dict, sorted_image_ids, new_poses, output_path):
    """
    将更新后的位姿写入 images.bin
    """
    with open(output_path, "wb") as fid:
        # 写入图片数量
        fid.write(struct.pack("<Q", len(sorted_image_ids)))

        for idx, img_id in enumerate(sorted_image_ids):
            img_data = images_dict[img_id]

            # 获取对应的 3x4 矩阵
            pose = new_poses[idx]  # Shape (3, 4)

            # --- 转换逻辑 ---
            # 1. 提取旋转矩阵 (3x3)
            rotation_matrix = pose[:3, :3]
            # 2. 提取平移向量 (3,)
            tvec = pose[:3, 3]

            # 3. 旋转矩阵 -> 四元数
            # Scipy 返回的是 [x, y, z, w]
            r = R.from_matrix(rotation_matrix)
            q_scipy = r.as_quat()
            # COLMAP 需要 [w, x, y, z]
            qvec = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            # --- 写入数据 ---
            fid.write(struct.pack("<i", img_data.id))
            fid.write(struct.pack("<dddd", *qvec))  # 更新后的 qvec
            fid.write(struct.pack("<ddd", *tvec))  # 更新后的 tvec
            fid.write(struct.pack("<i", img_data.camera_id))
            fid.write(img_data.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", img_data.num_points2D))
            fid.write(img_data.points_raw)  # 原始特征点


# ==========================================
# 2. 核心函数 save_colmap_data
# ==========================================

def save_colmap_data(colmap_dir, new_poses, save_path=None):
    """
    读取 COLMAP 原始数据，用输入的矩阵列表替换外参，并保存。

    Args:
        colmap_dir (str): 包含 images.bin 的原始目录 (sparse/0)
        new_poses (np.ndarray or torch.Tensor):
            形状为 (N, 3, 4) 或 (N, 4, 4) 的 World-to-Camera 矩阵列表。
            如果是 (4,4) 会自动截取前3行。
            **必须按文件名的字母顺序排序**，以对应 COLMAP 数据。
        save_path (str, optional): 输出文件路径。如果不填，默认覆盖原文件。
    """
    images_file = os.path.join(colmap_dir, "images.bin")
    if not os.path.exists(images_file):
        raise FileNotFoundError(f"images.bin not found in {colmap_dir}")

    # 1. 转换输入数据格式 (支持 Tensor 转 Numpy)
    if hasattr(new_poses, 'cpu'):
        new_poses = new_poses.detach().cpu().numpy()
    if isinstance(new_poses, list):
        new_poses = np.array(new_poses)

    # 确保是 (N, 3, 4)
    if new_poses.shape[1:] == (4, 4):
        new_poses = new_poses[:, :3, :]  # 截取前3行

    if new_poses.ndim != 3 or new_poses.shape[1:] != (3, 4):
        raise ValueError(f"Input poses shape error. Expected (N, 3, 4), got {new_poses.shape}")

    # 2. 读取原始数据
    print(f"Reading original data from {images_file}...")
    images_dict = read_images_binary(images_file)

    # 3. 排序 (确保 N 维矩阵与 COLMAP 图片一一对应)
    sorted_image_ids = sorted(images_dict.keys(), key=lambda k: images_dict[k].name)

    # 检查数量是否一致
    if len(sorted_image_ids) != len(new_poses):
        raise ValueError(f"Mismatch! COLMAP has {len(sorted_image_ids)} images, but input poses has {len(new_poses)}.")

    # 4. 确定保存路径
    if save_path is None:
        save_path = images_file  # 默认覆盖
        print("Warning: Overwriting original images.bin")

    # 5. 写入
    print(f"Saving updated extrinsics to {save_path}...")
    write_images_binary(images_dict, sorted_image_ids, new_poses, save_path)
    print("Done.")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    colmap_path = "/path/to/your/sparse/0"

    # 模拟生成一些 3x4 矩阵 (截去最后一行)
    # 假设有 100 张图
    # N = 100
    # dummy_poses = np.zeros((N, 3, 4))
    # dummy_poses[:, :3, :3] = np.eye(3) # 旋转单位阵
    # dummy_poses[:, :3, 3] = np.array([0, 0, 5]) # 平移 z=5

    # 调用函数
    # save_colmap_data(colmap_path, dummy_poses, save_path=os.path.join(colmap_path, "images_updated.bin"))