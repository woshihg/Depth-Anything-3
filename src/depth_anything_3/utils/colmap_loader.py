import os
import struct
import numpy as np
import collections
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. COLMAP 二进制文件读取/写入工具
# ==========================================

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
# [修改]: 增加 points_raw 用于存储原始二进制特征点数据，以便原样写回
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "points_raw",
                                         "num_points2D"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=CAMERA_MODELS[model_id].model_name,
                                        width=width, height=height, params=np.array(params))
    return cameras


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

            # [修改]: 读取并保存原始特征点数据，而不是跳过
            # 每个点占 24 字节 (double x, double y, int64 id)
            points_raw = fid.read(num_points2D * 24)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None,
                                     points_raw=points_raw, num_points2D=num_points2D)
    return images


def write_images_binary(images, modified_extrinsics, sorted_image_ids, path_to_output_file):
    """
    将修改后的外参写回 images.bin 格式
    """
    print(f"Saving modified images to {path_to_output_file}...")
    with open(path_to_output_file, "wb") as fid:
        # 1. 写入图片总数 (uint64)
        fid.write(struct.pack("<Q", len(images)))

        # 2. 遍历每一张图片写入数据
        # 注意：必须按照 extrinsics 的顺序 (即文件名排序后的顺序) 来对应矩阵
        for idx, img_id in enumerate(sorted_image_ids):
            img_data = images[img_id]

            # 获取修改后的 4x4 矩阵
            mod_matrix = modified_extrinsics[idx]

            # --- 矩阵转 COLMAP 格式 (R, t) ---
            # 提取旋转矩阵 (3x3) 和 平移向量 (3,)
            R_mat = mod_matrix[:3, :3]
            tvec = mod_matrix[:3, 3]

            # 旋转矩阵 -> 四元数 (Scipy 输出是 [x, y, z, w])
            r = R.from_matrix(R_mat)
            q_scipy = r.as_quat()

            # COLMAP 需要 [w, x, y, z]，所以需要调整顺序
            qvec = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            # --- 写入二进制数据 ---
            # Image_ID (int32)
            fid.write(struct.pack("<i", img_data.id))

            # Qvec (4 doubles)
            fid.write(struct.pack("<dddd", *qvec))

            # Tvec (3 doubles)
            fid.write(struct.pack("<ddd", *tvec))

            # Camera_ID (int32)
            fid.write(struct.pack("<i", img_data.camera_id))

            # Name (string + null byte)
            fid.write(img_data.name.encode("utf-8") + b"\x00")

            # Number of 2D points (uint64)
            fid.write(struct.pack("<Q", img_data.num_points2D))

            # 2D Points Data (Raw bytes) - 原样写回
            fid.write(img_data.points_raw)

    print("Save complete.")


# ==========================================
# 2. 核心转换逻辑
# ==========================================

def get_intrinsic_matrix(camera):
    params = camera.params
    K = np.eye(3)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f;
        K[1, 1] = f;
        K[0, 2] = cx;
        K[1, 2] = cy
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx;
        K[1, 1] = fy;
        K[0, 2] = cx;
        K[1, 2] = cy
    elif camera.model in ["SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f;
        K[1, 1] = f;
        K[0, 2] = cx;
        K[1, 2] = cy
    elif camera.model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx;
        K[1, 1] = fy;
        K[0, 2] = cx;
        K[1, 2] = cy
    else:
        raise NotImplementedError(f"Camera model {camera.model} conversion not implemented.")
    return K


def load_and_modify_colmap_data(colmap_dir, process_res=-1):
    """
    读取 COLMAP 数据，返回修改后的矩阵，同时返回原始数据以便保存
    """
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    print("Loading COLMAP data...")
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)

    # 按文件名排序
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)

    N = len(sorted_image_ids)

    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    print(f"Applying resize logic with process_res = {process_res}")

    # 1. 读取并构建矩阵
    for idx, img_id in enumerate(sorted_image_ids):
        img_data = images[img_id]
        cam_data = cameras[img_data.camera_id]

        # Extrinsics (World-to-Camera)
        r = R.from_quat([img_data.qvec[1], img_data.qvec[2], img_data.qvec[3], img_data.qvec[0]])
        rot_mat = r.as_matrix()

        extrinsics_np[idx, :3, :3] = rot_mat
        extrinsics_np[idx, :3, 3] = img_data.tvec
        extrinsics_np[idx, 3, 3] = 1.0

        # Intrinsics
        K = get_intrinsic_matrix(cam_data)
        if process_res > 0:
            orig_w = cam_data.width
            orig_h = cam_data.height
            scale = process_res / max(orig_w, orig_h)
            K[0, 0] *= scale;
            K[1, 1] *= scale
            K[0, 2] *= scale;
            K[1, 2] *= scale

        intrinsics_np[idx] = K
        image_names.append(img_data.name)

    # 2. 重新定位场景 (Recenter)
    if N > 0:
        print(f"Re-centering scene. Origin set to first camera: {image_names[0]}")
        first_extrinsic = extrinsics_np[0]
        first_pose_inv = np.linalg.inv(first_extrinsic)
        extrinsics_np = extrinsics_np @ first_pose_inv

    # 返回所有需要的数据
    return intrinsics_np, extrinsics_np, image_names, images, sorted_image_ids


def load_colmap_data(colmap_dir, process_res=-1):
    """
    读取 COLMAP 数据。

    Args:
        colmap_dir: 包含 cameras.bin 和 images.bin 的文件夹路径
        process_res: 目标处理分辨率 (长边)。如果为 -1，则保持原始内参。

    Returns:
        intrinsics: (N, 3, 3)
        extrinsics: (N, 4, 4) World-to-Camera Matrix (原始 COLMAP 坐标系)
        image_names: List[str] 按文件名排序
    """
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    print(f"Loading COLMAP data from {colmap_dir}...")
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)

    # 1. 排序：按文件名 A-Z 排序 (确保与 DataLoader 一致)
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)
    N = len(sorted_image_ids)

    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    print(f"Processing {N} images. Resize Mode: {'Original' if process_res <= 0 else f'Scale to {process_res}'}")

    for idx, img_id in enumerate(sorted_image_ids):
        img_data = images[img_id]
        cam_data = cameras[img_data.camera_id]

        # --- Extrinsics (World-to-Camera) ---
        # 直接读取原始四元数和平移，不进行逆运算或重定位
        r = R.from_quat([img_data.qvec[1], img_data.qvec[2], img_data.qvec[3], img_data.qvec[0]])
        rot_mat = r.as_matrix()

        extrinsics_np[idx, :3, :3] = rot_mat
        extrinsics_np[idx, :3, 3] = img_data.tvec
        extrinsics_np[idx, 3, 3] = 1.0

        # --- Intrinsics ---
        K = get_intrinsic_matrix(cam_data)

        # 缩放逻辑
        if process_res > 0:
            orig_w = cam_data.width
            orig_h = cam_data.height
            scale = process_res / max(orig_w, orig_h)

            K[0, 0] *= scale  # fx
            K[1, 1] *= scale  # fy
            K[0, 2] *= scale  # cx
            K[1, 2] *= scale  # cy

        intrinsics_np[idx] = K
        image_names.append(img_data.name)

    # 已移除 Recenter 逻辑，直接返回
    return intrinsics_np.astype(np.float32), extrinsics_np.astype(np.float32)

# ==========================================
# 在处理新的数据之前需要运行该脚本将修改后的外参写回 COLMAP 二进制文件
# ==========================================

if __name__ == "__main__":
    colmap_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/dslr-undistorted/sparse/0"
    target_resolution = -1

    # 输出文件路径 (建议不要覆盖原文件，而是保存为新的)
    output_bin_path = os.path.join(colmap_path, "images.bin")

    try:
        # 1. 加载并处理数据
        intrinsics, extrinsics, filenames, raw_images_dict, sorted_ids = load_and_modify_colmap_data(
            colmap_path, process_res=target_resolution
        )

        print(f"Loaded {len(filenames)} images.")

        # 验证第一张图
        is_identity = np.allclose(extrinsics[0], np.eye(4), atol=1e-6)
        print(f"Verification: First camera is at origin? {is_identity}")

        # 2. 将修改后的外参写回二进制文件
        write_images_binary(raw_images_dict, extrinsics, sorted_ids, output_bin_path)

        print(f"\n[Success] Modified extrinsics saved to: {output_bin_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()