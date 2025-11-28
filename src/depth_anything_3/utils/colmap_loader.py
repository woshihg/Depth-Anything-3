import os
import struct
import numpy as np
import collections
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. COLMAP 二进制文件读取工具 (保持不变)
# ==========================================

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

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
            fid.read(num_points2D * 24)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None)
    return images


# ==========================================
# 2. 核心转换逻辑
# ==========================================

def get_intrinsic_matrix(camera):
    params = camera.params
    K = np.eye(3)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f; K[1, 1] = f; K[0, 2] = cx; K[1, 2] = cy
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
    elif camera.model in ["SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f; K[1, 1] = f; K[0, 2] = cx; K[1, 2] = cy
    elif camera.model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
    else:
        raise NotImplementedError(f"Camera model {camera.model} conversion not implemented.")
    return K


def load_colmap_data(colmap_dir, process_res=-1):
    """
    读取 COLMAP 数据并返回 (intrinsics, extrinsics)
    并【将第一张图片的相机位置设为世界坐标原点】
    """
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    print("Loading COLMAP data...")
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)

    # 按文件名排序 (确保第0个总是同一张图)
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)

    N = len(sorted_image_ids)

    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    print(f"Applying resize logic with process_res = {process_res}")

    # 1. 正常读取所有数据
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

    # 2. 【新增逻辑】将第一张图设为参考系原点
    if N > 0:
        print(f"Re-centering scene. Origin set to first camera: {image_names[0]}")

        # 获取第一张图的外参 (W2C)
        first_extrinsic = extrinsics_np[0]  # Shape (4, 4)

        # 计算其逆矩阵 (C2W)
        # 这个逆矩阵代表：从 Camera 0 到 原World 的变换
        first_pose_inv = np.linalg.inv(first_extrinsic)

        # 将所有外参乘以这个逆矩阵
        # 公式: E_new = E_old @ E_0_inv
        # 解释: 先把点从 NewWorld(Cam0) 变回 OldWorld，再变到 TargetCamera
        extrinsics_np = extrinsics_np @ first_pose_inv

        # 验证: 第一张图的矩阵应该是单位矩阵 (Identity)
        # print("Debug: First Extrinsic after recenter:\n", extrinsics_np[0])

    return intrinsics_np.astype(np.float32), extrinsics_np.astype(np.float32), image_names

# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    colmap_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/dslr-undistorted/sparse/0"

    # 假设你模型的输入分辨率是 504 (通常是长边)
    target_resolution = 504

    try:
        intrinsics, extrinsics, filenames = load_colmap_data(colmap_path, process_res=target_resolution)

        print(f"Loaded {len(filenames)} images.")
        print(f"Intrinsics shape: {intrinsics.shape}")
        print(f"Extrinsics shape: {extrinsics.shape}")

    except Exception as e:
        print(f"Error: {e}")