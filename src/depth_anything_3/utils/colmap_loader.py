import os
import struct
import numpy as np
import collections
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. COLMAP 二进制文件读取工具 (内置，无需外部依赖)
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
    # 其他模型暂时省略，通常这几个最常用
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

            # 跳过 2D points 数据，因为这里只需要位姿
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24)  # x,y(16 bytes) + id(8 bytes)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None)
    return images


# ==========================================
# 2. 核心转换逻辑
# ==========================================

def get_intrinsic_matrix(camera):
    """
    将 COLMAP 的参数转换为 3x3 K 矩阵
    """
    params = camera.params
    K = np.eye(3)

    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy

    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

    elif camera.model in ["SIMPLE_RADIAL", "RADIAL"]:
        # 对于带畸变的模型，如果你只需要 K 矩阵，通常取前三个参数即可 (f, cx, cy)
        # 注意：这里忽略了畸变系数 (k1, k2...)
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy

    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

    else:
        raise NotImplementedError(f"Camera model {camera.model} conversion not implemented.")

    return K


def load_colmap_data(colmap_dir):
    """
    读取 COLMAP 数据并返回 (intrinsics, extrinsics)

    Returns:
        intrinsics: (N, 3, 3)
        extrinsics: (N, 4, 4) - World to Camera
        image_names: List[str] - 用于核对排序是否正确
    """
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    print("Loading COLMAP data...")
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)

    # *** 关键步骤 ***
    # COLMAP 的 image_id 是乱序的，必须按照 image_name 排序
    # 这样才能保证 N 维数组和你的图片文件列表一一对应
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)

    N = len(sorted_image_ids)

    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    for idx, img_id in enumerate(sorted_image_ids):
        img_data = images[img_id]
        cam_data = cameras[img_data.camera_id]

        # 1. 处理 Extrinsics (World-to-Camera)
        # COLMAP 存储的是 quaternion (w, x, y, z) 和 translation
        # 公式: P_cam = R * P_world + t
        r = R.from_quat([img_data.qvec[1], img_data.qvec[2], img_data.qvec[3], img_data.qvec[0]])  # scipy 顺序是 x,y,z,w
        rot_mat = r.as_matrix()

        extrinsics_np[idx, :3, :3] = rot_mat
        extrinsics_np[idx, :3, 3] = img_data.tvec
        extrinsics_np[idx, 3, 3] = 1.0

        # 2. 处理 Intrinsics
        K = get_intrinsic_matrix(cam_data)
        intrinsics_np[idx] = K

        image_names.append(img_data.name)

    return intrinsics_np, extrinsics_np, image_names


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    # 指向包含 cameras.bin 和 images.bin 的文件夹 (通常是 sparse/0/)
    colmap_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/dslr-undistorted/sparse/0"

    try:
        intrinsics, extrinsics, filenames = load_colmap_data(colmap_path)

        print(f"Loaded {len(filenames)} images.")
        print(f"Intrinsics shape: {intrinsics.shape}")  # (N, 3, 3)
        print(f"Extrinsics shape: {extrinsics.shape}")  # (N, 4, 4)
        print("First image name:", filenames[0])

        # 你的后续逻辑...
        # model(intrinsics=intrinsics, extrinsics=extrinsics)

    except Exception as e:
        print(f"Error: {e}")