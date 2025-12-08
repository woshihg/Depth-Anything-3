import os
import struct
import numpy as np
import collections
import glob
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. COLMAP 二进制文件读取/写入工具
# ==========================================

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
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
            points_raw = fid.read(num_points2D * 24)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name, xys=None, point3D_ids=None,
                                     points_raw=points_raw, num_points2D=num_points2D)
    return images


def write_images_binary(images_dict, modified_extrinsics, sorted_image_ids, path_to_output_file):
    """
    将修改后的外参写回 images.bin 格式。
    注意：这里只会写入 sorted_image_ids 中存在的图片。
    """
    print(f"Saving {len(sorted_image_ids)} filtered images to {path_to_output_file}...")
    with open(path_to_output_file, "wb") as fid:
        # 1. 写入过滤后的图片总数
        fid.write(struct.pack("<Q", len(sorted_image_ids)))

        for idx, img_id in enumerate(sorted_image_ids):
            img_data = images_dict[img_id]
            mod_matrix = modified_extrinsics[idx]

            # 矩阵转四元数 + 平移
            R_mat = mod_matrix[:3, :3]
            tvec = mod_matrix[:3, 3]
            r = R.from_matrix(R_mat)
            q_scipy = r.as_quat()  # x, y, z, w
            qvec = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # w, x, y, z

            fid.write(struct.pack("<i", img_data.id))
            fid.write(struct.pack("<dddd", *qvec))
            fid.write(struct.pack("<ddd", *tvec))
            fid.write(struct.pack("<i", img_data.camera_id))
            fid.write(img_data.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", img_data.num_points2D))
            fid.write(img_data.points_raw)

    print("Save complete.")


# ==========================================
# 2. 核心逻辑 (含过滤与重定位)
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


def load_and_filter_colmap_data(colmap_dir, image_folder, process_res=-1):
    """
    读取 COLMAP 数据，并根据 image_folder 中的文件进行过滤。
    只保留 image_folder 中存在的图片。
    然后执行重定位 (Recenter) 和 缩放。
    """
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    # 1. 获取图片文件夹中的所有有效图片名
    valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    # 获取文件名集合 (例如 {'00001.jpg', '00002.png'})
    valid_image_names = set()
    for f in os.listdir(image_folder):
        if os.path.splitext(f)[1] in valid_extensions:
            valid_image_names.add(f)

    print(f"Found {len(valid_image_names)} images in folder: {image_folder}")

    # 2. 读取 COLMAP 数据
    print("Loading COLMAP data...")
    cameras = read_cameras_binary(cameras_file)
    images_raw = read_images_binary(images_file)

    print(f"Original COLMAP has {len(images_raw)} images.")

    # 3. 过滤逻辑：只保留名字在 valid_image_names 中的 image_id
    filtered_image_ids = []
    skipped_count = 0

    # 先获取所有 ID 并按名字排序
    all_sorted_ids = sorted(images_raw.keys(), key=lambda k: images_raw[k].name)

    for img_id in all_sorted_ids:
        img_name = images_raw[img_id].name
        # 检查是否在文件夹中 (尝试直接匹配，如果 colmap 有子文件夹路径则取 basename)
        basename = os.path.basename(img_name)

        if basename in valid_image_names or img_name in valid_image_names:
            filtered_image_ids.append(img_id)
        else:
            skipped_count += 1

    print(f"Filtered down to {len(filtered_image_ids)} images. (Removed {skipped_count} images not in folder)")

    if len(filtered_image_ids) == 0:
        raise ValueError("No matching images found! Check your folder path and file extensions.")

    # 4. 构建数据
    N = len(filtered_image_ids)
    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    print(f"Applying resize logic with process_res = {process_res}")

    for idx, img_id in enumerate(filtered_image_ids):
        img_data = images_raw[img_id]
        cam_data = cameras[img_data.camera_id]

        # Extrinsics
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

    # 5. 重定位 (Recenter)
    if N > 0:
        print(f"Re-centering scene. Origin set to first camera: {image_names[0]}")
        first_extrinsic = extrinsics_np[0]
        first_pose_inv = np.linalg.inv(first_extrinsic)
        extrinsics_np = extrinsics_np @ first_pose_inv

    # 返回过滤后的 id 列表，以便写入时使用
    return intrinsics_np, extrinsics_np, image_names, images_raw, filtered_image_ids

def load_colmap_data(colmap_dir, split=None):
    """
    读取 COLMAP 数据。

    Args:
        colmap_dir: 包含 cameras.bin 和 images.bin 的文件夹路径
        split (bool): 如果为 True，则只加载 images/train/ 目录下的图像数据。

    Returns:
        intrinsics: (N, 3, 3)
        extrinsics: (N, 4, 4) World-to-Camera Matrix (原始 COLMAP 坐标系)
        image_names: List[str] 按文件名排序
    """
    cameras_file = os.path.join(colmap_dir, "sparse", "0", "cameras.bin")
    images_file = os.path.join(colmap_dir, "sparse", "0", "images.bin")

    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        raise FileNotFoundError(f"COLMAP binary files not found in {colmap_dir}")

    print(f"Loading COLMAP data from {colmap_dir}...")
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)

    # 1. 排序：按文件名 A-Z 排序 (确保与 DataLoader 一致)
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k].name)

    # 2. 如果需要，根据 train split 过滤
    if split:
        # 假设 colmap_dir 是 '.../sparse/0'，我们需要找到 '.../images/train'
        base_dir = os.path.abspath(os.path.join(colmap_dir, "..", ".."))
        train_dir = os.path.join(base_dir, "images", split)
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        train_images_set = set(os.listdir(train_dir))

        original_count = len(sorted_image_ids)
        sorted_image_ids = [
            img_id for img_id in sorted_image_ids
            if os.path.basename(images[img_id].name) in train_images_set
        ]
        print(f"Split mode: Filtered from {original_count} to {len(sorted_image_ids)} images based on '{train_dir}'.")

    N = len(sorted_image_ids)
    intrinsics_np = np.zeros((N, 3, 3), dtype=np.float32)
    extrinsics_np = np.zeros((N, 4, 4), dtype=np.float32)
    image_names = []

    print(f"Processing {N} images.")

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
        intrinsics_np[idx] = K
        image_names.append(img_data.name)

    # 已移除 Recenter 逻辑，直接返回
    return intrinsics_np.astype(np.float32), extrinsics_np.astype(np.float32)

# ==========================================
# 在处理新的数据之前需要运行该脚本将修改后的外参写回 COLMAP 二进制文件
# ==========================================

if __name__ == "__main__":
    # COLMAP sparse 文件夹路径
    colmap_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/gsnet/sparse/0"

    # 你的图片文件夹路径 (用作过滤器)
    image_folder_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/gsnet"

    target_resolution = -1

    # 输出文件保存位置 (覆盖原文件)
    output_bin_path = os.path.join(colmap_path, "images.bin")

    try:
        # 1. 加载、过滤、处理
        intrinsics, extrinsics, filenames, raw_images_dict, sorted_ids = load_and_filter_colmap_data(
            colmap_path,
            image_folder_path,
            process_res=target_resolution
        )

        print(f"Final dataset contains {len(filenames)} images.")

        # 验证第一张图是否归零
        is_identity = np.allclose(extrinsics[0], np.eye(4), atol=1e-6)
        print(f"Verification: First camera is at origin? {is_identity}")

        # 2. 将结果写回 images.bin
        # 注意：这里只会写入 filtered_ids (sorted_ids) 中的图片，从而删除了多余的图片
        write_images_binary(raw_images_dict, extrinsics, sorted_ids, output_bin_path)

        print(f"\n[Success] Filtered and Recentered images.bin saved to: {output_bin_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()