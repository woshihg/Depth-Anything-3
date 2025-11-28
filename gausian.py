import torch
import glob
import os
from depth_anything_3.api import DepthAnything3


def generate_3dgs_from_images(image_folder, output_dir, model_name="depth-anything/DA3-GIANT", process_res = 504):
    """
    从图像文件夹生成 3D Gaussian Splatting (.glb) 文件。

    Args:
        image_folder (str): 包含输入图像序列的文件夹路径。
        output_dir (str): 保存输出文件的目录。
        model_name (str): 要使用的预训练模型名称。
    """
    # 检查 CUDA 是否可用，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Depth Anything 3 model...")
    # 从预训练权重加载模型
    model = DepthAnything3.from_pretrained(model_name).to(device)

    # 获取并排序图像文件列表
    # 1. 收集所有格式的路径
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]  # 建议加上大写支持
    image_paths = []

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    image_paths.sort()

    print(f"Found {len(image_paths)} images. Starting inference to generate 3DGS...")

    colmap_path = "/home/woshihg/PycharmProjects/Depth-Anything-3/data/dslr-undistorted/sparse/0"

    from depth_anything_3.utils.colmap_loader import load_colmap_data
    intrinsics, extrinsics = load_colmap_data(colmap_path, -1)

    prediction = model.inference(
        image=image_paths,
        export_dir= output_dir,
        process_res = process_res,
        export_format="npz-glb-gs_ply-gs_video",
        align_to_input_ext_scale=True,
        infer_gs=True,  # Required for gs_ply and gs_video exports
        extrinsics = extrinsics,
        intrinsics = intrinsics,
        export_kwargs={"save_sh_dc_only": False},
    )

    print("\nInference complete!")
    print(f"3DGS output has been saved to '{output_dir}'.")
    print("You should find a '.glb' file which can be viewed in a 3D viewer (e.g., Windows 3D Viewer, Blender).")
    print("You will also find:")
    print("- A '.ply' file containing the raw Gaussian Splatting data.")
    print("- A 'depth' subfolder with individual depth maps.")

    # 打印一些返回的预测信息
    if prediction.extrinsics is not None:
        print(f"\nEstimated extrinsics for {prediction.extrinsics.shape[0]} images.")
    if prediction.depth is not None:
        print(f"Generated depth maps with shape: {prediction.depth.shape}")


if __name__ == '__main__':
    # --- 配置 ---
    # 假设您在 'assets/examples/SOH' 文件夹中有一系列图像
    # 您可以将其更改为您自己的图像文件夹路径
    # git clone https://github.com/woshihg/Depth-Anything-3.git
    # image_folder_path = "Depth-Anything-3/assets/examples/SOH"

    image_folder_path = r"/home/woshihg/PycharmProjects/Depth-Anything-3/data/dslr-undistorted"  # <--- 在这里更改为您的图像文件夹路径
    output_folder_path = "output/my_3dgs_scene"
    process_res = 720
    # 检查示例文件夹是否存在
    if not os.path.isdir(image_folder_path) or image_folder_path == "path/to/your/image/folder":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请将 `image_folder_path` 更改为包含您的图像序列的文件夹。 !!!")
        print("!!! 例如：'my_video_frames'                                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        generate_3dgs_from_images(image_folder_path, output_folder_path, process_res = process_res)