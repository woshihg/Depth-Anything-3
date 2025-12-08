import torch
import glob
import os
from depth_anything_3.api import DepthAnything3


def generate_3dgs_from_images(image_folder, output_dir, model_name="depth-anything/DA3-GIANT", process_res = 504, colmap_path=None, split=True):
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

    if split:
        image_folder = os.path.join(image_folder, "images", "train")
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    image_paths.sort()

    print(f"Found {len(image_paths)} images. Starting inference to generate 3DGS...")

    from depth_anything_3.utils.colmap_loader import load_colmap_data
    # 加载训练数据用于生成GS模型
    intrinsics, extrinsics = load_colmap_data(os.path.join(colmap_path), split='train')

    # 加载测试数据用于渲染
    test_intrinsics, test_extrinsics = load_colmap_data(os.path.join(colmap_path), split='test')


    prediction = model.inference(
        image=image_paths,
        export_dir= output_dir,
        process_res = process_res,
        export_format="npz-glb-gs_ply-gs_video",
        align_to_input_ext_scale=True,
        infer_gs=True,  # Required for gs_ply and gs_video exports
        extrinsics = extrinsics,
        intrinsics = intrinsics,
        render_exts=test_extrinsics,
        render_ixts=test_intrinsics,
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

    if colmap_path is not None:
        from depth_anything_3.utils.colmap_save import save_colmap_data
        save_colmap_data(colmap_path, prediction.extrinsics )
        print(f"Updated COLMAP extrinsics saved to '{colmap_path}/images.bin'.")

     # 如果有深度图，打印其形状
    if prediction.depth is not None:
        print(f"Generated depth maps with shape: {prediction.depth.shape}")


if __name__ == '__main__':
    # --- 配置 ---
    # 假设您在 'assets/examples/SOH' 文件夹中有一系列图像
    # 您可以将其更改为您自己的图像文件夹路径
    # git clone https://github.com/woshihg/Depth-Anything-3.git
    # image_folder_path = "Depth-Anything-3/assets/examples/SOH"
    data_folder = r"/home/woshihg/PycharmProjects/Depth-Anything-3/data/mydata/images_undistorted"
    image_folder_path = data_folder  # <--- 在这里更改为您的图像文件夹路径
    output_folder_path = "output/mydata_with_trinsics"
    process_res = 1080
    colmap_path = data_folder
    generate_3dgs_from_images(image_folder_path, output_folder_path, process_res = process_res, colmap_path=colmap_path)