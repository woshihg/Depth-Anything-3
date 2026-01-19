import os


def rename_jpg_extension(directory_path):
    """
    将指定目录中所有文件的 .JPG 后缀批量重命名为 .jpg。

    Args:
        directory_path (str): 目标文件夹的路径。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：找不到目录 '{directory_path}'")
        return

    print(f"正在扫描目录: '{directory_path}'...")
    renamed_count = 0
    # 遍历目录中的所有文件和文件夹
    for filename in os.listdir(directory_path):
        # 检查文件名是否以 .JPG 结尾
        if filename.endswith(".JPG"):
            # 构建完整的文件路径
            old_filepath = os.path.join(directory_path, filename)

            # 只处理文件，跳过文件夹
            if not os.path.isfile(old_filepath):
                continue

            # 创建新的文件名（将 .JPG 替换为 .jpg）
            new_filename = filename[:-4] + ".jpg"
            new_filepath = os.path.join(directory_path, new_filename)

            # 执行重命名
            try:
                os.rename(old_filepath, new_filepath)
                print(f"已重命名: '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except OSError as e:
                print(f"重命名 '{filename}' 时出错: {e}")

    if renamed_count > 0:
        print(f"\n操作完成。共重命名了 {renamed_count} 个文件。")
    else:
        print("未找到需要重命名的 .JPG 文件。")


if __name__ == '__main__':
    # --- 配置 ---
    # 请将下面的路径更改为包含 .JPG 文件的文件夹路径
    target_folder = r"/home/woshihg/kicker_dslr_jpg/kicker/images/dslr_images"  # <--- 在这里修改为你的文件夹路径

    # 运行重命名函数
    rename_jpg_extension(target_folder)
