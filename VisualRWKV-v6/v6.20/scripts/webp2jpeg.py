import os
import sys
from PIL import Image
from pathlib import Path

# 设置文件夹路径
folder_path = Path(sys.argv[1]) # 假设里面全部都是图片
filenames = [f for f in folder_path.rglob("*") if f.is_file()]

# 遍历文件夹中的所有文件
for file_path in filenames:
    try:
        # 尝试打开文件
        with Image.open(file_path) as img:
            # 检查图片格式是否为 WebP
            if img.format == "WEBP":
                # 去除原始扩展名，添加 .jpg 扩展名
                jpg_filename = file_path.stem + ".jpg"
                jpg_path = file_path.with_name(jpg_filename)
                # 将图片转换为 RGB 模式并保存为 .jpg
                img.convert("RGB").save(jpg_path, "JPEG")
                # 删除原始文件
                if str(file_path) != str(jpg_path):
                    file_path.unlink()
                print(f"{file_path.name} WebP转换并替换为JPEG {jpg_filename}")
    except (OSError, IOError):
        print(f"{file_path} 无法打开或读取")

