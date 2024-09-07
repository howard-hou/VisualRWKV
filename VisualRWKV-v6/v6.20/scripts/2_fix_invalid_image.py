import json
import sys
from PIL import Image
from pathlib import Path

def fix_suffix(not_exist_image):
    exist_image = {}
    for img_path in not_exist_image:
        jpg_path = img_path.with_suffix('.jpg')
        if jpg_path.exists():
            exist_image[img_path] = jpg_path
    still_not_exist_image = [f for f in not_exist_image if f not in exist_image]
    fixed_image_list = [f for f in exist_image.values()]
    return still_not_exist_image, fixed_image_list

def any2jpg(image_path_list):
    jpeg_image_list = []
    for file_path in image_path_list:
        img = Image.open(file_path)
        if img.format == "JPEG":
            jpeg_image_list.append(file_path)
        else:
            jpg_filename = file_path.stem + ".jpg"
            jpg_path = file_path.with_name(jpg_filename)
            img.convert("RGB").save(jpg_path, "JPEG")
            if str(file_path) != str(jpg_path):
                file_path.unlink()
            print(f"{file_path.name} 转换并替换为JPEG {jpg_filename}")
            jpeg_image_list.append(jpg_path)
    return jpeg_image_list



# 设置文件夹路径
input_json = sys.argv[1]
image_folder = Path(sys.argv[2])
output_file = input_json.replace('.json', '_fixed.json')
drop_file = input_json.replace('.json', '_drop.json')

data = json.load(open(input_json))

keep, drop = [], []
for i, line in enumerate(data):
    image_path_list = [image_folder / img for img in line['image']]
    not_exist_image = [f for f in image_path_list if not f.exists()]
    exist_image = [f for f in image_path_list if f.exists()]
    # 第一种情况：图片文件不存在，看看是不是后缀名错误
    if not_exist_image:
        still_not_exist_image, fixed_image_list = fix_suffix(not_exist_image)
        if still_not_exist_image:
            drop.append(line)
        if fixed_image_list and not still_not_exist_image:
            final_image_list = exist_image + fixed_image_list
            line['image'] = [str(f) for f in final_image_list]
            line.pop('invalid_images', None)
            keep.append(line)
    # 第二种情况：图片文件存在，但是无法打开, 主要是webp格式导致的，解决办法：全部转换为jpg格式
    else:
        jpeg_image_list = any2jpg(image_path_list)
        line['image'] = [str(f) for f in jpeg_image_list]
        line.pop('invalid_images', None)
        keep.append(line)

json.dump(keep, open(output_file, 'w'), indent=2, ensure_ascii=False)
json.dump(drop, open(drop_file, 'w'), indent=2, ensure_ascii=False)

