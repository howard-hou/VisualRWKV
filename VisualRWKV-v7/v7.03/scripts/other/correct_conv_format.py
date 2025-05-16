import json
import re
import sys

def get_num_of_image(text):
    # 使用正则表达式查找所有 <image> 标签
    matches = re.findall(r"<image>", text)
    # 获取匹配到的数量
    count = len(matches)
    return count

def validate_and_correct_conversations(conversations, num_image_paths):
    new_convs = []
    num_convs = len(conversations)
    num_images = 0
    if num_convs % 2 != 0:
        conversations = conversations[:-1]
    for i in range(0, len(conversations), 2):
        human, gpt = conversations[i], conversations[i+1]
        if 'from' not in human or 'from' not in gpt:
            continue
        if 'value' not in human or 'value' not in gpt:
            continue
        if not human or not gpt:
            continue
        human['from'] = 'human'
        gpt['from'] = 'gpt'
        num_images_human = get_num_of_image(human['value'])
        num_images_gpt = get_num_of_image(gpt['value'])
        if num_images_gpt != 0:
            gpt['value'] = gpt['value'].replace('<image>', '').strip()
        new_convs.append(human.copy())
        new_convs.append(gpt.copy())
        num_images += num_images_human

    if num_images < num_image_paths:
        # add image token at the beginning of the conversation
        image_prefix = "\n".join(["<image>"] * (num_image_paths - num_images))
        new_convs[0]['value'] = image_prefix + "\n" + new_convs[0]['value']
        num_images = num_image_paths

    return new_convs, num_images

# 读取 JSON 文件
with open(sys.argv[1], 'r') as file:
    data = json.load(file)

# 检查并修正 conversations 字段
new_data = []
for line in data:
    if 'image' not in line:
        new_data.append(line)
        continue
    if isinstance(line['image'], str):
        line['image'] = [line['image']]
    num_image_paths = len(line['image'])
    if 'conversations' in line and isinstance(line['conversations'], list):
        new_convs, num_images = validate_and_correct_conversations(line['conversations'], num_image_paths)
        if new_convs and num_images == num_image_paths:
            line["conversations"] = new_convs
            new_data.append(line)
        else:
            print(f"num_images in conv: {num_images}, num_image_paths: {num_image_paths}")

# 输出修正后的 JSON 数据
print("input data:", len(data)) 
print("keep data:", len(new_data)) 
with open(sys.argv[1] + '.corrected', 'w') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print("JSON 文件已成功修正并保存。")

