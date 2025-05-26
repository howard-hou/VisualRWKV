import os
import sys
from pathlib import Path

def get_entry_stems_first_level(folder):
    """返回 stem -> 完整路径 的映射（不判断文件类型，只看第一层条目）"""
    stem_to_path = {}
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        stem = Path(entry).stem
        stem_to_path[stem] = full_path
    return stem_to_path

def main():
    if len(sys.argv) != 3:
        print("用法：python diff_stem_delete_common.py <文件夹A路径> <文件夹B路径>")
        sys.exit(1)

    folder_a = sys.argv[1]
    folder_b = sys.argv[2]

    if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
        print("错误：路径无效或不是文件夹。")
        sys.exit(1)

    stems_a = get_entry_stems_first_level(folder_a)
    stems_b = get_entry_stems_first_level(folder_b)

    common_stems = set(stems_a.keys()) & set(stems_b.keys())

    print(f"共同 stem 数量：{len(common_stems)}")

    for stem in common_stems:
        try:
            os.remove(stems_a[stem])
            print(f"已删除：{stems_a[stem]}")
        except Exception as e:
            print(f"删除失败：{stems_a[stem]}，错误：{e}")

if __name__ == "__main__":
    main()
