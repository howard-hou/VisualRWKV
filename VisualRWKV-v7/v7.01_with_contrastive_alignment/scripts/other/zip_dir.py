import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

# 压缩命令
def zip_folder(folder_path, output_dir):
    folder_name = os.path.basename(folder_path)
    output_filename = os.path.join(output_dir, f"{folder_name}.zip")
    # 使用 -q 使 zip 安静运行，-0 表示不压缩
    subprocess.run(['zip', '-q', '-0', '-r', output_filename, folder_path])
    return output_filename

# 异步压缩
def async_zip(base_dir, output_dir, num_processes=4):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子文件夹
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # 使用 multiprocessing.Pool 来并行处理压缩任务
    pool = Pool(num_processes)
    results = []
    for folder in folders:
        res = pool.apply_async(zip_folder, args=(folder, output_dir))
        results.append(res)

    # 等待所有任务完成
    for res in tqdm(results, desc="zipping"):
        filename = res.get()
        print(f"zipped {filename}")
    
    pool.close()
    pool.join()
    

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1]       # 原始文件夹路径
    output_dir = sys.argv[2]     # 压缩包输出路径
    async_zip(base_dir, output_dir)
