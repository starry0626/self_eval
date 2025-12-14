import os
from huggingface_hub import snapshot_download

# 1. 设置国内镜像地址，加速下载（关键步骤）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 定义下载参数
repo_id = "jinyoungkim/NExT-GQA"
target_folder = "NExTVideo"  # 只下载这个文件夹
local_dir = "/home/ma-user/work/NExT-GQA"  # ModelArts 持久化存储路径

print(f"开始下载 {repo_id} 中的 {target_folder} 到 {local_dir} ...")

# 3. 开始下载
# allow_patterns 确保只下载 NExTVideo 目录下的所有文件
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=f"{target_folder}/*", 
    resume_download=True,  # 支持断点续传
    max_workers=8          # 多线程下载
)

print("下载完成！")