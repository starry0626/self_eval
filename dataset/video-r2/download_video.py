
import os
from huggingface_hub import login, snapshot_download



repo_id = "MBZUAI/Video-R2-Dataset"
local_dir = "./video_data"

print("🔐 正在进行强制登录...")
try:
    # 将 Token 写入本地缓存，确保 Xet 独立进程能读到
    login(token=MY_TOKEN, add_to_git_credential=True)
    print("✅ 登录成功！")
except Exception as e:
    print(f"❌ 登录失败: {e}")
    exit()

print("="*60)
print(f"🚀 准备从镜像站下载数据集...")
print(f"📦 目标仓库: {repo_id}")
print(f"📂 保存路径: {os.path.abspath(local_dir)}")
print("="*60)
print("⏳ 正在获取文件列表，马上出现下载进度条（支持断点续传）...\n")

try:
    # snapshot_download 默认会自动调用 tqdm 显示进度条
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="videos/*",  # 只下载视频文件夹
        local_dir=local_dir,        # 保存到当前目录的 video_data
        resume_download=True,       # 开启断点续传（哪怕意外断网，重新运行也能接着下）
        max_workers=8,               # 开启8个线程同时下载8个文件，拉满带宽
        token=MY_TOKEN
    )
    print("\n" + "="*60)
    print(f"🎉 恭喜！所有视频已成功下载并完整保存在:\n {downloaded_path}")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 下载过程中遇到问题: {e}")

