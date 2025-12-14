# 方法A: 使用gdown下载Google Drive
import gdown

# 下载原始视频
gdown.download(
    "https://drive.google.com/uc?id=1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH",
    "NExTVideo.zip",
    quiet=False
)

# 解压
import zipfile
with zipfile.ZipFile("NExTVideo.zip", 'r') as zip_ref:
    zip_ref.extractall("./NExT-GQA")

# # 方法B: 配置代理后使用hf下载
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['NO_PROXY'] = 'hf-mirror.com'

# from huggingface_hub import snapshot_download

# try:
#     snapshot_download(
#         repo_id="jinyoungkim/NExT-GQA",
#         local_dir="./NExT-GQA",
#         allow_patterns=["NExTVideo/*"],
#         resume_download=True
#     )
# except Exception as e:
#     print(f"HuggingFace下载失败: {e}")
#     print("请使用Google Drive方案")