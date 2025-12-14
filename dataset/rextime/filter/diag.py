# diagnostic.py - 在 eval.py 同目录下运行
import os
from datasets import load_dataset

# ================= 你的原始配置 =================
DATASET_ID = "./ReXTime"  # 你的本地数据集路径
VIDEO_ROOT_PATH = "./dataset/rextime/v1-2/train"  # 你的视频路径
MODEL_ID = "./models/qwen3-vl-2b-thinking"  # 你的模型路径

print("="*60)
print("步骤1: 检查数据集加载")
try:
    ds = load_dataset(DATASET_ID, split="train")
    print(f"✅ 数据集加载成功，共 {len(ds)} 条样本")
    if len(ds) == 0:
        print("❌ 数据集为空！")
        exit()
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    exit()

print("\n" + "="*60)
print("步骤2: 检查视频文件是否存在")
missing = 0
sample_item = ds[0]  # 取第一条样本查看结构
print(f"样本数据结构: {list(sample_item.keys())}")
print(f"样本 vid 字段值: {sample_item.get('vid')}")

for i, item in enumerate(ds):
    vid = item['vid']

    # 1. 若数据集本身带后缀，直接用上
    if vid.lower().endswith(('.mp4', '.mkv', '.avi')):
        video_filename = vid
    else:
        # 2. 无后缀 → 按优先级依次探测
        for ext in ('.mp4', '.mkv', '.avi'):
            probe = vid + ext
            if os.path.exists(os.path.join(VIDEO_ROOT_PATH, probe)):
                video_filename = probe
                break
        else:          # 一个都没找到，默认回退 .mp4（便于后面计数）
            video_filename = vid + '.mp4'

    video_path = os.path.join(VIDEO_ROOT_PATH, video_filename)
    if not os.path.exists(video_path):
        missing += 1
        if i < 5:
            print(f"❌ 不存在: {video_path}")
print(f"总计: {missing}/{len(ds)} 个视频文件缺失")

print("\n" + "="*60)
print("步骤3: 检查模型文件完整性")
required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'preprocessor_config.json']
for f in required_files:
    path = os.path.join(MODEL_ID, f)
    if os.path.exists(path):
        print(f"✅ {f} 存在")
    else:
        print(f"❌ {f} 缺失！")