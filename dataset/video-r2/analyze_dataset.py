"""分析 Video-R2 数据集格式"""
import json
import os

# 切换到数据目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Video-R2")
os.chdir(data_dir)

# 读取 GRPO 数据集
with open('video-r2-grpo-dataset.json', 'r', encoding='utf-8') as f:
    grpo_data = json.load(f)

print('=' * 60)
print('GRPO 数据集分析')
print('=' * 60)
print(f'总样本数: {len(grpo_data)}')
print()

# 分析第一个样本
sample = grpo_data[0]
print('样本字段:')
for key in sample.keys():
    print(f'  - {key}')

print()
print('示例数据:')
print(f'  id: {sample["id"]}')
print(f'  video: {sample["video"]}')
print()

# 分析 conversations
print('conversations 结构:')
for i, conv in enumerate(sample['conversations']):
    print(f'  [{i}] from: {conv["from"]}')
    value = conv['value']
    if len(value) > 200:
        print(f'      value (前200字符): {value[:200]}...')
    else:
        print(f'      value: {value}')
print()

# 分析 temporal_grounding
tg = sample['temporal_grounding']
non_null = {k: v for k, v in tg.items() if v is not None}
print(f'temporal_grounding:')
print(f'  总时间戳数: {len(tg)}')
print(f'  有标注数: {len(non_null)}')
if non_null:
    print(f'  示例标注:')
    for k, v in list(non_null.items())[:3]:
        print(f'    {k}: {v}')
print()

# 统计视频来源
sources = {}
for item in grpo_data:
    parts = item['video'].split('/')
    if parts:
        sources[parts[0]] = sources.get(parts[0], 0) + 1

print('视频来源分布:')
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'  {src}: {cnt}')

# 分析 SFT 数据集
print()
print('=' * 60)
print('SFT 数据集分析')
print('=' * 60)

with open('video-r2-sft-dataset.json', 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

print(f'总样本数: {len(sft_data)}')
print()

# 分析第一个样本
sft_sample = sft_data[0]
print('样本字段:')
for key in sft_sample.keys():
    print(f'  - {key}')

print()
print('示例数据:')
print(f'  id: {sft_sample["id"]}')
print(f'  video: {sft_sample["video"]}')
print()

# 分析 conversations
print('conversations 结构:')
for i, conv in enumerate(sft_sample['conversations']):
    print(f'  [{i}] from: {conv["from"]}')
    value = conv['value']
    if len(value) > 300:
        print(f'      value (前300字符): {value[:300]}...')
    else:
        print(f'      value: {value}')
