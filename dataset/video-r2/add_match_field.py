"""
为 Video-R2 数据集添加 temporal_grounding 匹配标记

匹配类型编码：
- E (Exact): 完全匹配 - 推理过程中的时间戳与 temporal_grounding 完全一致
- P (Partial): 部分匹配 - 推理过程中的时间戳与 temporal_grounding 有交集
- N (None): 无匹配 - 两者都有时间戳但无交集
- T (Temporal only): 只有 temporal_grounding 有标注，推理过程无时间戳
- R (Reasoning only): 只有推理过程有时间戳，无 temporal_grounding
- B (Both empty): 两者都为空
"""

import json
import re
from collections import defaultdict

# 时间戳正则表达式
timestamp_pattern = r'(\d{1,2}:\d{2}(?:\s*(?:onwards|-End))?(?:-\d{1,2}:\d{2})?)'

def extract_timestamps_from_text(text):
    """从文本中提取时间戳"""
    matches = re.findall(timestamp_pattern, text)
    normalized = set()
    for m in matches:
        m = m.strip()
        normalized.add(m)
    return normalized

def determine_match_type(sample):
    """
    判断匹配类型
    
    返回:
        E: 完全匹配
        P: 部分匹配
        N: 无匹配
        T: 只有temporal_grounding
        R: 只有推理过程有时间戳
        B: 两者都为空
    """
    temporal_grounding = sample.get('temporal_grounding', {})
    conversations = sample.get('conversations', [])
    
    # 获取 assistant 的回答
    assistant_response = ""
    for conv in conversations:
        if conv.get('from') == 'assistant':
            assistant_response = conv.get('value', '')
            break
    
    # 提取推理过程
    think_match = re.search(r'<think\]?(.*?)</think\]?', assistant_response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # 从推理过程中提取时间戳
    reasoning_timestamps = extract_timestamps_from_text(reasoning)
    
    # 获取 temporal_grounding 中的有效时间戳
    valid_temporal_timestamps = {k for k, v in temporal_grounding.items() if v is not None}
    
    # 判断匹配类型
    has_temporal = len(valid_temporal_timestamps) > 0
    has_reasoning_ts = len(reasoning_timestamps) > 0
    
    if not has_temporal and not has_reasoning_ts:
        return 'B'  # Both empty
    elif has_temporal and not has_reasoning_ts:
        return 'T'  # Temporal only
    elif not has_temporal and has_reasoning_ts:
        return 'R'  # Reasoning only
    else:
        # 两者都有，计算交集
        intersection = reasoning_timestamps & valid_temporal_timestamps
        
        if intersection == valid_temporal_timestamps and intersection == reasoning_timestamps:
            return 'E'  # Exact match
        elif intersection:
            return 'P'  # Partial match
        else:
            return 'N'  # No match

# 读取数据集
print("读取数据集...")
with open('video-r2-grpo-dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总样本数: {len(data)}")

# 统计各类型数量
match_stats = defaultdict(int)

# 添加匹配标记
for sample in data:
    match_type = determine_match_type(sample)
    sample['tg_match'] = match_type  # tg = temporal_grounding
    match_stats[match_type] += 1

# 打印统计
print("\n匹配类型统计:")
print("-" * 40)
type_descriptions = {
    'E': '完全匹配',
    'P': '部分匹配',
    'N': '无匹配',
    'T': '只有temporal_grounding',
    'R': '只有推理过程有时间戳',
    'B': '两者都为空'
}
for t in ['E', 'P', 'N', 'T', 'R', 'B']:
    count = match_stats[t]
    pct = count / len(data) * 100
    print(f"  {t} ({type_descriptions[t]}): {count} ({pct:.1f}%)")

# 保存更新后的数据集
output_file = 'video-r2-grpo-dataset-with-match.json'
print(f"\n保存到: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("完成!")

# 显示示例
print("\n" + "=" * 60)
print("示例数据")
print("=" * 60)
for sample in data[:3]:
    print(f"\nID: {sample['id']}")
    print(f"tg_match: {sample['tg_match']}")
    tg = sample.get('temporal_grounding', {})
    valid_tg = {k: v for k, v in tg.items() if v is not None}
    print(f"temporal_grounding 数量: {len(valid_tg)}")
    if valid_tg:
        for k, v in list(valid_tg.items())[:2]:
            print(f"  {k}: {v[:50]}..." if len(v) > 50 else f"  {k}: {v}")
