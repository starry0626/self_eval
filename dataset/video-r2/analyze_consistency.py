"""
分析 Video-R2 数据集中推理过程与 temporal_grounding 字段的一致性

目标：
1. 提取推理过程中提到的时间戳
2. 与 temporal_grounding 字段进行对比
3. 分析一致性和差异
"""

import json
import re
from collections import defaultdict

# 读取数据集
with open('video-r2-grpo-dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 70)
print("推理过程与 temporal_grounding 一致性分析")
print("=" * 70)
print(f"总样本数: {len(data)}")
print()

# 时间戳正则表达式
# 匹配格式: 00:00, 00:00-00:10, 00:00 onwards, 00:00-End 等
timestamp_pattern = r'(\d{1,2}:\d{2}(?:\s*(?:onwards|-End))?(?:-\d{1,2}:\d{2})?)'

def extract_timestamps_from_text(text):
    """从文本中提取时间戳"""
    matches = re.findall(timestamp_pattern, text)
    # 标准化时间戳格式
    normalized = set()
    for m in matches:
        # 去除多余空格
        m = m.strip()
        normalized.add(m)
    return normalized

# 统计变量
total_samples = len(data)
samples_with_temporal = 0
samples_with_reasoning_timestamps = 0
samples_with_both = 0
exact_match_count = 0
partial_match_count = 0
no_match_count = 0

# 详细分析结果
analysis_results = []

for i, sample in enumerate(data):
    temporal_grounding = sample.get('temporal_grounding', {})
    conversations = sample.get('conversations', [])
    
    # 获取 assistant 的回答
    assistant_response = ""
    for conv in conversations:
        if conv.get('from') == 'assistant':
            assistant_response = conv.get('value', '')
            break
    
    # 提取 <think/> 标签内的推理过程
    think_match = re.search(r'<think\]?(.*?)</think\]?', assistant_response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # 从推理过程中提取时间戳
    reasoning_timestamps = extract_timestamps_from_text(reasoning)
    
    # 获取 temporal_grounding 中的时间戳
    temporal_timestamps = set(temporal_grounding.keys())
    
    # 过滤掉值为 None 的时间戳
    valid_temporal_timestamps = {k for k, v in temporal_grounding.items() if v is not None}
    
    # 统计
    has_temporal = len(valid_temporal_timestamps) > 0
    has_reasoning_ts = len(reasoning_timestamps) > 0
    
    if has_temporal:
        samples_with_temporal += 1
    if has_reasoning_ts:
        samples_with_reasoning_timestamps += 1
    if has_temporal and has_reasoning_ts:
        samples_with_both += 1
    
    # 计算匹配程度
    if has_temporal and has_reasoning_ts:
        # 检查是否有交集
        intersection = reasoning_timestamps & valid_temporal_timestamps
        
        if intersection == valid_temporal_timestamps and intersection == reasoning_timestamps:
            exact_match_count += 1
            match_type = "exact"
        elif intersection:
            partial_match_count += 1
            match_type = "partial"
        else:
            no_match_count += 1
            match_type = "none"
        
        # 保存详细结果（只保存前20个有代表性的样本）
        if len(analysis_results) < 20:
            analysis_results.append({
                'id': sample['id'],
                'temporal_timestamps': valid_temporal_timestamps,
                'reasoning_timestamps': reasoning_timestamps,
                'intersection': intersection,
                'match_type': match_type,
                'temporal_content': {k: v for k, v in temporal_grounding.items() if v is not None},
                'reasoning_snippet': reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            })

# 打印统计结果
print("统计结果:")
print("-" * 70)
print(f"有 temporal_grounding 标注的样本数: {samples_with_temporal} ({samples_with_temporal/total_samples*100:.1f}%)")
print(f"推理过程中包含时间戳的样本数: {samples_with_reasoning_timestamps} ({samples_with_reasoning_timestamps/total_samples*100:.1f}%)")
print(f"同时包含两者的样本数: {samples_with_both} ({samples_with_both/total_samples*100:.1f}%)")
print()

if samples_with_both > 0:
    print("匹配程度分析 (在同时包含两者的样本中):")
    print(f"  完全匹配: {exact_match_count} ({exact_match_count/samples_with_both*100:.1f}%)")
    print(f"  部分匹配: {partial_match_count} ({partial_match_count/samples_with_both*100:.1f}%)")
    print(f"  无匹配: {no_match_count} ({no_match_count/samples_with_both*100:.1f}%)")
print()

# 打印详细分析
print("=" * 70)
print("详细样本分析")
print("=" * 70)

for i, result in enumerate(analysis_results[:10]):
    print(f"\n样本 {i+1}: {result['id']}")
    print("-" * 50)
    
    print(f"temporal_grounding 时间戳 ({len(result['temporal_timestamps'])} 个):")
    for ts in sorted(result['temporal_timestamps'])[:5]:
        content = result['temporal_content'].get(ts, 'N/A')
        if content:
            print(f"  {ts}: {content[:80]}..." if len(content) > 80 else f"  {ts}: {content}")
    
    print(f"\n推理过程中提到的时间戳 ({len(result['reasoning_timestamps'])} 个):")
    for ts in sorted(result['reasoning_timestamps'])[:10]:
        print(f"  {ts}")
    
    print(f"\n交集时间戳: {result['intersection']}")
    print(f"匹配类型: {result['match_type']}")

# 分析推理过程中时间戳的上下文
print("\n" + "=" * 70)
print("推理过程中时间戳的上下文分析")
print("=" * 70)

# 找一个有代表性的样本详细展示
for sample in data[:100]:
    temporal_grounding = sample.get('temporal_grounding', {})
    valid_temporal = {k: v for k, v in temporal_grounding.items() if v is not None}
    
    if len(valid_temporal) >= 2:
        conversations = sample.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'assistant':
                assistant_response = conv.get('value', '')
                think_match = re.search(r'<think\]?(.*?)</think\]?', assistant_response, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    reasoning_ts = extract_timestamps_from_text(reasoning)
                    
                    if reasoning_ts & set(valid_temporal.keys()):
                        print(f"\n样本 ID: {sample['id']}")
                        print("-" * 50)
                        
                        # 显示问题
                        for c in conversations:
                            if c.get('from') == 'human':
                                question = c.get('value', '')[:300]
                                print(f"问题: {question}...")
                                break
                        
                        print(f"\ntemporal_grounding 内容:")
                        for ts, content in list(valid_temporal.items())[:5]:
                            print(f"  [{ts}]: {content}")
                        
                        print(f"\n推理过程片段 (包含时间戳的部分):")
                        # 找到包含时间戳的句子
                        sentences = reasoning.split('\n')
                        for sent in sentences:
                            if any(ts in sent for ts in valid_temporal):
                                print(f"  {sent[:200]}...")
                        break
        break

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
1. temporal_grounding 字段提供了关键时间点的标注
2. 推理过程中可能引用这些时间戳，但格式可能不完全一致
3. 推理过程更侧重于逻辑推理，时间戳作为证据引用
4. 两者可以互补：temporal_grounding 提供精确标注，推理过程提供上下文
""")
