"""
MBZUAI/Video-R2-Dataset 数据集下载脚本

该数据集用于视频问答任务，包含：
- 视频文件
- 对话数据（问题和带推理过程的回答）
- 时间定位标注

数据集结构：
- id: 数据唯一标识符
- video: 视频文件路径
- conversations: 对话列表
  - from: "human" 或 "assistant"
  - value: 对话内容（assistant的回答包含 <think/> 和 <answer/> 标签）
- temporal_grounding: 时间定位字典
  - key: 时间戳（如 "00:00", "00:20-00:48"）
  - value: 该时间段的描述或 None

使用方法：
    python download.py
"""

import os
import json
from huggingface_hub import hf_hub_download, snapshot_download, HfApi


def download_dataset():
    """下载 Video-R2-Dataset 数据集"""
    
    # 设置工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 创建数据目录
    data_dir = "./Video-R2"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始下载 MBZUAI/Video-R2-Dataset 数据集")
    print("=" * 60)
    
    # 方法1: 下载数据集的 JSON 文件（推荐，数据量小）
    print("\n[步骤1] 下载 JSON 数据文件...")
    
    try:
        # 下载 grpo subset 的数据
        json_file = hf_hub_download(
            repo_id="MBZUAI/Video-R2-Dataset",
            filename="grpo/train.json",
            repo_type="dataset",
            local_dir=data_dir,
        )
        print(f"✓ JSON 文件已下载到: {json_file}")
        
        # 读取并分析数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n数据集统计:")
        print(f"  - 总样本数: {len(data)}")
        if len(data) > 0:
            sample = data[0]
            print(f"  - 字段: {list(sample.keys())}")
            print(f"  - 示例 ID: {sample.get('id', 'N/A')}")
            print(f"  - 视频路径: {sample.get('video', 'N/A')}")
            
    except Exception as e:
        print(f"✗ JSON 文件下载失败: {e}")
        print("尝试使用 snapshot_download...")
        
        # 方法2: 使用 snapshot_download
        try:
            snapshot_download(
                repo_id="MBZUAI/Video-R2-Dataset",
                local_dir=data_dir,
                repo_type="dataset",
                allow_patterns=["*.json"],
                resume_download=True
            )
            print(f"✓ 数据集已下载到: {data_dir}")
        except Exception as e2:
            print(f"✗ 下载失败: {e2}")
            return None
    
    # 列出数据集仓库中的文件
    print("\n[步骤2] 查看数据集仓库结构...")
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id="MBZUAI/Video-R2-Dataset", repo_type="dataset")
        
        # 分类显示文件
        json_files = [f for f in files if f.endswith('.json')]
        video_patterns = [f for f in files if f.endswith(('.mp4', '.webm', '.mkv'))]
        other_files = [f for f in files if not f.endswith(('.json', '.mp4', '.webm', '.mkv'))]
        
        print(f"\n仓库文件统计:")
        print(f"  - JSON 文件: {len(json_files)}")
        if json_files[:5]:
            print(f"    示例: {json_files[:5]}")
        print(f"  - 视频文件: {len(video_patterns)} (需要单独下载)")
        print(f"  - 其他文件: {len(other_files)}")
        
    except Exception as e:
        print(f"无法获取仓库文件列表: {e}")
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"数据目录: {os.path.abspath(data_dir)}")
    print("=" * 60)
    
    return data_dir


def download_videos(video_list=None, max_videos=None):
    """
    下载视频文件（可选）
    
    参数:
        video_list: 要下载的视频路径列表，None 表示下载所有
        max_videos: 最大下载数量，用于测试
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    data_dir = "./Video-R2"
    
    # 读取 JSON 文件获取视频列表
    json_path = os.path.join(data_dir, "grpo/train.json")
    if not os.path.exists(json_path):
        print(f"JSON 文件不存在: {json_path}")
        print("请先运行 download_dataset() 下载数据文件")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取视频路径列表
    if video_list is None:
        video_list = list(set(item['video'] for item in data))
    
    if max_videos:
        video_list = video_list[:max_videos]
    
    print(f"准备下载 {len(video_list)} 个视频文件...")
    
    # 下载视频
    for i, video_path in enumerate(video_list):
        try:
            print(f"[{i+1}/{len(video_list)}] 下载: {video_path}")
            hf_hub_download(
                repo_id="MBZUAI/Video-R2-Dataset",
                filename=video_path,
                repo_type="dataset",
                local_dir=data_dir,
            )
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
    
    print("视频下载完成!")


def analyze_dataset():
    """分析数据集格式和内容"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "Video-R2/grpo/train.json")
    
    if not os.path.exists(json_path):
        print(f"JSON 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 60)
    print("数据集分析报告")
    print("=" * 60)
    
    print(f"\n总样本数: {len(data)}")
    
    # 分析字段
    if data:
        sample = data[0]
        print(f"\n字段列表:")
        for key, value in sample.items():
            if key == "conversations":
                print(f"  - {key}: 列表, 长度 {len(value)}")
                if value:
                    for i, conv in enumerate(value):
                        print(f"      [{i}] from: {conv.get('from')}, value 长度: {len(conv.get('value', ''))}")
            elif key == "temporal_grounding":
                non_null_count = sum(1 for v in value.values() if v is not None)
                print(f"  - {key}: 字典, {len(value)} 个时间戳, {non_null_count} 个有标注")
            else:
                print(f"  - {key}: {type(value).__name__} = {str(value)[:50]}...")
    
    # 统计视频来源
    video_sources = {}
    for item in data:
        video_path = item.get('video', '')
        # 提取来源目录
        parts = video_path.split('/')
        if len(parts) >= 2:
            source = parts[0]
            video_sources[source] = video_sources.get(source, 0) + 1
    
    print(f"\n视频来源分布:")
    for source, count in sorted(video_sources.items(), key=lambda x: -x[1]):
        print(f"  - {source}: {count}")
    
    # 分析 temporal_grounding
    total_timestamps = 0
    total_annotations = 0
    for item in data:
        tg = item.get('temporal_grounding', {})
        total_timestamps += len(tg)
        total_annotations += sum(1 for v in tg.values() if v is not None)
    
    print(f"\n时间定位统计:")
    print(f"  - 总时间戳数: {total_timestamps}")
    print(f"  - 有标注的时间戳数: {total_annotations}")
    print(f"  - 标注比例: {total_annotations/total_timestamps*100:.1f}%" if total_timestamps > 0 else "  - 标注比例: N/A")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 下载数据集 JSON 文件
    download_dataset()
    
    # 分析数据集
    analyze_dataset()
    
    # 如果需要下载视频文件，取消下面的注释
    # download_videos(max_videos=10)  # 先下载10个测试
