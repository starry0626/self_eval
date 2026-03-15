#!/usr/bin/env python3
"""
Backfill Training Log — 为现有训练日志补充正确答案与教师额外信息

读取现有 JSONL 训练日志和原始数据集 JSON，按 sample_id 匹配，
使用 teacher_context_builder 中的提取函数补充以下字段：
  - correct_answer
  - correct_answer_text
  - teacher_extra_info (answer / temporal_text / reasoning / temporal_video_frames)

Usage:
    python scripts/backfill_training_log.py training_log_rank0.jsonl \
        --dataset dataset/video-r2/Video-R2/video-r2-grpo-dataset.json \
        --include-answer --include-temporal-text --include-temporal-video \
        --temporal-fps 1.0 --temporal-max-frames 8 --max-temporal-segments 5 \
        --max-total-extra-frames 48
"""

import argparse
import json
import os
import re
import sys

# 将 src/ 目录加入搜索路径以便导入 teacher_context_builder
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, os.pardir, "src")
sys.path.insert(0, os.path.abspath(_src_dir))

from open_r1_video.trainer.teacher_context_builder import (
    TeacherContextConfig,
    extract_answer_from_conversation,
    extract_reasoning_from_conversation,
    extract_temporal_grounding_text,
    compute_temporal_video_frame_timestamps,
)


def parse_question_options(text: str):
    """
    从问题文本中解析出题干和选项（与 sdpo_trainer._parse_question_options 相同逻辑）
    """
    if not text:
        return "", {}

    _ponder_marker = "Please think about this question as if you were a human pondering deeply"
    if _ponder_marker in text:
        text = text[:text.index(_ponder_marker)].rstrip()

    parts = re.split(r'\n(?=[A-E]\.)', text)
    if len(parts) <= 1:
        return text.strip(), {}

    question = parts[0].strip()
    options = {}
    for part in parts[1:]:
        part = part.strip()
        if part and len(part) >= 2 and part[0] in "ABCDE" and part[1] == ".":
            options[part[0]] = part[2:].strip()

    return question, options


def backfill_record(record, dataset_lookup, config):
    """为单条日志记录补充新字段"""
    sample_id = record.get("sample_id", "")
    sample = dataset_lookup.get(sample_id)

    if sample is None:
        # 找不到匹配的数据集样本，填充空值
        record.setdefault("correct_answer", "")
        record.setdefault("correct_answer_text", "")
        record.setdefault("teacher_extra_info", {})
        return record

    conversations = sample.get("conversations", [])

    # 提取正确答案
    answer_text, answer_letter = extract_answer_from_conversation(conversations)

    # 解析选项（优先用记录中已有的 options，否则从数据集重新解析）
    options = record.get("options", {})
    if not options:
        raw_question = conversations[0]["value"] if conversations else ""
        _, options = parse_question_options(raw_question)

    correct_answer = answer_letter
    if answer_letter and answer_letter in options:
        correct_answer_text = f"{answer_letter}. {options[answer_letter]}"
    else:
        correct_answer_text = answer_text

    record["correct_answer"] = correct_answer
    record["correct_answer_text"] = correct_answer_text

    # 构建 teacher_extra_info
    teacher_extra_info = {}

    if config.include_answer and answer_text:
        teacher_extra_info["answer"] = correct_answer_text

    temporal_grounding = sample.get("temporal_grounding")

    if config.include_temporal_text and temporal_grounding:
        temporal_text = extract_temporal_grounding_text(temporal_grounding)
        if temporal_text:
            teacher_extra_info["temporal_text"] = temporal_text

    if config.include_reasoning:
        reasoning = extract_reasoning_from_conversation(conversations)
        if reasoning:
            teacher_extra_info["reasoning"] = reasoning

    if config.include_temporal_video and temporal_grounding:
        temporal_frames = compute_temporal_video_frame_timestamps(
            temporal_grounding, config
        )
        if temporal_frames:
            teacher_extra_info["temporal_video_frames"] = temporal_frames

    record["teacher_extra_info"] = teacher_extra_info

    return record


def main():
    parser = argparse.ArgumentParser(
        description="为现有训练日志补充正确答案与教师额外信息"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="训练日志 JSONL 文件路径（支持多个）",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集 JSON 文件路径",
    )
    parser.add_argument("-o", "--output", help="输出路径（默认：输入文件名加 _backfilled 后缀）")

    # 信息包含开关
    parser.add_argument("--include-answer", action="store_true", default=False)
    parser.add_argument("--include-temporal-text", action="store_true", default=False)
    parser.add_argument("--include-temporal-video", action="store_true", default=False)
    parser.add_argument("--include-reasoning", action="store_true", default=False)

    # 时间段采样参数
    parser.add_argument("--temporal-fps", type=float, default=1.0)
    parser.add_argument("--temporal-max-frames", type=int, default=8)
    parser.add_argument("--point-frames-count", type=int, default=4)
    parser.add_argument("--point-frames-range", type=float, default=1.0)
    parser.add_argument("--max-temporal-segments", type=int, default=5)
    parser.add_argument("--max-total-extra-frames", type=int, default=None)

    args = parser.parse_args()

    # 构建配置
    config = TeacherContextConfig(
        include_answer=args.include_answer,
        include_temporal_text=args.include_temporal_text,
        include_temporal_video=args.include_temporal_video,
        include_reasoning=args.include_reasoning,
        temporal_fps=args.temporal_fps,
        temporal_max_frames=args.temporal_max_frames,
        point_frames_count=args.point_frames_count,
        point_frames_range=args.point_frames_range,
        max_temporal_segments=args.max_temporal_segments,
        max_total_extra_frames=args.max_total_extra_frames,
    )

    # 加载数据集并建立 id -> sample 索引
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_lookup = {}
    for sample in dataset:
        sid = sample.get("id", "")
        if sid:
            dataset_lookup[sid] = sample
    print(f"  Indexed {len(dataset_lookup)} samples from dataset")

    # 逐文件处理
    for input_path in args.input:
        print(f"\nProcessing: {input_path}")

        records = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        matched = 0
        for record in records:
            backfill_record(record, dataset_lookup, config)
            if record.get("correct_answer"):
                matched += 1

        # 确定输出路径
        if args.output and len(args.input) == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_backfilled{ext}"

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Records: {len(records)} | Matched: {matched} | Output: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
