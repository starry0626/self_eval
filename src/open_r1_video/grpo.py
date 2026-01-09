# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from typing import List, Tuple

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1_video.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy",],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )

def get_generated_text(content):
    """
    Helper to extract text from GRPO generated content.
    The content might be a string or a list of dicts (if conversational).
    """
    # 如果是列表且包含字典（GRPO Trainer封装的对话格式）
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return content[0].get("content", "")
    # 如果已经是字符串
    return content

def extract_multiple_time_ranges(response: str) -> List[Tuple[float, float]]:
    """从响应中提取多个时间范围"""
    matches = re.findall(
        r'<time_range>\s*([\d.]+)\s*,\s*([\d.]+)\s*</time_range>', 
        response, 
        re.IGNORECASE
    )
    time_ranges = []
    for match in matches:
        start, end = float(match[0]), float(match[1])
        if end > start:
            time_ranges.append((start, end))
    return time_ranges

def calculate_temporal_iou_multi(pred_ranges: List[Tuple[float, float]], gt_windows: List[List[int]]) -> float:
    """计算预测时间范围与多个真实窗口的最大IoU"""
    if not pred_ranges or not gt_windows:
        return 0.0
    
    max_iou = 0.0
    gt_intervals = [(float(w[0]), float(w[1])) for w in gt_windows]
    
    for pred_start, pred_end in pred_ranges:
        if pred_end <= pred_start: continue
        for gt_start, gt_end in gt_intervals:
            if gt_end <= gt_start: continue
            
            inter_start = max(pred_start, gt_start)
            inter_end = min(pred_end, gt_end)
            
            if inter_end <= inter_start: continue
            
            inter_len = inter_end - inter_start
            union_len = (pred_end - pred_start) + (gt_end - gt_start) - inter_len
            
            iou = inter_len / max(union_len, 1e-8)
            max_iou = max(max_iou, iou)
    return max_iou

# --- 定义新的 GRPO 奖励函数 ---

def accuracy_reward_video(completions, solution, **kwargs):
    """视频QA准确率奖励"""
    rewards = []
    for content, sol in zip(completions, solution):
        content = get_generated_text(content)
        # 提取模型答案 <answer>A</answer>
        pred_match = re.search(r'<answer>\s*([A-D])\s*</answer>', content, re.IGNORECASE)
        # 提取真值答案 (假设 dataset 中 solution 是 "A" 或者包含标签的字符串)
        # 如果 dataset['solution'] 已经是 'A', 'B' 等，直接用，否则需要提取
        gt_match = re.search(r'<answer>\s*([A-D])\s*</answer>', sol, re.IGNORECASE)
        gt_char = gt_match.group(1).upper() if gt_match else sol.strip().upper()
        
        if pred_match and pred_match.group(1).upper() == gt_char:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def temporal_reward(completions, relevant_windows, **kwargs):
    """时间定位 IoU 奖励"""
    rewards = []
    # GRPO 传递进来的 relevant_windows 列表长度等于 batch size
    # completions 是 list of list (or strings), 取决于 trainer 实现，通常是 strings
    for content, gt_wins in zip(completions, relevant_windows):
        content = get_generated_text(content)
        pred_ranges = extract_multiple_time_ranges(content)
        # relevant_windows 在 dataset 中通常是 list of list，这里直接使用
        iou = calculate_temporal_iou_multi(pred_ranges, gt_wins)
        rewards.append(iou)
    return rewards

def format_reward_video(completions, **kwargs):
    """严格遵循 evaluate_model.py 的格式奖励"""
    rewards = []
    for content in completions:
        content = get_generated_text(content)
        has_think = bool(re.search(r'<think>.*?</think>', content, re.IGNORECASE | re.DOTALL))
        has_time = bool(re.search(r'<time_range>\s*[\d.]+\s*,\s*[\d.]+\s*</time_range>', content, re.IGNORECASE))
        has_answer = bool(re.search(r'<answer>[A-D]</answer>', content, re.IGNORECASE))
        
        score = 0.0
        if has_think: score += 0.3
        if has_time: score += 0.3
        if has_answer: score += 0.4
        rewards.append(min(score, 1.0))
    return rewards

# 更新注册表
reward_funcs_registry = {
    "accuracy": accuracy_reward_video,
    "format": format_reward_video,
    "temporal": temporal_reward,
}

VIDEO_GROUNDED_QA_PROMPT = """You are a video understanding assistant. Please analyze the provided video and answer the multiple-choice question.

IMPORTANT: You MUST follow this exact format:
1. First, enclose your step-by-step thinking process within <think> tags
2. Then provide one or more relevant time ranges (in seconds) that support your answer, each enclosed in <time_range> tags
3. Finally, provide your final answer choice enclosed in <answer> tags

Required format:
<think>Your detailed reasoning process here...</think>
<time_range>start_time1, end_time1</time_range>
<time_range>start_time2, end_time2</time_range>  (if there are multiple relevant segments)
<answer>A/B/C/D</answer>

Question: {question}
Options:
{options}

Note: The video duration is {duration} seconds. Sample frames are provided from key moments. There may be multiple relevant time periods in the video."""

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        # 格式化选项
        if isinstance(example["options"], list):
            options_text = "\n".join(example["options"])
        else:
            options_text = example["options"]

        prompt_text = VIDEO_GROUNDED_QA_PROMPT.format(
            question=example["problem"],
            options=options_text,
            duration=example["duration"]
        )

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": example["path"], "fps": 0.5, "max_frames":32}, # 只传递路径，让 process_vision_info 处理
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            # 重要：将这些字段传递给 dataset，以便 GRPO 能够将其作为 **kwargs 传递给 reward functions
            "relevant_windows": example["relevant_windows"], 
            "solution": example["solution"]
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features or "path" in dataset[script_args.dataset_train_split].features:
        # 检测到 video 或 path 字段使用视频处理逻辑
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
