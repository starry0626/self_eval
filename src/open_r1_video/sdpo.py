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

"""
SDPO (On-Policy Self-Distillation) 训练启动脚本

本脚本用于启动 SDPO 训练，基于 Video-R2 数据集实现视频问答的自蒸馏训练。
SDPO 使用模型自身作为教师，教师接收额外的上下文信息（答案、时间定位、推理过程），
通过 Jensen-Shannon 散度来衡量学生和教师输出分布的差异。

使用方法:
    python src/open_r1_video/sdpo.py \
        --model_name_or_path Qwen3-VL-2B-Thinking \
        --jsonl_path ./dataset/video-r2/video-r2-grpo-dataset.json \
        --output_dir ./output/sdpo \
        --include_answer true \
        --include_temporal_text true \
        --include_temporal_video false \
        --include_reasoning true
"""

import dataclasses
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from datasets import load_dataset, Dataset, DatasetDict
from transformers import HfArgumentParser, TrainerCallback
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1_video.trainer import Qwen2VLSDPOTrainer
from open_r1_video.trainer.sdpo_trainer import SDPOConfig
from open_r1_video.trainer.teacher_context_builder import TeacherContextConfig
from open_r1_video.trainer.divergence import DivergenceConfig


@dataclass
class SDPOScriptArguments(ScriptArguments):
    """
    SDPO 训练脚本的参数配置

    继承自 ScriptArguments，添加 SDPO 特有的参数

    参数:
        jsonl_path: JSON/JSONL 数据集文件路径
        max_pixels: 图像/视频处理的最大像素数
        min_pixels: 图像/视频处理的最小像素数
        video_base_dir: 视频文件基础目录
        include_answer: 是否在教师上下文中包含标准答案
        include_temporal_text: 是否在教师上下文中包含时间定位文本
        include_temporal_video: 是否在教师上下文中包含时间定位视频片段
        include_reasoning: 是否在教师上下文中包含推理流程
        temporal_fps: 时间段视频采样帧率（每秒采样帧数）
        temporal_max_frames: 时间段采样的最大帧数（与 fps 冲突时优先限制帧数）
        teacher_temporal_max_pixels: 教师额外视觉输入的最大像素数（None 表示使用处理器默认值）
        point_frames_count: 时间点采样帧数
        max_temporal_segments: 最大时间片段数量
        use_fixed_teacher: 是否使用固定的独立教师模型（参数不随训练更新）
        teacher_model_path: 固定教师模型路径（None 表示使用与学生相同的模型）
    """
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "JSON/JSONL 数据集文件路径"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "图像/视频处理的最大像素数"}
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "图像/视频处理的最小像素数"}
    )
    video_base_dir: Optional[str] = field(
        default=None,
        metadata={"help": "视频文件基础目录，用于视频帧采样"}
    )

    include_answer: bool = field(
        default=True,
        metadata={"help": "是否在教师上下文中包含标准答案"}
    )
    include_temporal_text: bool = field(
        default=True,
        metadata={"help": "是否在教师上下文中包含时间定位文本"}
    )
    include_temporal_video: bool = field(
        default=False,
        metadata={"help": "是否在教师上下文中包含时间定位视频片段"}
    )
    include_reasoning: bool = field(
        default=True,
        metadata={"help": "是否在教师上下文中包含推理流程"}
    )

    temporal_fps: float = field(
        default=1.0,
        metadata={"help": "时间段视频采样帧率（每秒采样帧数）"}
    )
    temporal_max_frames: int = field(
        default=8,
        metadata={"help": "时间段采样的最大帧数（与 fps 冲突时优先限制帧数）"}
    )
    teacher_temporal_max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "教师额外视觉输入的最大像素数（None 表示使用处理器默认值）"}
    )
    point_frames_count: int = field(
        default=4,
        metadata={"help": "时间点采样时的帧数（前后采样）"}
    )
    max_temporal_segments: int = field(
        default=5,
        metadata={"help": "最大时间片段数量，避免上下文过长"}
    )
    divergence_method: str = field(
        default="full",
        metadata={"help": "散度计算方法: full (完整计算), top_k (top-k 估计), k3 (K3 估计)"}
    )
    divergence_top_k: int = field(
        default=20,
        metadata={"help": "Top-k 估计的 k 值（仅当 divergence_method=top_k 时有效）"}
    )
    use_fixed_teacher: bool = field(
        default=False,
        metadata={"help": "是否使用固定的独立教师模型（参数不随训练更新）"}
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "固定教师模型路径（None 表示使用与学生相同的模型）"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大训练样本数，若小于数据集大小则随机采样，若大于则使用完整数据集（None 表示使用完整数据集）"}
    )


VIDEO_QA_PROMPT = """Please analyze the provided video and answer the multiple-choice question strictly based on the content of the video.

You need to follow the following process: 
1. First, enclose your step-by-step thinking process within <think> and </think> tags.
2. Then provide your final answer choice enclosed in <answer> and </answer> tags. Only output the letter corresponding to the correct option (A, B, C, or D), and nothing else, do not restate the answer text.

Respond in the following format:
<think>
Your detailed reasoning process here...
</think>
<answer>
A/B/C/D
</answer>

{question}"""


def create_dataset_from_json(json_path: str) -> DatasetDict:
    """
    从 JSON 文件创建数据集
    
    参数:
        json_path: JSON 文件路径
    
    返回:
        DatasetDict: 包含 train split 的数据集字典
    """
    dataset = Dataset.from_json(json_path)
    return DatasetDict({
        "train": dataset
    })


def make_conversation_video(example: Dict[str, Any], prompt_template: str, video_base_dir: str = None) -> Dict[str, Any]:
    """
    将数据集样本转换为对话格式
    
    参数:
        example: 数据集样本
        prompt_template: prompt 模板
    
    返回:
        转换后的样本，包含 prompt 字段
    """
    conversations = example.get("conversations", [])
    
    if conversations:
        question_text = conversations[0].get("value", "")
    else:
        question_text = example.get("problem", example.get("question", ""))

    # 数据集中部分样本的问题文本末尾附带了引导式思考指令（如 "Please think about this
    # question as if you were a human pondering deeply..."），与训练脚本自定义的 prompt
    # 格式重复，在此统一截断。
    _ponder_marker = "Please think about this question as if you were a human pondering deeply"
    if _ponder_marker in question_text:
        question_text = question_text[:question_text.index(_ponder_marker)].rstrip()
    
    prompt_text = prompt_template.format(question=question_text)
    
    video_path = example.get("video", example.get("path", ""))
    
    if video_base_dir and not os.path.isabs(video_path):
        video_path = os.path.join(video_base_dir, video_path)
    
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 1, "max_frames": 32, "max_pixels": 128 * 32 * 32},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "conversations": conversations,
        "temporal_grounding": example.get("temporal_grounding"),
        "video": video_path,
        "id": example.get("id", ""),
    }


def _collect_run_config(
    script_args: "SDPOScriptArguments",
    training_args: "SDPOConfig",
    model_args: "ModelConfig",
) -> dict:
    """
    收集启动脚本中设置的所有参数，返回可序列化的扁平字典。

    参数按来源分组并加前缀：
      script/*   — SDPOScriptArguments（数据集路径、教师上下文、散度配置等）
      model/*    — ModelConfig（模型路径、精度、注意力实现等）
      training/* — SDPOConfig 中启动脚本明确设置的训练超参数
    """

    def _safe(v):
        """将不可 JSON 序列化的值转换为字符串"""
        return v if isinstance(v, (bool, int, float, str, type(None))) else str(v)

    config: Dict[str, Any] = {}

    # --- script_args ---
    for f in dataclasses.fields(script_args):
        config[f"script/{f.name}"] = _safe(getattr(script_args, f.name))

    # --- model_args ---
    for f in dataclasses.fields(model_args):
        config[f"model/{f.name}"] = _safe(getattr(model_args, f.name))

    # --- training_args: 仅记录启动脚本中明确设置的关键超参数 ---
    _training_keys = [
        "output_dir",
        "deepspeed",
        "learning_rate",
        "warmup_ratio",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "bf16",
        "gradient_checkpointing",
        "logging_steps",
        "save_steps",
        "save_only_model",
        "data_seed",
        "dataloader_pin_memory",
        # GRPOConfig / SDPOConfig 专有字段
        "max_prompt_length",
        "max_completion_length",
        "beta",
    ]
    for k in _training_keys:
        config[f"training/{k}"] = _safe(getattr(training_args, k, None))

    return config


class _SwanLabConfigCallback(TrainerCallback):
    """
    训练开始时将所有运行配置写入 SwanLab config。

    SwanLabCallback 会在 on_train_begin 中调用 swanlab.init()；
    本 callback 在其之后执行，因此可安全调用 swanlab.config.update()。
    """

    def __init__(self, config: dict):
        self._config = config

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            import swanlab  # noqa: PLC0415
            swanlab.config.update(self._config)
        except Exception as exc:
            print(f"[SwanLab] 写入 config 失败（不影响训练）: {exc}")


def main(script_args: SDPOScriptArguments, training_args: SDPOConfig, model_args: ModelConfig):
    """
    主函数：初始化并启动 SDPO 训练
    
    参数:
        script_args: 脚本参数
        training_args: 训练参数
        model_args: 模型参数
    """
    print("=" * 60)
    print("SDPO Training Configuration")
    print("=" * 60)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.jsonl_path}")
    print(f"Output: {training_args.output_dir}")
    print(f"Video Base Dir: {script_args.video_base_dir}")
    print("-" * 60)
    print("Teacher Context Configuration:")
    print(f"  - include_answer: {script_args.include_answer}")
    print(f"  - include_temporal_text: {script_args.include_temporal_text}")
    print(f"  - include_temporal_video: {script_args.include_temporal_video}")
    print(f"  - include_reasoning: {script_args.include_reasoning}")
    print(f"  - temporal_fps: {script_args.temporal_fps}")
    print(f"  - temporal_max_frames: {script_args.temporal_max_frames}")
    print(f"  - teacher_temporal_max_pixels: {script_args.teacher_temporal_max_pixels}")
    print(f"  - max_temporal_segments: {script_args.max_temporal_segments}")
    print(f"  - use_fixed_teacher: {script_args.use_fixed_teacher}")
    print(f"  - teacher_model_path: {script_args.teacher_model_path}")
    print("=" * 60)

    teacher_context_config = TeacherContextConfig(
        include_answer=script_args.include_answer,
        include_temporal_text=script_args.include_temporal_text,
        include_temporal_video=script_args.include_temporal_video,
        include_reasoning=script_args.include_reasoning,
        temporal_fps=script_args.temporal_fps,
        temporal_max_frames=script_args.temporal_max_frames,
        temporal_max_pixels=script_args.teacher_temporal_max_pixels,
        point_frames_count=script_args.point_frames_count,
        max_temporal_segments=script_args.max_temporal_segments,
    )
    
    divergence_config = DivergenceConfig(
        method=script_args.divergence_method,
        top_k=script_args.divergence_top_k,
    )
    
    if script_args.jsonl_path:
        dataset = create_dataset_from_json(script_args.jsonl_path)
    else:
        if script_args.dataset_name:
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        else:
            raise ValueError("Either jsonl_path or dataset_name must be provided")
    
    print(f"\nDataset loaded: {len(dataset['train'])} samples")
    
    dataset = dataset.map(
        lambda x: make_conversation_video(x, VIDEO_QA_PROMPT, script_args.video_base_dir),
        remove_columns=dataset["train"].column_names if hasattr(dataset["train"], "column_names") else None
    )
    
    print(f"Dataset processed, sample keys: {dataset['train'][0].keys()}")

    # 随机采样训练数据
    if script_args.max_train_samples is not None:
        train_split = script_args.dataset_train_split
        train_size = len(dataset[train_split])
        n = min(script_args.max_train_samples, train_size)
        if n < train_size:
            dataset[train_split] = dataset[train_split].shuffle(seed=training_args.data_seed).select(range(n))
            print(f"Randomly sampled {n} / {train_size} training samples (seed={training_args.data_seed})")
        else:
            print(f"max_train_samples ({script_args.max_train_samples}) >= dataset size ({train_size}), using full dataset")

    # 将 ModelConfig 的 torch_dtype 传递给 training_args.model_init_kwargs
    # 确保模型以指定精度加载（如 bfloat16），避免 Flash Attention 2 的 float32 警告
    if model_args.torch_dtype is not None:
        if training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {}
        training_args.model_init_kwargs["torch_dtype"] = model_args.torch_dtype

    trainer = Qwen2VLSDPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        teacher_context_config=teacher_context_config,
        divergence_config=divergence_config,
        video_base_dir=script_args.video_base_dir,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        attn_implementation=model_args.attn_implementation,
        use_fixed_teacher=script_args.use_fixed_teacher,
        teacher_model_path=script_args.teacher_model_path,
        callbacks=[_SwanLabConfigCallback(_collect_run_config(script_args, training_args, model_args))],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        print(f"Pushing to hub: {training_args.hub_model_id}")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = TrlParser((SDPOScriptArguments, SDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
