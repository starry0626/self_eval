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
On-Policy Self-Distillation (SDPO) Trainer

本模块实现了基于策略的自蒸馏训练器，用于视觉语言模型的训练。
核心思想：使用训练模型本身作为教师模型，但教师模型接收额外的上下文信息，
通过计算学生模型和教师模型输出分布之间的 Jensen-Shannon 散度作为损失函数。

与 GRPO 的主要区别：
1. 每个样本只采样一次回答（而非多次）
2. 不使用奖励函数和优势计算
3. 不需要独立的参考模型
4. 使用 JS 散度替代 KL 散度
"""

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info
from .teacher_context_builder import TeacherContextBuilder, TeacherContextConfig
from .divergence import DivergenceConfig, compute_reverse_kl

if is_peft_available():
    from peft import PeftConfig, get_peft_model

import importlib.util
from dataclasses import dataclass

def is_swanlab_available():
    """检查 swanlab 库是否可用"""
    return importlib.util.find_spec("swanlab") is not None

if is_swanlab_available():
    import swanlab


@dataclass
class SDPOConfig(GRPOConfig):
    """
    SDPO 训练配置

    继承 GRPOConfig 以复用 max_prompt_length、max_completion_length 等参数，
    但跳过 num_generations >= 2 的验证限制，因为 SDPO 每个样本只生成一次回答。
    """

    def __post_init__(self):
        # 跳过 GRPOConfig.__post_init__ 中的 num_generations 验证
        # SDPO 不需要多次生成，直接调用 TrainingArguments 的初始化
        from transformers import TrainingArguments
        TrainingArguments.__post_init__(self)
        self.num_generations = 1


class Qwen2VLSDPOTrainer(Trainer):
    """
    On-Policy Self-Distillation (SDPO) 训练器
    
    核心思想：使用训练模型本身作为教师模型，但教师模型接收额外的上下文信息。
    损失函数为学生模型和教师模型输出分布之间的 Jensen-Shannon 散度。

    使用示例:

    ```python
    from datasets import load_dataset
    from trainers import Qwen2VLSDPOTrainer

    dataset = load_dataset("your_dataset", split="train")

    trainer = Qwen2VLSDPOTrainer(
        model="Qwen/Qwen2-VL-7B-Instruct",
        train_dataset=dataset,
        teacher_context_field="teacher_context",  # 数据集中包含额外上下文的字段名
    )

    trainer.train()
    ```

    参数说明:
        model (`Union[str, PreTrainedModel]`):
            待训练的模型，可以是：
            - 字符串：huggingface.co 上的模型 ID 或本地路径
            - PreTrainedModel 对象：已实例化的模型
        args ([`GRPOConfig`], *optional*):
            训练配置，如果为 None 则使用默认配置
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            训练数据集，必须包含 "prompt" 列，可选包含 teacher_context_field 指定的列
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or dict):
            评估数据集，要求与训练数据集相同
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            数据处理器，padding 侧必须设置为 "left"
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            自定义回调函数列表
        optimizers (`tuple`, *optional*):
            优化器和学习率调度器的元组
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT 配置，用于参数高效微调
        teacher_context_field (`str`, *optional*, defaults to `"teacher_context"`):
            数据集中包含教师模型额外上下文的字段名
            该上下文会在计算教师概率时注入到 prompt 中
        teacher_context_config (`TeacherContextConfig`, *optional*):
            教师上下文配置，控制额外输入的各部分是否包含
            如果为 None，则使用默认配置（包含答案、时间定位文本、推理流程）
        divergence_config (`DivergenceConfig`, *optional*):
            散度计算配置，控制散度类型和估计方法
            - method: "full" (完整计算), "top_k" (top-k 估计), "k3" (K3 估计)
            - top_k: top-k 的 k 值（仅当 method="top_k" 时有效）
            - epsilon: 数值稳定性常数
            如果为 None，则使用默认配置（完整计算）
        video_base_dir (`str`, *optional*):
            视频文件基础目录，用于视频帧采样
        max_pixels (`int`, *optional*, defaults to 12845056):
            图像/视频处理的最大像素数
        min_pixels (`int`, *optional*, defaults to 3136):
            图像/视频处理的最小像素数
        attn_implementation (`str`, *optional*, defaults to `"flash_attention_2"`):
            注意力机制的实现方式
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: SDPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        teacher_context_field: str = "teacher_context",
        teacher_context_config: Optional[TeacherContextConfig] = None,
        divergence_config: Optional[DivergenceConfig] = None,
        video_base_dir: Optional[str] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        use_fixed_teacher: bool = False,
        teacher_model_path: Optional[str] = None,
    ):
        # ==================== 参数初始化 ====================
        # 如果未提供配置，则根据模型名创建默认配置
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SDPOConfig(f"{model_name}-SDPO")

        # ==================== 模型加载 ====================
        # 获取模型初始化参数
        model_init_kwargs = args.model_init_kwargs or {}
        # 设置注意力机制实现方式
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            # 模型以字符串形式提供，需要从预训练权重加载
            model_id = model
            
            # 处理 torch_dtype 参数
            # 支持多种输入格式：torch.dtype、字符串（如 "float32"）、"auto" 或 None
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # 已经是正确格式，无需处理
            elif isinstance(torch_dtype, str):
                # 将字符串转换为对应的 torch.dtype
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            # 梯度检查点与 KV cache 不兼容，需要禁用 use_cache
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            # 根据模型类型加载对应的模型类
            # 不同版本的 Qwen-VL 模型有不同的实现类和配置结构
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen3-VL" in model_id:
                # Qwen3-VL 的 use_cache 参数位于 text_config 中，需要特殊处理
                model_init_kwargs.pop("use_cache", None)
                model = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                model.config.text_config.use_cache = (False if args.gradient_checkpointing else True)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                # 默认使用 AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            # 模型已经实例化
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        
        self.model_id = model_id
        
        # 如果提供了 PEFT 配置，将模型包装为 PEFT 模型
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # ==================== 固定教师模型加载（可选）====================
        # 当 use_fixed_teacher=True 时，加载独立的固定教师模型（参数不更新）
        # 教师模型使用与学生相同的模型类型和精度，但不启用梯度检查点
        self._fixed_teacher_model = None
        if use_fixed_teacher:
            teacher_id = teacher_model_path if teacher_model_path else model_id
            _teacher_kwargs = {"attn_implementation": attn_implementation}
            _teacher_dtype = model_init_kwargs.get("torch_dtype")
            if _teacher_dtype is None:
                _teacher_dtype = next(model.parameters()).dtype
            if _teacher_dtype is not None and _teacher_dtype != "auto":
                _teacher_kwargs["torch_dtype"] = _teacher_dtype

            if "Qwen2-VL" in teacher_id:
                _teacher = Qwen2VLForConditionalGeneration.from_pretrained(teacher_id, **_teacher_kwargs)
            elif "Qwen2.5-VL" in teacher_id:
                _teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(teacher_id, **_teacher_kwargs)
            elif "Qwen3-VL" in teacher_id:
                _teacher = Qwen3VLForConditionalGeneration.from_pretrained(teacher_id, **_teacher_kwargs)
            elif "Aria" in teacher_id:
                _teacher = AriaForConditionalGeneration.from_pretrained(teacher_id, **_teacher_kwargs)
            else:
                _teacher = AutoModelForCausalLM.from_pretrained(teacher_id, **_teacher_kwargs)

            for param in _teacher.parameters():
                param.requires_grad = False
            _teacher.eval()
            self._fixed_teacher_model = _teacher

        # ==================== 处理器加载 ====================
        # 处理器用于将文本和视觉信息转换为模型输入
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id or "Aria" in model_id:
                # 多模态模型使用 AutoProcessor
                processing_class = AutoProcessor.from_pretrained(model_id)
                # 同步 tokenizer 的属性到 processor
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # 设置图像/视频处理的分辨率范围
                if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                # 纯文本模型使用 AutoTokenizer
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # ==================== 数据整理器 ====================
        # SDPO 不需要特殊的数据整理，直接返回原始特征
        def data_collator(features):
            return features

        # ==================== 训练参数设置 ====================
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        
        # 生成配置：每个样本只生成一次回答
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=1,  # 与 GRPO 的关键区别：只生成一次
            pad_token_id=pad_token_id,
        )
        
        # 教师上下文字段名，用于从数据集中获取额外信息(用于直接从数据集中提取教师额外信息的情况下)
        self.teacher_context_field = teacher_context_field
        
        # 教师上下文配置和构建器
        self.teacher_context_config = teacher_context_config or TeacherContextConfig()
        self.teacher_context_builder = TeacherContextBuilder(self.teacher_context_config)
        
        # 散度计算配置
        self.divergence_config = divergence_config or DivergenceConfig()
        
        self.video_base_dir = video_base_dir

        # 计算生成提示（generation prompt）的 token 长度
        # 用于在拼接教师输入时正确地将 generation prompt 移至额外上下文之后
        _tokenizer = getattr(processing_class, 'tokenizer', processing_class)
        _dummy_msg = [{"role": "user", "content": "a"}]
        _text_with = processing_class.apply_chat_template(
            _dummy_msg, tokenize=False, add_generation_prompt=True
        )
        _text_without = processing_class.apply_chat_template(
            _dummy_msg, tokenize=False, add_generation_prompt=False
        )
        _ids_with = _tokenizer.encode(_text_with, add_special_tokens=False)
        _ids_without = _tokenizer.encode(_text_without, add_special_tokens=False)
        self._generation_prompt_len = len(_ids_with) - len(_ids_without)

        # 抑制 token 数量估计警告
        model.warnings_issued["estimate_tokens"] = True

        # 初始化指标记录字典
        self._metrics = defaultdict(list)

        # 调用父类初始化
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # 禁用损失相关的 kwargs 检查，因为我们自定义了损失计算
        self.model_accepts_loss_kwargs = False

        # 将固定教师模型移至训练设备
        if self._fixed_teacher_model is not None:
            self._fixed_teacher_model = self._fixed_teacher_model.to(self.accelerator.device)

    def _set_signature_columns_if_needed(self):
        """
        设置签名列，防止 Trainer 移除我们需要的列
        
        当 remove_unused_columns=True 时，Trainer 会移除非签名列。
        我们需要保留 prompt、teacher_context_field 指定的列，以及 TeacherContextBuilder 需要的列。
        """
        if self.args.remove_unused_columns:
            self._signature_columns = [
                "prompt", 
                self.teacher_context_field,
                "conversations",
                "temporal_grounding",
                "video"
            ]

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        准备输入数据（直接返回，不做处理）
        
        SDPO 的数据处理在 compute_loss 中完成，这里直接返回原始输入。
        """
        return inputs

    def _inject_teacher_context(self, prompt_messages: list, teacher_context: Any) -> list:
        """
        将教师上下文注入到 prompt 消息中（简单注入模式）
        
        这是 SDPO 的核心方法之一，用于构建教师模型的输入。
        教师模型接收与学生模型相同的问题，但包含额外的上下文信息。
        
        此方法用于处理数据集中已预先构建好的 teacher_context 字段。
        对于需要从原始数据构建教师上下文的情况，请使用 _build_teacher_prompt 方法。
        
        参数:
            prompt_messages: 原始 prompt 消息列表
                格式: [{"role": "user", "content": [...]}]
            teacher_context: 教师上下文，支持多种格式:
                - 字符串: 纯文本上下文
                - 列表: 结构化内容列表，如 [{"type": "text", "text": "..."}]
                - 字典: 包含 text、image、video 等字段的结构化数据
        
        返回:
            修改后的 prompt 消息列表，包含注入的上下文
        """
        # 深拷贝避免修改原始数据
        modified_messages = copy.deepcopy(prompt_messages)
        
        # 如果没有教师上下文，直接返回原始消息
        if teacher_context is None:
            return modified_messages
        
        # 根据教师上下文的类型进行不同的注入处理
        if isinstance(teacher_context, str):
            # 字符串类型：作为额外文本追加到用户消息末尾
            for msg in modified_messages:
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), list):
                        # content 是列表格式，追加文本元素
                        msg["content"].append({
                            "type": "text",
                            "text": f"\n\nAdditional Context:\n{teacher_context}"
                        })
                    else:
                        # content 是字符串格式，直接拼接
                        msg["content"] = msg["content"] + f"\n\nAdditional Context:\n{teacher_context}"
                    break
                    
        elif isinstance(teacher_context, list):
            # 列表类型：直接扩展到 content 列表中
            for msg in modified_messages:
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), list):
                        msg["content"].extend(teacher_context)
                    else:
                        # 将字符串 content 转换为列表格式
                        msg["content"] = [{"type": "text", "text": msg["content"]}] + teacher_context
                    break
                    
        elif isinstance(teacher_context, dict):
            # 字典类型：按字段分别注入
            for msg in modified_messages:
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), list):
                        # 注入文本上下文
                        if "text" in teacher_context:
                            msg["content"].append({"type": "text", "text": teacher_context["text"]})
                        # 注入图像上下文
                        if "image" in teacher_context:
                            msg["content"].append({"type": "image", "image": teacher_context["image"]})
                        # 注入视频上下文
                        if "video" in teacher_context:
                            msg["content"].append({"type": "video", "video": teacher_context["video"]})
                    break
        
        return modified_messages
    
    def _build_teacher_prompt_from_raw_data(
        self,
        prompt_messages: list,
        conversations: list,
        temporal_grounding: Optional[dict] = None,
        video_path: Optional[str] = None
    ) -> list:
        """
        从原始数据构建教师模型的输入 prompt（高级构建模式）
        
        使用 TeacherContextBuilder 从原始数据构建教师上下文，支持：
        - 标准答案提取
        - 时间定位文本
        - 时间定位视频片段采样
        - 推理流程提取
        
        参数:
            prompt_messages: 原始 prompt 消息列表
            conversations: 对话列表，包含问题和回答
            temporal_grounding: 时间定位字典（可选）
            video_path: 视频文件路径（可选）
        
        返回:
            教师模型的输入消息列表
        """
        return self.teacher_context_builder.build_teacher_prompt(
            original_prompt_messages=prompt_messages,
            conversations=conversations,
            temporal_grounding=temporal_grounding,
            video_path=video_path,
            video_base_dir=self.video_base_dir
        )
    
    def _build_extra_context_from_prebuilt(self, teacher_context: Any) -> list:
        """
        从预构建的 teacher_context 构建额外上下文消息
        
        参数:
            teacher_context: 预构建的教师上下文（可以是字符串、列表或字典）
        
        返回:
            额外上下文消息列表
        """
        if teacher_context is None:
            return []
        
        extra_content = []
        
        if isinstance(teacher_context, str):
            extra_content.append({"type": "text", "text": teacher_context})
        elif isinstance(teacher_context, list):
            extra_content.extend(teacher_context)
        elif isinstance(teacher_context, dict):
            if "text" in teacher_context:
                extra_content.append({"type": "text", "text": teacher_context["text"]})
            if "image" in teacher_context:
                extra_content.append({"type": "image", "image": teacher_context["image"]})
        
        if not extra_content:
            return []
        
        return [{"role": "user", "content": extra_content}]
    
    def _concat_student_and_extra_inputs(
        self,
        student_input_ids: torch.Tensor,
        student_attention_mask: torch.Tensor,
        student_prompt_inputs: dict,
        extra_input_ids: torch.Tensor,
        extra_attention_mask: torch.Tensor,
        extra_prompt_inputs: dict,
        pad_token_id: int,
        generation_prompt_len: int = 0,
        per_sample_has_extra: list = None,
    ) -> dict:
        """
        拼接学生输入和额外上下文

        处理流程：
        1. 去除学生输入的填充（根据 attention_mask）
        2. 去除额外输入的填充
        3. 将 generation prompt 从学生输入末尾移至额外上下文之后
        4. 拼接 input_ids 和 attention_mask
        5. 拼接视觉信息
        6. 重新填充到 batch 内最大长度

        参数:
            student_input_ids: 学生输入的 input_ids (B, L1)
            student_attention_mask: 学生输入的 attention_mask (B, L1)
            student_prompt_inputs: 学生输入的完整字典（包含视觉信息）
            extra_input_ids: 额外输入的 input_ids (B, L2)
            extra_attention_mask: 额外输入的 attention_mask (B, L2)
            extra_prompt_inputs: 额外输入的完整字典
            pad_token_id: 填充 token ID
            generation_prompt_len: generation prompt 的 token 长度，用于重排拼接顺序
            per_sample_has_extra: 每个样本是否有额外上下文的布尔列表

        返回:
            拼接后的教师输入字典
        """
        batch_size = student_input_ids.size(0)
        device = student_input_ids.device

        # 存储每个样本的拼接结果
        concatenated_input_ids = []
        concatenated_attention_mask = []

        for i in range(batch_size):# batch内每条数据分开处理
            # 去除学生输入的填充
            student_mask = student_attention_mask[i].bool() #mask矩阵(0,1)转化为bool阵
            student_ids_no_pad = student_input_ids[i][student_mask] #利用bool索引保留有效token

            # 检查该样本是否需要构建给教师模型的额外上下文信息
            has_extra = per_sample_has_extra[i] if per_sample_has_extra is not None else True

            if has_extra:
                # 去除额外输入的填充
                extra_mask = extra_attention_mask[i].bool()
                extra_ids_no_pad = extra_input_ids[i][extra_mask]

                if len(extra_ids_no_pad) > 0 and generation_prompt_len > 0:
                    # 将LLM生成所需要的引导符(Assistant:)从学生末尾移至额外上下文之后
                    # 原顺序：[student_question | gen_prompt]
                    # 目标顺序：[student_question | extra_context | gen_prompt]
                    student_main = student_ids_no_pad[:-generation_prompt_len]
                    gen_prompt = student_ids_no_pad[-generation_prompt_len:]
                    concat_ids = torch.cat([student_main, extra_ids_no_pad, gen_prompt], dim=0)
                else:
                    concat_ids = torch.cat([student_ids_no_pad, extra_ids_no_pad], dim=0)
            else:
                # 该样本无额外上下文，保留学生输入原样
                concat_ids = student_ids_no_pad
             #  为拼接后的新序列生成全新的 attention mask（全是 1，因为目前没有 Pad）
            concat_mask = torch.ones(concat_ids.size(0), dtype=torch.long, device=device)

            concatenated_input_ids.append(concat_ids)
            concatenated_attention_mask.append(concat_mask)

        # 找到添加额外信息后的最大长度
        max_length = max(ids.size(0) for ids in concatenated_input_ids)

        # 重新填充到最大长度（左侧填充）
        padded_input_ids = []
        padded_attention_mask = []

        for i in range(batch_size):
            ids = concatenated_input_ids[i]
            mask = concatenated_attention_mask[i]
            seq_len = ids.size(0)
            pad_len = max_length - seq_len #需要填充的padding的数量

            if pad_len > 0:
                # 左侧填充
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=device)
                pad_mask = torch.zeros(pad_len, dtype=mask.dtype, device=device)

                ids = torch.cat([pad_ids, ids], dim=0)
                mask = torch.cat([pad_mask, mask], dim=0)

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)

        # 堆叠成 batch
        teacher_input_ids = torch.stack(padded_input_ids, dim=0)
        teacher_attention_mask = torch.stack(padded_attention_mask, dim=0)
        
        # 构建教师输入字典
        teacher_prompt_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
        }
        
        # 处理视觉信息
        if "pixel_values_videos" in extra_prompt_inputs or"video_grid_thw" in extra_prompt_inputs:
            raise ValueError("不支持学生信息与额外信息均包含视频视觉信息")
        # pixel_values_videos: 不包含 batch 维度，直接复制
        if "pixel_values_videos" in student_prompt_inputs:
            teacher_prompt_inputs["pixel_values_videos"] = student_prompt_inputs["pixel_values_videos"]
        
        # video_grid_thw: 包含 batch 维度，直接复制
        if "video_grid_thw" in student_prompt_inputs:
            teacher_prompt_inputs["video_grid_thw"] = student_prompt_inputs["video_grid_thw"]
        
        # pixel_values: 图像像素值，需要拼接
        # 注意：额外上下文可能包含图像帧
        if "pixel_values" in extra_prompt_inputs:
            # 额外上下文有图像
            if "pixel_values" in student_prompt_inputs:
                # # 学生输入也有图像，需要拼接
                # # pixel_values 的形状是 (total_pixels, channels)，没有 batch 维度
                # # 需要按照样本分别处理
                # teacher_prompt_inputs["pixel_values"] = torch.cat([
                #     student_prompt_inputs["pixel_values"],
                #     extra_prompt_inputs["pixel_values"]
                # ], dim=0)
                raise ValueError("不支持学生信息与额外信息均包含图像视觉信息")
            else:
                teacher_prompt_inputs["pixel_values"] = extra_prompt_inputs["pixel_values"]
        elif "pixel_values" in student_prompt_inputs:
            teacher_prompt_inputs["pixel_values"] = student_prompt_inputs["pixel_values"]
        
        # image_grid_thw: 图像网格信息
        if "image_grid_thw" in extra_prompt_inputs:
            if "image_grid_thw" in student_prompt_inputs:
                # teacher_prompt_inputs["image_grid_thw"] = torch.cat([
                #     student_prompt_inputs["image_grid_thw"],
                #     extra_prompt_inputs["image_grid_thw"]
                # ], dim=0)
                raise ValueError("不支持学生信息与额外信息均包含图像视觉信息")
            else:
                teacher_prompt_inputs["image_grid_thw"] = extra_prompt_inputs["image_grid_thw"]
        elif "image_grid_thw" in student_prompt_inputs:
            teacher_prompt_inputs["image_grid_thw"] = student_prompt_inputs["image_grid_thw"]
        
        return teacher_prompt_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失函数 - SDPO 的核心方法
        
        整体流程：
        1. 准备学生模型的输入（原始 prompt）
        2. 生成回答（每个样本一次）
        3. 计算学生模型的对数概率
        4. 构建教师模型输入（注入额外上下文）
        5. 计算教师模型的对数概率（无梯度）
        6. 计算 JS 散度作为损失
        
        参数:
            model: 训练模型
            inputs: 输入数据，包含 prompt 和可选的 teacher_context
            return_outputs: 是否返回输出（SDPO 不支持）
            num_items_in_batch: batch 中的项目数
        
        返回:
            loss: 计算得到的损失值
        """
        if return_outputs:
            raise ValueError("The SDPOTrainer does not support returning outputs")

        model_id = self.model_id
        
        # ==================== 步骤 1: 准备学生模型输入 ====================
        # 提取 prompt 消息和教师上下文
        prompts_messages = [x["prompt"] for x in inputs]
        teacher_contexts = [x.get(self.teacher_context_field) for x in inputs]
        
        # 递归清理字典中的 None 值，消除 Arrow Schema 对齐带来的副作用
        def clean_none_values(obj):
            if isinstance(obj, list):
                return [clean_none_values(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: clean_none_values(v) for k, v in obj.items() if v is not None}
            return obj
        
        prompts_messages = clean_none_values(prompts_messages)

        # 使用官方工具处理视觉信息（视频加载、采样等）
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prompts_messages, 
            return_video_kwargs=True,
            return_video_metadata=True if "Qwen3-VL" in model_id else False
        )

        # 应用 chat template 生成文本
        texts = [
            self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompts_messages
        ]
        
        # 使用 processor 生成 input_ids 和 pixel_values
        if "Qwen3-VL" in model_id:
            # Qwen3-VL 需要额外的 metadata 参数（fps, index 等）
            videos, video_metadatas = zip(*video_inputs) if video_inputs else ([], [])
            videos, video_metadata = list(videos), list(video_metadatas)
            prompt_inputs = self.processing_class(
                text=texts,
                images=image_inputs,
                videos=videos if videos else None,
                video_metadata=video_metadata if video_metadata else None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                **video_kwargs 
            )
        else:
            prompt_inputs = self.processing_class(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                **video_kwargs
            )
        
        # 将输入移动到设备
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # 截断过长的 prompt
        # 使用左侧截断, 即去掉开始的部分, 保留结尾部分. 但若是开始的视觉信息以及prompt就被截断了这个样本起到的效果多半也是负面的(!!!!!!!!!有待提升)
        # 如果文本中的视觉占位符被截掉但视觉token数量没变是否会报错
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # 备份 input_ids 和 attention_mask（用于后续教师输入拼接）
        student_input_ids_backup = prompt_inputs["input_ids"].clone()
        student_attention_mask_backup = prompt_inputs["attention_mask"].clone()

        # ==================== 步骤 2: 生成回答 ====================
        # 使用学生模型生成回答（每个样本只生成一次）
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

        # 计算各部分的长度
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_only_ids = completion_ids[:, prompt_length:]

        # 构建学生模型完整的 attention_mask（prompt + completion）
        student_completion_attention_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones(
                (prompt_inputs["attention_mask"].size(0), completion_ids.size(1) - prompt_length),
                dtype=prompt_inputs["attention_mask"].dtype,
                device=prompt_inputs["attention_mask"].device,
            )
        ], dim=1)

        # 移除已处理的输入，保留视觉信息用于后续前向传播
        # input_ids已经包含在completion_ids中, 因而前向传播不再需要
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        # ==================== 步骤 3: 计算学生模型的对数概率 ====================
        def get_logits(model, input_ids, **kwargs):
            """
            模型前向传播，获取 logits

            参数:
                model: 模型
                input_ids: 输入 token IDs
                **kwargs: 其他模型输入（如 pixel_values）

            返回:
                logits: 原始 logits，shape (B, L-1, V)
            """
            outputs = model(input_ids, **kwargs)
            # 去掉最后一个位置（它预测的是下一个 token，但我们没有对应的 label）
            logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
            del outputs
            return logits

        # 计算学生模型的 logits
        student_logits = get_logits(
            model, completion_ids, attention_mask=student_completion_attention_mask, **prompt_inputs
        )
        # 只保留生成部分
        student_logits = student_logits[:, prompt_length - 1:, :]

        # ==================== 步骤 4: 构建教师模型输入（优化版）====================
        # 优化策略：只处理额外信息，然后与学生输入拼接，避免重复处理视频数据
        
        # 保存学生输入的原始数据（用于后续拼接）
        student_input_ids = student_input_ids_backup
        student_attention_mask = student_attention_mask_backup
        
        # 构建额外上下文消息
        extra_contexts_messages = []
        for i, x in enumerate(inputs):
            has_raw_data = "conversations" in x or "temporal_grounding" in x
            
            if has_raw_data:
                # 提取问题文本
                question_text = ""
                for item in prompts_messages[i][0].get("content", []):
                    if item.get("type") == "text":
                        question_text = item.get("text", "")
                        break
                
                # 只构建额外上下文（不包含原始视频和问题）
                extra_msg = self.teacher_context_builder.build_extra_context_only(
                    conversations=x.get("conversations", []),
                    question_text=question_text,
                    temporal_grounding=x.get("temporal_grounding"),
                    video_path=x.get("video"),
                    video_base_dir=self.video_base_dir
                )
            else:
                # 简单注入模式：构建额外上下文
                extra_msg = self._build_extra_context_from_prebuilt(teacher_contexts[i])
            
            extra_contexts_messages.append(extra_msg)
        
        extra_contexts_messages = clean_none_values(extra_contexts_messages)
        
        # 检查是否有额外上下文（per-sample）
        per_sample_has_extra = [
            len(msg) > 0 and len(msg[0].get("content", [])) > 0
            for msg in extra_contexts_messages
        ]
        has_extra_context = any(per_sample_has_extra)
        
        if not has_extra_context:
            # 没有额外上下文，教师输入与学生输入相同
            teacher_prompt_inputs = {
                "input_ids": student_input_ids.clone() if student_input_ids is not None else None,
                "attention_mask": student_attention_mask.clone() if student_attention_mask is not None else None,
            }
            # 复制视觉信息
            for key in ["pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"]:
                if key in prompt_inputs:
                    teacher_prompt_inputs[key] = prompt_inputs[key].clone() if isinstance(prompt_inputs[key], torch.Tensor) else prompt_inputs[key]
            
            teacher_prompt_length = prompt_length
        else:
            # 有额外上下文，需要处理并拼接
            # 处理额外上下文的视觉信息（只有图像帧，没有视频）
            extra_image_inputs, extra_video_inputs, extra_video_kwargs = process_vision_info(
                extra_contexts_messages,
                return_video_kwargs=True,
                return_video_metadata=True if "Qwen3-VL" in model_id else False
            )
            
            # 应用 chat template 生成额外文本
            extra_texts = [
                self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in extra_contexts_messages
            ]
            
            # 处理额外文本和图像
            if "Qwen3-VL" in model_id:
                extra_prompt_inputs = self.processing_class(
                    text=extra_texts,
                    images=extra_image_inputs,
                    videos=None,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                )
            else:
                extra_prompt_inputs = self.processing_class(
                    text=extra_texts,
                    images=extra_image_inputs,
                    videos=None,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                )
            
            extra_prompt_inputs = super()._prepare_inputs(extra_prompt_inputs)
            
            # ==================== 拼接学生输入和额外上下文 ====================
            teacher_prompt_inputs = self._concat_student_and_extra_inputs(
                student_input_ids=student_input_ids,
                student_attention_mask=student_attention_mask,
                student_prompt_inputs=prompt_inputs,
                extra_input_ids=extra_prompt_inputs.get("input_ids"),
                extra_attention_mask=extra_prompt_inputs.get("attention_mask"),
                extra_prompt_inputs=extra_prompt_inputs,
                pad_token_id=self.processing_class.tokenizer.pad_token_id,
                generation_prompt_len=self._generation_prompt_len,
                per_sample_has_extra=per_sample_has_extra,
            )
            
            teacher_prompt_length = teacher_prompt_inputs["input_ids"].size(1)
        
        # 截断教师输入
        if self.max_prompt_length is not None:
            teacher_prompt_inputs["input_ids"] = teacher_prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            teacher_prompt_inputs["attention_mask"] = teacher_prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
            teacher_prompt_length = min(teacher_prompt_length, self.max_prompt_length)

        # ==================== 步骤 5: 计算教师模型的对数概率 ====================
        # 教师模型使用相同的生成结果，但输入包含额外上下文
        # 构建教师完整输入：[teacher_prompt | completion_only]
        teacher_input_ids = torch.cat([
            teacher_prompt_inputs["input_ids"],
            completion_only_ids
        ], dim=1)

        # 构建教师 attention_mask：使用教师 prompt 的 mask + 生成部分全 1
        teacher_attention_mask = torch.cat([
            teacher_prompt_inputs["attention_mask"],
            torch.ones_like(completion_only_ids)
        ], dim=1)

        # 移除已处理的输入（视觉信息保留在 teacher_prompt_inputs 中）
        teacher_prompt_inputs.pop("input_ids", None)
        teacher_prompt_inputs.pop("attention_mask", None)

        # 计算教师模型的 logits（无梯度）
        # 如果存在固定教师模型则使用固定教师，否则使用当前训练模型
        teacher_model_for_forward = self._fixed_teacher_model if self._fixed_teacher_model is not None else model
        with torch.inference_mode():
            teacher_logits = get_logits(
                teacher_model_for_forward, teacher_input_ids, attention_mask=teacher_attention_mask, **teacher_prompt_inputs
            )
        # 只保留生成部分
        teacher_logits = teacher_logits[:, teacher_prompt_length - 1:, :]

        # ==================== 步骤 6: 对齐序列长度 ====================
        # 由于 prompt 长度可能不同，生成的 logits 序列长度也可能不同
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
        completion_only_ids = completion_only_ids[:, :min_seq_len]

        # ==================== 步骤 7: 创建 EOS 掩码 ====================
        # 创建 EOS token 后的掩码，只计算有效 token 的损失
        is_eos = completion_only_ids == self.processing_class.eos_token_id
        device = self.accelerator.device

        # 找到每个序列中 EOS token 的位置
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        # 创建序列索引
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)

        # 最终的 completion_mask：EOS 及之前的 token 为 1，之后为 0
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).float()

        # ==================== 步骤 8: 计算反向 KL 散度损失 ====================
        # 使用配置的方法计算 per-token KL 散度
        # KL(teacher || student) = Σ teacher(x) * log(teacher(x) / student(x))
        kl_per_token = compute_reverse_kl(
            logits_student=student_logits,
            logits_teacher=teacher_logits,
            config=self.divergence_config,
            mask=completion_mask,
            sampled_tokens=completion_only_ids,
            reduce=False,
        )  # 返回 (B, L)，已乘以 completion_mask

        # 计算平均损失：先在序列维度平均（除以有效 token 数），再在 batch 维度平均
        loss = (kl_per_token.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

        # ==================== 步骤 9: 记录指标 ====================
        # 记录平均回答长度
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # 记录平均 KL 散度
        mean_kl_divergence = (kl_per_token.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        self._metrics["kl_divergence"].append(
            self.accelerator.gather_for_metrics(mean_kl_divergence).mean().item()
        )

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        记录训练日志
        
        将收集的指标平均后添加到日志中。
        """
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        创建模型卡片
        
        生成包含训练信息的 README.md 文件。
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{self_distillation,
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=get_comet_experiment_url(self.accelerator),
            trainer_name="On-Policy Self-Distillation",
            trainer_citation=citation,
            paper_title="On-Policy Self-Distillation for Vision-Language Models",
            paper_id="self-distillation-vlm",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
