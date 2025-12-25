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

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """
    # 继承于transformers的trainer库
    # 重写了损失函数的计算, 多模态数据的处理, 多模型的管理(参考模型)等功能
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        #模型的初始化参数
        model_init_kwargs = args.model_init_kwargs or {}
        #设置attention的实现方式(flash_attention) !!!!!!!!可能需要进行npu适配
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str): #判断model是否为字符串
            model_id = model
            # 设置参数格式 !!!!!!!!!!!!!!npu
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                # 如果已经是torch.dtype的格式, 或者为自动处理或不指定, 则无需进行处理
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                # 如果为字符串则需要修改为对应的torch.dtype的格式
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # 若开启梯度检查点则禁用KVcache(不支持) ???????????待详细了解
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            # 加载模型, 对于特定的qwenvl模型要加载特定的模型类???????具体区别
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen3-VL" in model_id:
                # "use_cache"参数会尝试用于覆盖config, 但Qwen3-VLconfig的该参数并不位于顶层字段, 无法正确被覆盖. 因此这里初始化后进行手动设置
                model_init_kwargs.pop("use_cache", None)
                model = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                model.config.text_config.use_cache = (False if args.gradient_checkpointing else True)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            # 如果模型已经被加载
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # 加载参考模型(GRPO中计算KL散度的参考)
        if is_deepspeed_zero3_enabled():
            # deepspeed_zero3会对模型参数进行切分, 我们需要加载一份完整的模型作为参考模型
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen3-VL" in model_id:
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                self.ref_model.config.text_config.use_cache = (False if args.gradient_checkpointing else True)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # 加载processor ????????????
        # processor似乎与加载模型时不同, 加载时只传入模型名即可, 加载后再修改参数. 还是说因为需要修改的参数少所以才这么做????????
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                # 手动同步属性, 是否必须????????
                # 将tokenlizer对应的pad_token与eos_token的id属性赋给processor
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id:
                    # 手动设置分辨率
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class(使用模型作为奖励函数)
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

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

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # 1. 准备 Prompts (Qwen-VL-Utils 需要原始的消息结构)
        # inputs 是一个 list of dicts (from dataset)
        # 这里的 "prompt" 键包含我们在 make_conversation_video 中构建的 [{"role": "user", "content": [...]}]
        prompts_messages = [x["prompt"] for x in inputs]
        
        # 2. 处理视觉信息 (使用官方工具)
        # process_vision_info 会处理视频加载、采样等
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prompts_messages, 
            return_video_kwargs=True
        )
        
        # 3. 使用 Processor 生成 input_ids 和 pixel_values
        # Apply chat template specifically for the prompt text structure
        texts = [
            self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompts_messages
        ]
        # qwen3-vl的兼容, 额外的metadata参数
        if"Qwen3-VL" in model_id:
            videos, video_metadatas = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadatas)

            prompt_inputs = self.processing_class(
            text=texts,
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata,# 传入 fps, index等参数
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
                **video_kwargs # 传入 fps, duration 等参数
            )
        
        # 准备 inputs (move to device)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # Truncate logic (from original code)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # 生成GRPO的采样
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
            num_generations = self.generation_config.num_return_sequences
            # copy一份num_return_sequences=1的配置, 每次只生成一份
            # 为什么不能并行? 效率是否过低?
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1

            all_completions = []

            # 循环生成num_generations份回答
            for i in range(num_generations):  # -1 because we already have one generation
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                all_completions.append(completion)

            # 计算最大的回答长度
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            # 填充长度较短的回答
            for completion in all_completions:
                if completion.size(1) < max_length:

                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)), #(batch, 缺的部分)
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            # 在dim0拼接起来, 结果是(B * G, Sequence_Length).
            # 顺序为[Batch1_Gen1, Batch2_Gen1, ..., Batch1_GenG, Batch2_GenG]
            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        prompt_length = prompt_inputs["input_ids"].size(1) #计算prompt长度, 因为prompt_inputs是被padding过的, 所以所有batch长度一致
        completion_ids = prompt_completion_ids[:, prompt_length:] #提取纯回答部分

        # --- 关键修改：处理 input tensors 的重复 (Replication) ---
        # GRPO 生成了 N 个样本，因此视觉特征(pixel_values等)需要重复 N 次以匹配 batch size
        
        # prompt_inputs本质上是一个tensor组成的字典
        prompt_inputs.pop("input_ids") #去掉分词后的文本输入(包括视觉token占位符与文字时间戳)
        prompt_inputs.pop("attention_mask") #去掉对padding的注意力掩码tensor ?????????


        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, **kwargs):
            # input_ids传入模型的的输出prompt_completion_ids(本质上时token_id. padding, dim0为拼接后的batch和group)
            # 剩余参数传入的是prompt_inputs
            # .logits是模型输出层预测的未归一化的分数(batch(这里是batch*group), 序列长, 词表大小)
            logits = model(input_ids, **kwargs).logits  #(B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), 去掉最后一个词的概率, 这预测的是下一个词
            input_ids = input_ids[:, 1:]  # (B, L-1), 预测出的每个词表内所有词的分数对应的应该取的那个词(训练模型采样出来的词)
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            # 遍历dim0, 逐个进行计算(防止爆显存)
            for logits_row, input_ids_row in zip(logits, input_ids):
                # logits_row(L-1, V), input_ids_row(V)
                log_probs = logits_row.log_softmax(dim=-1) #最后一个维度转化为对数概率分布
                # torch.gather()要求索引tensor与被收集tensor的维度数一致, 这里先通过unsqueeze给索引增加一个维度
                # torch.gather() 作用的维度不会消失, 这里用squeeze(1)把多余的那个维度去掉, 最终维度(L-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)# (B, L-1)

        if "video" in inputs[0]:
            # 这里有问题, 视觉信息不应该复制len(prompt_completion_ids)=B*G份, 而只应该复制G份, 这使得代码只适配于batch=1的情况.
            # prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            # prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(num_generations, 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(num_generations, 1)
        
        
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        # 去掉prompt, 只保留生成部分的概率
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            # 计算参考模型的概率, 启动模型的推理模式以节省显存
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs) # Fix Bug
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs) # Fix Bug
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :] # 去掉prompt, 只保留生成部分的概率

        # Compute the KL divergence between the model and the reference model
        diff = ref_per_token_logps - per_token_logps
        diff = torch.clamp(diff, min=-11.0, max=11.0) 
        per_token_kl = torch.exp(diff) - (diff) - 1

        # 将EOS token后的token进行掩码
        is_eos = completion_ids == self.processing_class.eos_token_id #(batch*group, seq). bool阵, EOS token处为True, 其余False.
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) #(batch*group,), 用最大序列长度初始化一个EOS的位置矩阵
        # 对于包含EOS_tpken的行用EOS_tpken来更新前面初始化好的EOS的位置矩阵
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 创建一个表示“当前是第几个Token”的矩阵。
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        #最终的attention_mask矩阵(padding为0, 实际内容1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 对生成的的回答进行解码
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # 将prompt复制group次, 顺序和答案的顺序对不上???????
        # ===========这里也存在顺序问题, 这里的顺序是1个样本的所有采样在一起
        prompts = [prompt for prompt in prompts_messages for _ in range(self.num_generations)]
        # 初始化奖励矩阵(batch*group, 奖励函数个数)
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # import pdb; pdb.set_trace()
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]): # true
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                # 这里的inputs是从数据集中读入的数据, 其中应该包含标准答案
                # 这里将input中的所有key除了"prompt" and "completion"初始化一个字典
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # 先将batch内的每个样本复制num_generations份, 再依次添加到 reward_kwargs中
                        # ===========这里也存在顺序问题, 这里的顺序是1个样本的所有采样在一起
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)# (B*G)

        # Compute grouped-wise rewards
        # ========.view()对原始的tensor按行来切分, 这里也是对应1个样本的所有采样在一起的顺序
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) #(B)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) #(B)

        # Normalize the rewards to compute the advantages
        # .repeat_interleave()对每个样本复制而非整体复制, 这里顺序是一个样本的所有采样在一起
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)# (B*G)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)# (B*G)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)# (B*G)

        # x - x.detach() allows for preserving gradients from x
        # 重要性采样, 但由于这里一个batch仅更新了一次, /pi_/theta_{old}就是/pi_/theta  (B*G, seq)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # 加入KL散度
        per_token_loss = -(per_token_loss - self.beta * per_token_kl) # default 0.04
        # 添加掩码, 在序列维度进行平均, 在batch维度进行平均
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # import pdb; pdb.set_trace()

        # Log the metrics
        # 每个batch*group的平均回答长度(通过completion_mask)
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        self._metrics["advantages"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
        
        self._metrics["reward_mean"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # import pdb; pdb.set_trace()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
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
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))