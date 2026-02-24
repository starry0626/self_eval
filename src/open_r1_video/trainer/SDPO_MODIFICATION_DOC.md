# On-Policy Self-Distillation Trainer 修改文档

## 待解决的问题
1. 教师模型额外输入的构建
   - 不同prompt带来的长度差异如何解决
2. 不同divergence的实现, 估计的实现
   - JSD是否有估计形式, 还是说估计形式仅适用于KL散度?
   - 估计形式是诱骗估计怎么解决
   - 不采用估计采用整个词表带了的额外显存以及耗时的消耗有多少 
   - 采用top-k的估计形式相较于top-1是否会改变无偏有偏, 效果如何, 具体如何实现

### 教师模型额外输入的构建

使用 **MBZUAI/Video-R2-Dataset** 数据集，该数据集的特点：
- 包含带推理过程的问答对（`<think/>` 标签）
- GRPO 子集包含时间定位标注（`temporal_grounding`）
- 可将推理过程和时间定位信息作为教师上下文

[数据集 HuggingFace 链接](https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset)

---
## 概述

基于原有的 `Qwen2VLGRPOTrainer`，创建了新的 `Qwen2VLSDPOTrainer`，实现了 On-Policy Self-Distillation 算法。核心思想是使用训练模型本身作为教师模型，但教师模型接收额外的上下文信息，通过 Jensen-Shannon 散度来衡量学生模型和教师模型输出分布的差异作为损失函数。

---

## 修改点详细说明

### 1. `__init__` 方法修改

| 修改项 | 原GRPO实现 | 新SDPO实现 |
|--------|-----------|-----------|
| 奖励函数参数 | `reward_funcs`, `reward_processing_classes` | **已移除** |
| 参考模型 | `self.ref_model` (独立模型副本) | **已移除** |
| 新增参数 | - | `teacher_context_field: str` (指定数据集中教师上下文的字段名) |
| 采样次数 | `num_generations` (多次) | 固定为 1 次 |
| GenerationConfig | `num_return_sequences=self.num_generations` | `num_return_sequences=1` |

**代码位置**: `sdpo_trainer.py:128-271`

---

### 2. 采样部分修改

**原GRPO实现**:
```python
# 循环生成num_generations份回答
for i in range(num_generations):
    completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
    all_completions.append(completion)
# 拼接所有回答，结果shape为(B * G, Sequence_Length)
prompt_completion_ids = torch.cat(padded_completions, dim=0)
```

**新SDPO实现**:
```python
# 每个样本只生成一次回答
completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
# 结果shape为(B, Sequence_Length)
```

**代码位置**: `sdpo_trainer.py:449-456`

---

### 3. 移除优势计算和奖励计算

**已完全移除以下代码块**:
- 奖励函数调用 (`reward_func(prompts=prompts, completions=completions, ...)`)
- 奖励模型前向传播 (`reward_func(**reward_inputs).logits`)
- 分组奖励计算 (`mean_grouped_rewards`, `std_grouped_rewards`)
- 优势计算 (`advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)`)

---

### 4. 教师模型输入构建

新增 `_inject_teacher_context` 方法，支持三种类型的教师上下文注入：

| 上下文类型 | 处理方式 |
|-----------|---------|
| 字符串 | 作为额外文本追加到用户消息末尾 |
| 列表 | 直接扩展到消息的 content 列表中 |
| 字典 | 支持结构化注入，可包含 `text`, `image`, `video` 字段 |

**代码位置**: `sdpo_trainer.py:291-359`

**示例**:
```python
# 数据集中的 teacher_context 字段可以是:
# 1. 纯文本
teacher_context = "参考答案: xxx"

# 2. 结构化内容列表
teacher_context = [
    {"type": "text", "text": "参考信息: xxx"},
    {"type": "image", "image": "path/to/reference.jpg"}
]

# 3. 字典形式
teacher_context = {
    "text": "参考答案: xxx",
    "image": "path/to/reference.jpg"
}
```

---

### 5. 教师模型前向传播

**关键实现**:
```python
# 构建教师模型的输入（包含额外上下文）
teacher_prompts_messages = [
    self._inject_teacher_context(msg, ctx) 
    for msg, ctx in zip(prompts_messages, teacher_contexts)
]

# 处理教师输入的视觉信息
teacher_image_inputs, teacher_video_inputs, teacher_video_kwargs = process_vision_info(
    teacher_prompts_messages, ...
)

# 教师模型前向传播（无梯度）
with torch.inference_mode():
    per_token_logps_teacher = get_per_token_logps(
        model, teacher_input_ids, attention_mask=teacher_attention_mask, **teacher_prompt_inputs
    )
```

**代码位置**: `sdpo_trainer.py:520-609`

---

### 6. 训练模型前向传播简化

**原GRPO实现**:
```python
# 视觉信息需要复制 num_generations 份
prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(num_generations, 1)
prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(num_generations, 1)

# 计算logits时batch维度为 B*G
per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
```

**新SDPO实现**:
```python
# 无需复制视觉信息，保持原始batch维度 B
per_token_logps_student = get_per_token_logps(model, completion_ids, **prompt_inputs)
```

**代码位置**: `sdpo_trainer.py:515-518`

---

### 7. Jensen-Shannon 散度损失函数

新增 `compute_js_divergence` 函数实现 JS 散度计算：

```python
def compute_js_divergence(log_p_student, log_p_teacher):
    """
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    p_student = torch.exp(log_p_student)
    p_teacher = torch.exp(log_p_teacher)
    
    m = 0.5 * (p_student + p_teacher)
    log_m = torch.log(m + 1e-10)
    
    kl_pm = p_student * (log_p_student - log_m)
    kl_qm = p_teacher * (log_p_teacher - log_m)
    
    js_divergence = 0.5 * kl_pm.sum(dim=-1) + 0.5 * kl_qm.sum(dim=-1)
    return js_divergence
```

**JS散度的优势**:
1. 对称性：JS(P || Q) = JS(Q || P)
2. 有界：0 <= JS(P || Q) <= log(2)
3. 避免了 KL 散度的偏向性问题

**代码位置**: `sdpo_trainer.py:619-660`

---

### 8. 指标记录修改

| 移除的指标 | 新增的指标 |
|-----------|-----------|
| `rewards/{func_name}` | - |
| `reward` | - |
| `advantages` | - |
| `reward_mean` | - |
| `reward_std` | - |
| `kl` | - |
| - | `js_divergence` |

**保留的指标**: `completion_length`

**代码位置**: `sdpo_trainer.py:683-692`

---

## 文件结构

```
self_eval/src/open_r1_video/trainer/
├── grpo_trainer.py           # 原始GRPO实现（保留）
├── sdpo_trainer.py           # 新的SDPO实现
└── teacher_context_builder.py # 教师上下文构建模块
```

---

## 使用示例

### 基本使用（预构建 teacher_context）

```python
from trainers import Qwen2VLSDPOTrainer
from datasets import load_dataset

dataset = load_dataset("your_dataset", split="train")

trainer = Qwen2VLSDPOTrainer(
    model="Qwen/Qwen2-VL-7B-Instruct",
    train_dataset=dataset,
    teacher_context_field="teacher_context",  # 数据集中包含额外上下文的字段名
    max_pixels=12845056,
    min_pixels=3136,
)

trainer.train()
```

### 使用 Video-R2 数据集（自动构建教师上下文）

```python
from trainers import Qwen2VLSDPOTrainer
from teacher_context_builder import TeacherContextConfig

# 配置教师上下文
config = TeacherContextConfig(
    include_answer=True,
    include_temporal_text=True,
    include_temporal_video=False,  # 按需开启
    include_reasoning=True
)

trainer = Qwen2VLSDPOTrainer(
    model="Qwen/Qwen2-VL-7B-Instruct",
    train_dataset=dataset,
    teacher_context_config=config,  # 传入配置
    video_base_dir="./Video-R2",    # 视频文件目录
    max_pixels=12845056,
    min_pixels=3136,
)

trainer.train()
```

---

## 教师上下文构建流程

### 两种构建模式

`sdpo_trainer.py` 支持两种教师上下文构建模式：

| 模式 | 触发条件 | 使用方法 |
|------|---------|---------|
| **简单注入模式** | 数据包含 `teacher_context` 字段 | `_inject_teacher_context()` |
| **高级构建模式** | 数据包含 `conversations` 或 `temporal_grounding` 字段 | `_build_teacher_prompt_from_raw_data()` |

### 调用流程图

```
compute_loss()
    │
    ├── 步骤 1-3: 学生模型输入处理和生成
    │
    └── 步骤 4: 构建教师模型输入
            │
            ├── 检查数据格式
            │       │
            │       ├── 有 conversations/temporal_grounding 字段?
            │       │       │
            │       │       └── YES → 高级构建模式
            │       │               │
            │       │               └── _build_teacher_prompt_from_raw_data()
            │       │                       │
            │       │                       └── TeacherContextBuilder.build_teacher_prompt()
            │       │                               │
            │       │                               ├── extract_answer_from_conversation()
            │       │                               ├── extract_temporal_grounding_text()
            │       │                               ├── sample_temporal_video_segments() [可选]
            │       │                               └── extract_reasoning_from_conversation()
            │       │
            │       └── NO → 简单注入模式
            │               │
            │               └── _inject_teacher_context()
            │                       │
            │                       └── 直接注入预构建的 teacher_context
            │
            └── 步骤 5-9: 教师模型前向传播和损失计算
```

### 代码调用链

```python
# compute_loss 中的关键代码 (sdpo_trainer.py:574-601)

for i, x in enumerate(inputs):
    has_raw_data = "conversations" in x or "temporal_grounding" in x
    
    if has_raw_data:
        # 高级构建模式
        teacher_msg = self._build_teacher_prompt_from_raw_data(
            prompt_messages=prompts_messages[i],
            conversations=x.get("conversations", []),
            temporal_grounding=x.get("temporal_grounding"),
            video_path=x.get("video")
        )
    else:
        # 简单注入模式
        teacher_msg = self._inject_teacher_context(
            prompts_messages[i], 
            teacher_contexts[i]
        )
```

---

## 数据集格式要求

### 格式一：预构建 teacher_context

数据集需要包含以下字段：
- `prompt`: 原始问题/输入（必需）
- `teacher_context`: 教师模型的额外上下文信息（可选，字段名可配置）

### 格式二：Video-R2 原始格式

数据集需要包含以下字段：
- `prompt`: 原始问题/输入（必需）
- `conversations`: 对话列表（必需，用于提取答案和推理过程）
- `temporal_grounding`: 时间定位字典（可选）
- `video`: 视频文件路径（可选，用于视频帧采样）

---

## 推荐数据集：MBZUAI/Video-R2-Dataset

### 数据集概述

[MBZUAI/Video-R2-Dataset](https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset) 是一个用于视频问答的数据集，特别适合用于 SDPO 训练。该数据集包含带有详细推理过程的问答对，可用于构建教师上下文。

### 数据集下载

数据集已下载到项目目录：
```
self_eval/dataset/video-r2/Video-R2/
├── video-r2-grpo-dataset.json   # GRPO 训练数据 (4,804 样本)
└── video-r2-sft-dataset.json    # SFT 训练数据 (10,467 样本)
```

下载脚本位置：`self_eval/dataset/video-r2/download.py`

### 数据集格式

#### GRPO 数据集 (video-r2-grpo-dataset.json)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 数据唯一标识符 |
| `video` | string | 视频文件相对路径 |
| `conversations` | list | 对话列表，包含 human 问题和 assistant 回答 |
| `temporal_grounding` | dict | 时间定位标注，key 为时间戳，value 为描述 |

**示例数据**:
```json
{
  "id": "68d93344-abb1-4f28-8d24-897f27b9f03d",
  "video": "academic_source/ego4d/68d93344-abb1-4f28-8d24-897f27b9f03d.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "Why does the person use a white cloth after making adjustments?\nA. To cover the component\nB. To clean the area and maintain cleanliness\n..."
    },
    {
      "from": "assistant",
      "value": "<think]\nOkay, I need to figure out why the person in the video is using a white cloth...\n</think]\n<answer>\nB. To clean the area and maintain cleanliness\n</answer>"
    }
  ],
  "temporal_grounding": {
    "00:20-00:48": "He actively wipes internal surfaces of the mechanical housing with a white cloth.",
    "01:06": "He begins reassembling the components."
  }
}
```

#### SFT 数据集 (video-r2-sft-dataset.json)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 数据唯一标识符 |
| `video` | string | 视频文件相对路径 |
| `conversations` | list | 对话列表 |

**与 GRPO 数据集的区别**：
- 不包含 `temporal_grounding` 字段
- 样本数量更多 (10,467 vs 4,804)

### 视频来源分布

| 来源 | 样本数 |
|------|--------|
| LLaVA-Video-178K | 2,716 |
| academic_source | 708 |
| STAR | 326 |
| perception_test | 302 |
| NeXT-QA | 248 |
| CLEVRER | 192 |
| PerceptionTest | 170 |
| NextQA | 142 |

### 数据集特点

1. **带推理过程的回答**：assistant 的回答包含 `<think/>` 标签内的详细推理过程，适合用于构建教师上下文
2. **时间定位标注**：GRPO 数据集包含 temporal_grounding 字段，可用于提供额外的时间相关信息
3. **多选题格式**：问题以多选题形式呈现，便于评估

### 用于 SDPO 的数据转换建议

为了将 Video-R2-Dataset 用于 SDPO 训练，建议进行以下转换：

```python
def convert_to_sdpo_format(sample):
    """将 Video-R2 数据集转换为 SDPO 格式"""
    
    # 提取问题和推理过程
    question = sample['conversations'][0]['value']
    assistant_response = sample['conversations'][1]['value']
    
    # 提取 <think/> 内的推理过程作为教师上下文
    import re
    think_match = re.search(r'<think\]?(.*?)</think\]?', assistant_response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # 提取答案
    answer_match = re.search(r'<answer\]?(.*?)</answer\]?', assistant_response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # 构建教师上下文（可包含推理提示或时间定位信息）
    teacher_context = f"参考推理过程：\n{reasoning}"
    
    # 添加时间定位信息（如果有）
    temporal_info = sample.get('temporal_grounding', {})
    if temporal_info:
        valid_timestamps = {k: v for k, v in temporal_info.items() if v is not None}
        if valid_timestamps:
            teacher_context += f"\n\n关键时间点：\n"
            for ts, desc in list(valid_timestamps.items())[:3]:
                teacher_context += f"- {ts}: {desc}\n"
    
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample['video']},
                    {"type": "text", "text": question}
                ]
            }
        ],
        "teacher_context": teacher_context
    }
```

### 视频文件下载

视频文件需要单独下载，可通过以下方式：

```python
from huggingface_hub import hf_hub_download

# 下载单个视频
video_path = hf_hub_download(
    repo_id="MBZUAI/Video-R2-Dataset",
    filename="academic_source/ego4d/68d93344-abb1-4f28-8d24-897f27b9f03d.mp4",
    repo_type="dataset",
    local_dir="./Video-R2",
)
```

**注意**：视频文件较大，建议按需下载。

---

## 核心流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    compute_loss 流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  步骤1: 准备学生模型输入                                      │
│     ├── 提取 prompt 消息                                     │
│     ├── 处理视觉信息 (process_vision_info)                   │
│     ├── 应用 chat template                                  │
│     └── 备份 input_ids 和 attention_mask                    │
│                                                             │
│  步骤2: 生成回答 (每个样本一次)                               │
│     └── unwrapped_model.generate()                          │
│                                                             │
│  步骤3: 计算学生模型的对数概率                                │
│     └── get_per_token_logps(model, completion_ids)          │
│                                                             │
│  步骤4: 构建教师模型输入（优化版）                            │
│     ├── 构建额外上下文消息（不含视频）                        │
│     ├── 处理额外上下文的视觉信息（只有图像帧）                │
│     ├── 去除学生输入的填充（根据 attention_mask）            │
│     ├── 拼接学生输入和额外上下文                             │
│     └── 重新填充到 batch 内最大长度                          │
│                                                             │
│  步骤5: 计算教师模型的对数概率 (无梯度)                        │
│     └── get_per_token_logps(model, teacher_input_ids)       │
│                                                             │
│  步骤6: 对齐序列长度                                         │
│                                                             │
│  步骤7: 计算 JS 散度损失                                     │
│     └── compute_js_divergence(logps_student, logps_teacher) │
│                                                             │
│  步骤8: 创建掩码并计算最终损失                                │
│     └── loss = (per_token_loss * mask).mean()               │
│                                                             │
│  步骤9: 记录指标                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 教师输入处理优化

### 优化背景

原始实现中，教师模型的输入会重新处理整个消息（包括视频），这导致：
1. **重复处理视频数据**：视频被加载、解码、采样两次
2. **额外显存占用**：视频张量被重复存储
3. **处理时间增加**：视频处理是耗时操作

### 优化策略

**核心思想**：只处理额外信息，然后与学生输入拼接，避免重复处理视频数据。

```
原始流程（重复处理）:
学生输入: video + text → processor → tensors
教师输入: video + text + 额外信息 → processor → tensors (视频被重复处理！)

优化后流程:
学生输入: video + text → processor → tensors (保存)
额外信息: text + images → processor → tensors (仅处理新增部分)
拼接: 学生tensors + 额外信息tensors → 教师tensors
```

### 实现细节

#### 1. 备份学生输入

在步骤 1 中，备份 `input_ids` 和 `attention_mask`：

```python
# 备份 input_ids 和 attention_mask（用于后续教师输入拼接）
prompt_inputs["input_ids_backup"] = prompt_inputs["input_ids"].clone()
prompt_inputs["attention_mask_backup"] = prompt_inputs["attention_mask"].clone()
```

#### 2. 构建额外上下文

使用 `build_extra_context_only()` 函数只构建额外信息（不包含原始视频和问题）：

```python
# 只构建额外上下文（不包含原始视频和问题）
extra_msg = self.teacher_context_builder.build_extra_context_only(
    conversations=x.get("conversations", []),
    question_text=question_text,
    temporal_grounding=x.get("temporal_grounding"),
    video_path=x.get("video"),
    video_base_dir=self.video_base_dir
)
```

#### 3. 拼接输入

使用 `_concat_student_and_extra_inputs()` 方法拼接学生输入和额外上下文：

```python
teacher_prompt_inputs = self._concat_student_and_extra_inputs(
    student_input_ids=student_input_ids,
    student_attention_mask=student_attention_mask,
    student_prompt_inputs=prompt_inputs,
    extra_input_ids=extra_prompt_inputs.get("input_ids"),
    extra_attention_mask=extra_prompt_inputs.get("attention_mask"),
    extra_prompt_inputs=extra_prompt_inputs,
    pad_token_id=self.processing_class.tokenizer.pad_token_id,
)
```

#### 4. 去填充和重新填充逻辑

```python
def _concat_student_and_extra_inputs(...):
    # 1. 去除学生输入的填充（根据 attention_mask）
    student_mask = student_attention_mask[i].bool()
    student_ids_no_pad = student_input_ids[i][student_mask]
    
    # 2. 去除额外输入的填充
    extra_mask = extra_attention_mask[i].bool()
    extra_ids_no_pad = extra_input_ids[i][extra_mask]
    
    # 3. 拼接：学生输入在前，额外输入在后
    concat_ids = torch.cat([student_ids_no_pad, extra_ids_no_pad], dim=0)
    
    # 4. 重新填充到 batch 内最大长度（左侧填充）
    # ...
```

### 视觉信息处理

| 视觉信息 | 处理方式 |
|---------|---------|
| `pixel_values_videos` | 直接复制（不包含 batch 维度） |
| `video_grid_thw` | 直接复制（包含 batch 维度） |
| `pixel_values` | 拼接学生和额外的图像像素值 |
| `image_grid_thw` | 拼接学生和额外的图像网格信息 |

### 优化效果

| 指标 | 原始实现 | 优化后 |
|------|---------|--------|
| 视频处理次数 | 2 次 | 1 次 |
| 视频显存占用 | 2 份 | 1 份 |
| 额外信息处理 | 包含视频 | 仅文本和图像帧 |

---

## 教师上下文构建模块

### 模块概述

`teacher_context_builder.py` 模块提供了构建教师模型额外输入的完整功能，支持四种类型的额外信息，每种都可以通过配置参数独立控制：

| 额外输入类型 | 配置参数 | 说明 |
|-------------|---------|------|
| 标准答案 | `include_answer` | 选项字母及对应内容 |
| 时间定位文本 | `include_temporal_text` | temporal_grounding 的文本描述 |
| 时间定位视频片段 | `include_temporal_video` | temporal_grounding 时间段的采样帧 |
| 推理流程 | `include_reasoning` | `<think/>` 标签内的推理过程 |

**代码位置**: `teacher_context_builder.py`

---

### TeacherContextConfig 配置类

```python
@dataclass
class TeacherContextConfig:
    """教师上下文配置类"""
    
    # 控制各部分是否包含
    include_answer: bool = True           # 是否包含标准答案
    include_temporal_text: bool = True    # 是否包含时间定位文本
    include_temporal_video: bool = False  # 是否包含时间定位视频片段
    include_reasoning: bool = True        # 是否包含推理流程
    
    # 视频采样参数
    temporal_frames_count: int = 4        # 时间段采样帧数（等间隔采样）
    point_frames_count: int = 4           # 时间点采样帧数（前后采样）
    point_frames_range: float = 1.0       # 时间点采样的前后范围（秒）
    max_temporal_segments: int = 5        # 最大时间片段数量
```

---

### 四种额外输入详细说明

#### 1. 标准答案 (include_answer)

从对话中提取标准答案，包括选项字母和对应内容。

**提取逻辑**:
1. 从 assistant 回答中提取 `<answer>` 标签内容
2. 从问题文本中提取所有选项
3. 匹配答案字母与选项内容

**输出格式**:
```
[Correct Answer]
The correct answer is provided for your reference: B. To clean the area and maintain cleanliness
```

**相关函数**:
- `extract_answer_from_conversation()`: 提取答案文本和字母
- `extract_question_and_options()`: 提取问题和选项字典

---

#### 2. 时间定位文本 (include_temporal_text)

提取 temporal_grounding 字段的文本内容。

**输入格式**:
```json
{
  "temporal_grounding": {
    "00:20-00:48": "He actively wipes internal surfaces with a white cloth.",
    "01:06": "He begins reassembling the components."
  }
}
```

**输出格式**:
```
[Key Temporal Moments]
The following are key moments with their descriptions. Use them to guide your analysis:
1. [00:20-00:48]: He actively wipes internal surfaces with a white cloth.
2. [01:06]: He begins reassembling the components.
```

**相关函数**:
- `extract_temporal_grounding_text()`: 提取时间片段列表
- `parse_timestamp()`: 解析时间戳字符串

---

#### 3. 时间定位视频片段 (include_temporal_video)

采样 temporal_grounding 时间段对应的视频帧，并注入到教师模型输入中。

**采样策略**:

| 时间戳类型 | 采样方式 | 参数 |
|-----------|---------|------|
| 时间段 (如 `00:20-00:48`) | 等间隔采样 | `temporal_frames_count` 帧 |
| 时间点 (如 `01:06`) | 前后采样 | `point_frames_count` 帧，前后各 `point_frames_range` 秒 |

**示例**:
- `00:20-00:48` (28秒时间段): 等间隔采样 4 帧，分别在 00:20, 00:29, 00:38, 00:48
- `01:06` (时间点): 在 01:05-01:07 范围内采样 4 帧

**输出格式**:

视频帧注入采用 Qwen3-VL 官方的 Interleaved Timestamp-Image Pairs 格式，使用 **PIL.Image 对象**直接传递图像：

```python
# 每帧以时间戳文本 + 图像的方式交错注入
msg["content"].append({
    "type": "text",
    "text": "<20.0 seconds>"  # 时间戳标签
})
msg["content"].append({
    "type": "image",
    "image": pil_image  # PIL.Image 对象，无需 Base64 编码
})
```

**为什么使用 PIL.Image 对象？**

| 传递方式 | 优点 | 缺点 |
|---------|------|------|
| PIL.Image 对象 | 无需编码/解码，内存直接传递，速度最快 | 需要预先加载图像 |
| Base64 Data URI | 内嵌数据，无外部依赖 | 数据量增大约 33%，编码/解码开销 |
| 本地文件路径 | 简单直接 | 需要文件 I/O |

对于动态采样的视频帧，使用 PIL.Image 对象是最优选择：
1. 帧数据已在内存中（采样时加载）
2. 无需额外的编码/解码步骤
3. 避免文件 I/O 开销

**相关函数**:
- `sample_frames_from_video()`: 从视频中采样帧，返回帧和时间戳
- `sample_temporal_video_segments()`: 采样所有时间片段

---

#### 4. 推理流程 (include_reasoning)

从 assistant 回答中提取 `<think/>` 标签内的推理过程。

**输入格式**:
```json
{
  "from": "assistant",
  "value": "<think\nOkay, I need to figure out why...\nLet me analyze...\n</think\n<answer\nB. To clean\n</answer>"
}
```

**输出格式**:
```
[Reference Reasoning]
The following is a reference reasoning process. Learn from it to improve your reasoning:
Okay, I need to figure out why...
Let me analyze...
```

**相关函数**:
- `extract_reasoning_from_conversation()`: 提取推理流程

---

### Prompt 模板

教师模型的输入 = 原始学生输入 + 额外上下文信息。额外上下文追加在原始问题之后：

```
<video>
Why does the person use a white cloth?
A. To cover
B. To clean
C. To dry
D. To signal

[Correct Answer]
The correct answer is provided for your reference: B. To clean the area and maintain cleanliness

[Key Temporal Moments]
The following are key moments with their descriptions. Use them to guide your analysis:
1. [00:20-00:48]: He actively wipes internal surfaces with a white cloth.
2. [01:06]: He begins reassembling the components.

[Reference Reasoning]
The following is a reference reasoning process. Learn from it to improve your reasoning:
Okay, I need to figure out why the person is using a white cloth...
```

当 `include_temporal_video=True` 时，还会按照 Qwen3-VL 官方的 Interleaved Timestamp-Image Pairs 格式注入采样帧：

```
<video>  <!-- 原始完整视频 -->
Why does the person use a white cloth?
...

[Key Temporal Frames]
The following frames are sampled from key moments. Use them as reference:
<20.0 seconds>
<image>
<29.0 seconds>
<image>
<38.0 seconds>
<image>
<48.0 seconds>
<image>
<65.0 seconds>
<image>
<66.0 seconds>
<image>
<67.0 seconds>
<image>

[Correct Answer]
The correct answer is provided for your reference: B. To clean the area and maintain cleanliness

[Key Temporal Moments]
The following are key moments with their descriptions. Use them to guide your analysis:
1. [00:20-00:48]: He actively wipes internal surfaces with a white cloth.
2. [01:06]: He begins reassembling the components.

[Reference Reasoning]
The following is a reference reasoning process. Learn from it to improve your reasoning:
Okay, I need to figure out why the person is using a white cloth...
```

---

### TeacherContextBuilder 类

主要类，用于构建教师模型的额外输入上下文。

**使用示例**:

```python
from teacher_context_builder import TeacherContextBuilder, TeacherContextConfig

# 创建配置
config = TeacherContextConfig(
    include_answer=True,
    include_temporal_text=True,
    include_temporal_video=False,  # 视频片段采样较慢，按需开启
    include_reasoning=True,
    max_temporal_segments=3
)

# 创建构建器
builder = TeacherContextBuilder(config)

# 构建教师 prompt
teacher_messages = builder.build_teacher_prompt(
    original_prompt_messages=prompt_messages,
    conversations=sample["conversations"],
    temporal_grounding=sample.get("temporal_grounding"),
    video_path=sample["video"],
    video_base_dir="./Video-R2"
)

# 构建上下文字典
context_dict = builder.build_context_dict(
    conversations=sample["conversations"],
    temporal_grounding=sample.get("temporal_grounding"),
    video_path=sample["video"],
    video_base_dir="./Video-R2"
)
```

---

### 数据转换工具

提供 `convert_video_r2_to_sdpo_format()` 函数，将 Video-R2 数据集样本转换为 SDPO 训练格式：

```python
from teacher_context_builder import convert_video_r2_to_sdpo_format, TeacherContextConfig

config = TeacherContextConfig(
    include_answer=True,
    include_temporal_text=True,
    include_temporal_video=False,
    include_reasoning=True
)

sdpo_sample = convert_video_r2_to_sdpo_format(
    sample=video_r2_sample,
    config=config,
    video_base_dir="./Video-R2"
)

# 输出格式
{
    "id": "xxx",
    "prompt": [...],           # 学生模型输入
    "teacher_prompt": [...],   # 教师模型输入
    "video": "path/to/video.mp4",
    "teacher_context": {...}   # 结构化上下文字典
}
```

---

### 视频采样实现细节

#### 时间戳解析

支持的时间戳格式：
- `00:20` -> 单个时间点 (20秒)
- `00:20-00:48` -> 时间段 (20秒到48秒)
- `01:06` -> 单个时间点 (66秒)
- `01:30:00` -> 带小时的时间戳

```python
def parse_timestamp(timestamp_str: str) -> Tuple[float, float]:
    """解析时间戳，返回 (开始时间, 结束时间) 秒数"""
    # "00:20-00:48" -> (20.0, 48.0)
    # "01:06" -> (66.0, 66.0)
```

#### 帧采样算法

```python
def sample_frames_from_video(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int,
    is_point: bool = False,
    point_range: float = 1.0
) -> List[np.ndarray]:
    """
    从视频中采样帧
    
    时间段采样: np.linspace(start_time, end_time, num_frames)
    时间点采样: np.linspace(center - range, center + range, num_frames)
    """
```

---

## 注意事项

1. **教师上下文注入**：支持字符串、列表、字典三种形式，可灵活处理文本和视觉信息
2. **JS散度计算**：使用对称的 JS 散度而非 KL 散度，避免偏向问题
3. **显存优化**：保留了原有的 CrossEntropyLoss 融合计算策略，避免大词表导致的显存峰值
4. **序列对齐**：教师模型和学生模型的 prompt 长度可能不同，代码已处理对齐逻辑
5. **视频采样性能**：`include_temporal_video` 选项会触发视频帧采样，建议仅在需要时开启
6. **时间片段限制**：通过 `max_temporal_segments` 参数控制最大片段数，避免上下文过长

---

## 快速开始

### 文件结构

```
self_eval/
├── src/open_r1_video/
│   ├── sdpo.py                    # SDPO 训练启动脚本
│   └── trainer/
│       ├── sdpo_trainer.py        # SDPO Trainer 实现
│       └── teacher_context_builder.py  # 教师上下文构建模块
├── scripts/
│   └── train_sdpo.sh              # 训练启动脚本
└── dataset/video-r2/
    ├── video-r2-grpo-dataset.json # 数据集
    └── Video-R2/                  # 视频文件目录
```

### 启动训练

**方式一：使用 Shell 脚本**

```bash
# 修改脚本中的路径配置
vim scripts/train_sdpo.sh

# 启动训练
bash scripts/train_sdpo.sh
```

**方式二：使用 Python 命令**

```bash
python src/open_r1_video/sdpo.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Thinking \
    --jsonl_path ./dataset/video-r2/video-r2-grpo-dataset.json \
    --video_base_dir ./dataset/video-r2/Video-R2 \
    --output_dir ./output/sdpo \
    --include_answer true \
    --include_temporal_text true \
    --include_temporal_video false \
    --include_reasoning true \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --bf16 true \
    --gradient_checkpointing true
```

---

## 参数配置详解

### 教师上下文参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_answer` | bool | True | 是否在教师上下文中包含标准答案 |
| `include_temporal_text` | bool | True | 是否在教师上下文中包含时间定位文本 |
| `include_temporal_video` | bool | False | 是否在教师上下文中包含时间定位视频片段 |
| `include_reasoning` | bool | True | 是否在教师上下文中包含推理流程 |
| `temporal_frames_count` | int | 4 | 时间段采样帧数（等间隔采样） |
| `point_frames_count` | int | 4 | 时间点采样帧数（前后采样） |
| `max_temporal_segments` | int | 5 | 最大时间片段数量 |

**配置建议**：

1. **基础配置**（推荐初次使用）：
   ```bash
   --include_answer true \
   --include_temporal_text true \
   --include_temporal_video false \
   --include_reasoning true
   ```
   只使用文本形式的额外信息，无需视频文件。

2. **完整配置**（需要视频文件）：
   ```bash
   --include_answer true \
   --include_temporal_text true \
   --include_temporal_video true \
   --include_reasoning true \
   --video_base_dir ./dataset/video-r2/Video-R2
   ```
   包含视频帧采样，需要确保视频文件存在。

3. **最小配置**（仅答案）：
   ```bash
   --include_answer true \
   --include_temporal_text false \
   --include_temporal_video false \
   --include_reasoning false
   ```

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_prompt_length` | int | 8192 | 最大 prompt 长度（包含额外上下文） |
| `max_completion_length` | int | 1024 | 最大生成长度 |
| `learning_rate` | float | 2e-6 | 学习率，建议 1e-6 ~ 5e-6 |
| `per_device_train_batch_size` | int | 2 | 每设备训练批次大小 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `num_train_epochs` | int | 1 | 训练轮次 |
| `warmup_ratio` | float | 0.05 | 预热比例 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name_or_path` | str | 必填 | 模型路径或 HuggingFace ID |
| `bf16` | bool | True | 使用 BF16 精度 |
| `gradient_checkpointing` | bool | True | 梯度检查点（节省显存） |
| `attn_implementation` | str | flash_attention_2 | 注意力实现方式 |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `jsonl_path` | str | None | JSON/JSONL 数据集文件路径 |
| `video_base_dir` | str | None | 视频文件基础目录 |
| `max_pixels` | int | 12845056 | 图像/视频处理的最大像素数 |
| `min_pixels` | int | 3136 | 图像/视频处理的最小像素数 |

---

## 使用流程

### 步骤 1：准备数据集

确保数据集符合以下格式之一：

**格式一：Video-R2 格式**（推荐）
```json
{
  "id": "sample-001",
  "video": "path/to/video.mp4",
  "conversations": [
    {"from": "human", "value": "问题内容..."},
    {"from": "assistant", "value": "<think]推理过程</think]<answer>答案</answer>"}
  ],
  "temporal_grounding": {
    "00:20-00:48": "描述内容"
  }
}
```

**格式二：预构建 teacher_context 格式**
```json
{
  "id": "sample-001",
  "prompt": [...],
  "teacher_context": "额外上下文信息"
}
```

### 步骤 2：下载视频文件（可选）

如果使用 `include_temporal_video=true`，需要下载视频文件：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MBZUAI/Video-R2-Dataset",
    repo_type="dataset",
    local_dir="./dataset/video-r2/Video-R2"
)
```

### 步骤 3：配置训练参数

编辑 `scripts/train_sdpo.sh`，修改以下配置：

```bash
# 模型路径
MODEL_PATH="Qwen/Qwen3-VL-2B-Thinking"

# 数据集路径
DATA_PATH="./dataset/video-r2/video-r2-grpo-dataset.json"
VIDEO_BASE_DIR="./dataset/video-r2/Video-R2"

# 教师上下文配置
INCLUDE_ANSWER=true
INCLUDE_TEMPORAL_TEXT=true
INCLUDE_TEMPORAL_VIDEO=false
INCLUDE_REASONING=true
```

### 步骤 4：启动训练

```bash
bash scripts/train_sdpo.sh
```

### 步骤 5：监控训练

训练日志会输出到控制台，同时上传到 SwanLab（如果配置）：

```
SDPO Training Configuration
============================================================
Model: Qwen/Qwen3-VL-2B-Thinking
Dataset: ./dataset/video-r2/video-r2-grpo-dataset.json
Output: ./output/sdpo/...
------------------------------------------------------------
Teacher Context Configuration:
  - include_answer: True
  - include_temporal_text: True
  - include_temporal_video: False
  - include_reasoning: True
============================================================

Dataset loaded: 4804 samples
Dataset processed, sample keys: dict_keys(['prompt', 'conversations', ...])

Starting training...
```

---

## 显存需求参考

| 配置 | batch_size | 显存需求 | 说明 |
|------|------------|----------|------|
| Qwen3-VL-2B | 2 | ~16GB | 不包含视频帧采样 |
| Qwen3-VL-2B | 2 | ~20GB | 包含视频帧采样 |
| Qwen3-VL-7B | 1 | ~40GB | 需要更大显存 |

**显存优化建议**：
1. 启用 `gradient_checkpointing=true`
2. 减小 `per_device_train_batch_size`
3. 增加 `gradient_accumulation_steps`
4. 使用 `include_temporal_video=false`
