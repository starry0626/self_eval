# vLLM + Qwen3-VL 视频推理完全指南

本文档详细讲解如何使用 vLLM 库对 Qwen3-VL 系列模型进行视频推理，涵盖从环境安装到完整推理流程的每一个环节。

---

## 目录

1. [概述](#1-概述)
2. [环境准备](#2-环境准备)
3. [核心概念](#3-核心概念)
4. [完整推理流程](#4-完整推理流程)
   - 4.1 [加载 Processor](#41-加载-processor)
   - 4.2 [构造 Messages](#42-构造-messages)
   - 4.3 [应用 Chat Template 生成 Prompt](#43-应用-chat-template-生成-prompt)
   - 4.4 [提取视觉数据](#44-提取视觉数据)
   - 4.5 [组装 vLLM 输入](#45-组装-vllm-输入)
   - 4.6 [初始化 vLLM LLM 引擎](#46-初始化-vllm-llm-引擎)
   - 4.7 [配置 SamplingParams](#47-配置-samplingparams)
   - 4.8 [执行推理](#48-执行推理)
5. [端到端代码示例](#5-端到端代码示例)
   - 5.1 [单视频推理](#51-单视频推理)
   - 5.2 [批量视频推理](#52-批量视频推理)
   - 5.3 [图片推理](#53-图片推理)
6. [Thinking 模式](#6-thinking-模式)
7. [关键参数参考](#7-关键参数参考)
8. [vLLM 在线服务模式](#8-vllm-在线服务模式)
9. [常见问题与排查](#9-常见问题与排查)

---

## 1. 概述

### vLLM 与 HuggingFace Transformers 的区别

| 特性 | HuggingFace Transformers | vLLM |
|------|-------------------------|------|
| 推理方式 | 逐样本串行推理 | 批量并行推理（Continuous Batching） |
| KV Cache 管理 | 静态分配 | PagedAttention 动态管理 |
| 吞吐量 | 较低 | 显著更高（通常 2-5x） |
| 多 GPU 支持 | `device_map="auto"` | 原生张量并行（Tensor Parallelism） |
| 适用场景 | 少量样本、调试 | 大规模评估、生产部署 |

### Qwen3-VL 可用模型

| 模型 | 类型 | 参数量 |
|------|------|--------|
| `Qwen/Qwen3-VL-2B-Instruct` | Dense | 2B |
| `Qwen/Qwen3-VL-4B-Instruct` | Dense | 4B |
| `Qwen/Qwen3-VL-8B-Instruct` | Dense | 8B |
| `Qwen/Qwen3-VL-32B-Instruct` | Dense | 32B |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | MoE | 30B (3B active) |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | MoE | 235B (22B active) |

> 以上模型均有 `-Thinking` 和 `-FP8` 变体。

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# vLLM（需要 >= 0.8.5，推荐最新版）
pip install -U vllm

# Qwen VL 工具库（处理视觉输入）
pip install qwen-vl-utils>=0.0.14

# HuggingFace Transformers（用于 Processor/Chat Template）
pip install transformers>=4.51

# 其他依赖
pip install accelerate
```

### 2.2 环境变量

```bash
# 多进程模式（多 GPU 必须设置）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 可选：避免 CPU 线程争用
export OMP_NUM_THREADS=1

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0      # 单卡
export CUDA_VISIBLE_DEVICES=0,1    # 双卡
```

### 2.3 硬件需求参考

| 模型 | 最低 GPU 配置 |
|------|--------------|
| 2B / 4B | 单卡 24GB (RTX 4090) |
| 8B | 单卡 40GB+ (A100-40G) |
| 32B | 2-4 卡 80GB (A100/H100) |
| 235B-A22B (FP8) | 8 卡 80GB |

---

## 3. 核心概念

使用 vLLM 对 Qwen3-VL 进行视频推理，核心流程可以概括为：

```
Messages (对话格式)
    │
    ▼
┌─────────────────────────────────────┐
│ processor.apply_chat_template()     │  → 生成 prompt 文本字符串
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ process_vision_info()               │  → 提取视频帧数据 + 处理参数
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 组装 vLLM 输入字典                   │  → { prompt, multi_modal_data,
│                                     │      mm_processor_kwargs }
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ llm.generate()                      │  → vLLM 内部完成 tokenization、
│                                     │    视觉编码、KV Cache 管理、
│                                     │    token 生成、解码
└─────────────────────────────────────┘
    │
    ▼
  生成的文本
```

**关键区别**：与 HuggingFace Transformers 不同，vLLM 内部会自行完成 tokenization 和视觉数据预处理，因此我们只需传入**原始 prompt 文本**和**视觉数据**（PIL Image 列表），而不需要手动调用 `processor()` 来生成 `input_ids`。

---

## 4. 完整推理流程

### 4.1 加载 Processor

Processor 在 vLLM 推理中仅用于 **应用 Chat Template**（将 messages 格式转换为模型能理解的 prompt 字符串），不用于 tokenization。

```python
from transformers import AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
```

### 4.2 构造 Messages

Messages 采用 OpenAI 风格的多轮对话格式，视频通过 `type: "video"` 内容块传入：

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/path/to/video.mp4",   # 本地路径或 URL
                "fps": 1.0,                       # 采样帧率（帧/秒）
                "max_frames": 32,                 # 最大采样帧数
            },
            {
                "type": "text",
                "text": "请描述这个视频的内容。",
            },
        ],
    }
]
```

#### 视频内容块支持的参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `video` | `str` | 视频文件路径（本地或 URL） | 必填 |
| `fps` | `float` | 采样帧率 | 2.0 |
| `max_frames` | `int` | 最大采样帧数 | 由模型决定 |
| `max_pixels` | `int` | 每帧最大像素数 | 处理器默认值 |
| `total_pixels` | `int` | 所有帧总像素预算 | `20480 * 28 * 28` |
| `min_pixels` | `int` | 每帧最小像素数 | `16 * 28 * 28` |

### 4.3 应用 Chat Template 生成 Prompt

Chat Template 将 messages 列表转换为模型训练时使用的格式化 prompt 字符串：

```python
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,             # 返回字符串，不返回 token ids
    add_generation_prompt=True, # 末尾添加 assistant 开头标记
)
```

生成的 prompt 类似如下结构（ChatML 格式）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>请描述这个视频的内容。<|im_end|>
<|im_start|>assistant
```

其中：
- `<|vision_start|>` / `<|vision_end|>` — 视觉内容的边界标记
- `<|video_pad|>` — 视频数据占位符（vLLM 会在内部替换为实际的视觉 token）
- `<|im_start|>` / `<|im_end|>` — ChatML 消息边界标记

### 4.4 提取视觉数据

使用 `qwen_vl_utils` 库的 `process_vision_info` 函数从 messages 中提取视频帧数据：

```python
from qwen_vl_utils import process_vision_info

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,   # 返回视频处理参数（如 fps、尺寸等）
)
```

返回值说明：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `image_inputs` | `list[PIL.Image]` 或 `None` | 图片列表（无图片时为 None） |
| `video_inputs` | `list[np.ndarray]` 或 `None` | 视频帧数组列表，每个元素形状为 `(num_frames, H, W, 3)` |
| `video_kwargs` | `dict` | 视频处理参数（传递给 vLLM 的 `mm_processor_kwargs`） |

**`process_vision_info` 内部做了什么？**

1. 遍历 messages 中所有 `type: "video"` 的内容块
2. 使用 OpenCV/decord 读取视频文件
3. 按照指定 `fps` 和 `max_frames` 进行帧采样
4. 将采样帧转换为 numpy 数组（`uint8`, `0-255`）
5. 收集视频处理参数（帧尺寸、网格信息等）返回为 `video_kwargs`

### 4.5 组装 vLLM 输入

将上述结果组装为 vLLM `llm.generate()` 接受的输入字典：

```python
mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

vllm_input = {
    "prompt": prompt,                    # 4.3 生成的 prompt 字符串
    "multi_modal_data": mm_data,         # 视觉数据
    "mm_processor_kwargs": video_kwargs, # 视频处理参数
}
```

**`multi_modal_data["video"]` 接受的数据类型：**

- `list[PIL.Image.Image]` — 帧列表（每帧是一张 PIL 图片）
- `numpy.ndarray` — 形状 `(num_frames, H, W, 3)`
- `torch.Tensor` — 帧张量
- `list[numpy.ndarray]` — 多段视频，每段为一个数组

### 4.6 初始化 vLLM LLM 引擎

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.80,       # GPU 显存利用率
    tensor_parallel_size=1,            # 张量并行 GPU 数
    limit_mm_per_prompt={"video": 1},  # 每个 prompt 最多 1 个视频
)
```

#### LLM 构造参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `model` | `str` | 模型名称或本地路径 | - |
| `trust_remote_code` | `bool` | 允许执行模型自定义代码 | `True` |
| `tensor_parallel_size` | `int` | 张量并行 GPU 数 | GPU 数量 |
| `gpu_memory_utilization` | `float` | 显存利用率上限 | `0.70` - `0.85` |
| `max_model_len` | `int` | 最大序列长度 | 按需设置（减小可节省显存） |
| `limit_mm_per_prompt` | `dict` | 每个 prompt 允许的媒体数量 | `{"video": 1}` |
| `enforce_eager` | `bool` | 禁用 CUDA Graph（调试用） | `False` |
| `max_num_seqs` | `int` | 最大并发序列数 | `5` - `32` |
| `mm_processor_kwargs` | `dict` | 引擎级别视觉处理默认参数 | 按需设置 |

**多 GPU 注意事项：**

```python
import torch

llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),  # 使用所有可用 GPU
)
```

**MoE 模型（235B）专用配置：**

```python
llm = LLM(
    model="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    tensor_parallel_size=8,
    mm_encoder_tp_mode="data",       # 视觉编码器使用数据并行
    enable_expert_parallel=True,     # 将 MoE 专家分布到多卡
)
```

### 4.7 配置 SamplingParams

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0,       # 贪心解码（确定性输出）
    max_tokens=1024,     # 最大生成 token 数
)
```

#### 推荐参数配置

| 参数 | Instruct 模型 | Thinking 模型 | 说明 |
|------|--------------|--------------|------|
| `temperature` | `0`（评估）/ `0.7`（对话） | `0.6` | 采样温度 |
| `top_p` | `1.0` | `0.95` | 核采样概率 |
| `top_k` | `-1`（禁用） | `20` | Top-K 采样 |
| `max_tokens` | `1024` | `32768` | 最大生成长度 |
| `repetition_penalty` | `1.0` | `1.0` | 重复惩罚 |

### 4.8 执行推理

```python
outputs = llm.generate([vllm_input], sampling_params=sampling_params)

# 提取生成文本
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
```

`llm.generate()` 接受一个输入列表，支持批量推理：

```python
# 批量推理（多个样本同时处理）
outputs = llm.generate(
    [vllm_input_1, vllm_input_2, vllm_input_3, ...],
    sampling_params=sampling_params,
)
```

#### 输出结构

```python
output = outputs[0]
output.prompt           # 输入的 prompt
output.outputs[0].text  # 生成的文本
output.outputs[0].token_ids      # 生成的 token id 列表
output.outputs[0].finish_reason  # 停止原因: "stop" 或 "length"
```

---

## 5. 端到端代码示例

### 5.1 单视频推理

```python
"""单视频推理：最简完整示例"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---- 配置 ----
model_path = "Qwen/Qwen3-VL-8B-Instruct"
video_path = "/path/to/video.mp4"
question = "请描述这个视频中发生了什么。"

# ---- 1. 构造 messages ----
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 1.0, "max_frames": 32},
            {"type": "text", "text": question},
        ],
    }
]

# ---- 2. 处理 prompt 和视觉数据 ----
processor = AutoProcessor.from_pretrained(model_path)

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True
)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

vllm_input = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
    "mm_processor_kwargs": video_kwargs,
}

# ---- 3. 加载模型并推理 ----
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.80,
    tensor_parallel_size=1,
    limit_mm_per_prompt={"video": 1},
)

sampling_params = SamplingParams(temperature=0, max_tokens=1024)
outputs = llm.generate([vllm_input], sampling_params=sampling_params)

# ---- 4. 输出结果 ----
print(outputs[0].outputs[0].text)
```

### 5.2 批量视频推理

```python
"""批量视频推理：适用于评估场景"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

model_path = "Qwen/Qwen3-VL-8B-Instruct"


def prepare_vllm_input(processor, messages):
    """将一组 messages 转换为 vLLM 输入格式"""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ---- 准备多个样本 ----
samples = [
    {"video": "/path/to/video1.mp4", "question": "视频中有几个人？"},
    {"video": "/path/to/video2.mp4", "question": "视频中的主要活动是什么？"},
    {"video": "/path/to/video3.mp4", "question": "请描述视频的场景。"},
]

processor = AutoProcessor.from_pretrained(model_path)

# 构建所有输入
vllm_inputs = []
for sample in samples:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": sample["video"], "fps": 1.0, "max_frames": 32},
                {"type": "text", "text": sample["question"]},
            ],
        }
    ]
    vllm_inputs.append(prepare_vllm_input(processor, messages))

# ---- 加载模型 ----
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.80,
    tensor_parallel_size=1,
    limit_mm_per_prompt={"video": 1},
)

# ---- 批量推理（一次性传入所有样本） ----
sampling_params = SamplingParams(temperature=0, max_tokens=1024)
outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

# ---- 输出结果 ----
for i, output in enumerate(outputs):
    print(f"\n--- 样本 {i+1} ---")
    print(f"问题: {samples[i]['question']}")
    print(f"回答: {output.outputs[0].text}")
```

### 5.3 图片推理

```python
"""图片推理示例"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/image.jpg",  # 本地路径或 URL
            },
            {"type": "text", "text": "请描述这张图片的内容。"},
        ],
    }
]

# 其余流程与视频推理完全相同，只需将 limit_mm_per_prompt 改为：
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},  # 图片模式
)
```

---

## 6. Thinking 模式

Qwen3-VL 的 Thinking 变体（如 `Qwen3-VL-2B-Thinking`）支持在输出答案前先进行推理思考。

### 6.1 启用/禁用 Thinking

通过 `apply_chat_template` 的 `enable_thinking` 参数控制：

```python
# 启用思考模式（默认行为）
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,          # 模型会先输出 <think>...</think> 再回答
)

# 禁用思考模式（直接回答，更快）
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,         # 模型直接输出答案
)
```

### 6.2 Thinking 模式的输出格式

```
<think>
首先，我需要观察视频中的场景...
视频开头展示了一个公园的画面...
我可以看到有三个人在跑步...
综合以上观察，答案应该是 B。
</think>
<answer>
B
</answer>
```

### 6.3 Thinking 模式推荐参数

```python
sampling_params = SamplingParams(
    temperature=0.6,      # Thinking 模型推荐使用非零温度
    top_p=0.95,
    top_k=20,
    max_tokens=32768,     # 思考过程可能很长，需要足够的生成空间
)
```

---

## 7. 关键参数参考

### 7.1 LLM 构造参数

```python
LLM(
    # === 基础配置 ===
    model="Qwen/Qwen3-VL-8B-Instruct",  # 模型路径
    trust_remote_code=True,               # 信任远程代码

    # === GPU 资源管理 ===
    tensor_parallel_size=1,               # 张量并行 GPU 数
    gpu_memory_utilization=0.80,          # 显存利用率 (0-1)
    max_model_len=4096,                   # 最大序列长度（可减小以节省显存）

    # === 多模态限制 ===
    limit_mm_per_prompt={"video": 1},     # 每个 prompt 最多包含的媒体数

    # === 性能调优 ===
    max_num_seqs=5,                       # 最大并发序列数
    enforce_eager=False,                  # True=禁用CUDA Graph（调试模式）

    # === 视觉处理默认参数（引擎级别） ===
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "fps": 1,
    },
)
```

### 7.2 SamplingParams

```python
SamplingParams(
    temperature=0,           # 0=贪心解码，>0=随机采样
    top_p=1.0,               # 核采样概率
    top_k=-1,                # Top-K 采样（-1=禁用）
    max_tokens=1024,         # 最大生成 token 数
    repetition_penalty=1.0,  # 重复惩罚
    stop=None,               # 停止字符串列表
    stop_token_ids=None,     # 停止 token id 列表
)
```

### 7.3 视频处理参数

在 messages 的 video 内容块中设置：

```python
{
    "type": "video",
    "video": "path.mp4",
    "fps": 1.0,                    # 帧率：每秒采样多少帧
    "max_frames": 32,              # 上限：最多采样多少帧
    "max_pixels": 360 * 420,       # 每帧最大像素数
    "total_pixels": 20480 * 28 * 28,  # 所有帧的总像素预算
    "min_pixels": 16 * 28 * 28,    # 每帧最小像素数
}
```

> **帧数计算公式**：`实际采样帧数 = min(max_frames, video_duration × fps)`

---

## 8. vLLM 在线服务模式

除了离线推理，vLLM 还支持启动 OpenAI 兼容的 API 服务器。

### 8.1 启动服务

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000
```

Thinking 模型：

```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --host 0.0.0.0 \
    --port 8000
```

### 8.2 客户端调用

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "file:///path/to/video.mp4"},
                },
                {"type": "text", "text": "描述视频内容"},
            ],
        }
    ],
    max_tokens=1024,
    extra_body={
        "mm_processor_kwargs": {"fps": 1.0}
    },
)

print(response.choices[0].message.content)
```

> **注意**：在线模式使用 `video_url` 而非 `video`，路径需加 `file://` 前缀。

---

## 9. 常见问题与排查

### Q1: OOM（显存不足）

**解决方案**（按优先级）：

1. 减小 `max_model_len`（如设为 `4096` 或 `8192`）
2. 降低 `gpu_memory_utilization`（如 `0.70`）
3. 减少视频帧数（降低 `fps` 或 `max_frames`）
4. 减少 `max_pixels` 以降低单帧分辨率
5. 增加 `tensor_parallel_size`（使用更多 GPU）
6. 使用 FP8 量化模型（如 `Qwen3-VL-8B-Instruct-FP8`）

### Q2: 多 GPU 推理报错

确保设置了环境变量：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

且 `CUDA_VISIBLE_DEVICES` 包含了 `tensor_parallel_size` 指定数量的 GPU。

### Q3: 视频读取失败

确认：
- 视频文件路径正确且文件存在
- 安装了 `decord` 或 `av`（视频解码库）
- 视频编码格式受支持（H.264/H.265）

```bash
pip install decord
# 或
pip install av
```

### Q4: process_vision_info 缺少 return_video_kwargs 参数

升级 `qwen-vl-utils` 到 `>= 0.0.14`：

```bash
pip install -U qwen-vl-utils
```

### Q5: vLLM 版本兼容性

Qwen3-VL 需要 vLLM >= 0.8.5。检查版本：

```bash
python -c "import vllm; print(vllm.__version__)"
```

### Q6: 批量推理时部分样本失败

建议在准备输入阶段 try/except 捕获错误，将失败样本单独记录，确保其他样本正常推理。参见本项目 `eval_videoqa.py` 中的实现。

---

## 参考资料

- [vLLM 官方文档 - Qwen3-VL 使用指南](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [vLLM 多模态输入文档](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [vLLM 离线推理示例](https://docs.vllm.ai/en/stable/examples/offline_inference/vision_language/)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen 官方 vLLM 部署文档](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
