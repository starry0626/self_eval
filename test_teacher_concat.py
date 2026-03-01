#!/usr/bin/env python3
"""
教师模型额外输入拼接逻辑验证脚本

验证步骤：
1. 从数据集加载第一条样本
2. 构建学生输入（原始 prompt + 视频）
3. 构建教师额外上下文（build_extra_context_only，不含原始视频）
4. 将额外上下文拼接进学生输入（插入 user 消息内部）
5. 打印学生 / 教师的 tensor 结构及解码文本

运行方式（在 self_eval/ 目录下）：
    python test_teacher_concat.py
"""

import os
import sys
import json
import torch

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from open_r1_video.trainer.teacher_context_builder import TeacherContextConfig, TeacherContextBuilder

# ======================== 路径配置（按实际情况修改）========================
MODEL_PATH     = "./Qwen3-VL-2B-Thinking"
DATA_PATH      = "./dataset/video-r2/Video-R2/video-r2-grpo-dataset.json"
VIDEO_BASE_DIR = "./video_data/videos"
IS_QWEN3_VL    = True   # Qwen3-VL 需要传 video_metadata；Qwen2-VL 设为 False

# 教师额外输入配置（与 train_sdpo.sh 保持一致）
INCLUDE_ANSWER         = True
INCLUDE_TEMPORAL_TEXT  = True
INCLUDE_TEMPORAL_VIDEO = True
INCLUDE_REASONING      = True
TEMPORAL_FPS           = 1.0
TEMPORAL_MAX_FRAMES    = 8
MAX_TEMPORAL_SEGMENTS  = 5
# ==========================================================================

VIDEO_QA_PROMPT = (
    "You are a video understanding assistant. Please analyze the provided video "
    "and answer the multiple-choice question.\n\n"
    "IMPORTANT: You MUST follow this exact format:\n"
    "1. First, enclose your step-by-step thinking process within <think> and </think> tags\n"
    "2. Then provide your final answer choice enclosed in <answer> and </answer> tags\n\n"
    "Required format:\n<think>\nYour detailed reasoning process here...\n</think>\n"
    "<answer>\nA/B/C/D\n</answer>\n\n"
    "Question: {question}\nOptions:\n{options}\n\n"
    "Note: The video duration is approximately {duration} seconds."
)

SEP = "=" * 70


def print_tensor_dict(d: dict, title: str):
    """打印 tensor 字典的结构（形状 + 首尾数值）"""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  keys = {list(d.keys())}")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"  ['{k}']  shape={tuple(v.shape)}  dtype={v.dtype}")
            flat = v.flatten()
            print(f"           values = [{flat[0].item():.4f}, {flat[1].item():.4f}, ..., "
                  f"{flat[-2].item():.4f}, {flat[-1].item():.4f}]")
        else:
            print(f"  ['{k}']  type={type(v).__name__}")


def compress_visual_padding(text: str) -> str:
    """
    将解码文本中连续重复的视觉 padding token 压缩为 <|token|>*N 格式。

    例如：
        <|image_pad|><|image_pad|><|image_pad|>  →  <|image_pad|>*3
        <|video_pad|><|video_pad|>               →  <|video_pad|>*2

    单个 token（不重复）保持原样，不加 *1 后缀。
    """
    import re

    def _replace_run(match: re.Match) -> str:
        token = match.group(1)           # 被重复的那个 token，如 <|image_pad|>
        count = len(match.group(0)) // len(token)
        return token if count == 1 else f"{token}*{count}"

    # 匹配连续重复的视觉 padding token（一次覆盖 image 和 video 两种）
    pattern = r"(<\|(?:image_pad|video_pad)\|>)+"
    return re.sub(pattern, _replace_run, text)


def decode_ids(tokenizer, ids: torch.Tensor, title: str):
    """解码 input_ids 并打印，连续视觉 padding 压缩为 token*N 格式"""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    text = compress_visual_padding(text)
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(text)


# ───────────────────────────────────────────────────────────────
# 1. 加载 Processor（无需加载模型权重）
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 1: Loading processor (no model weights needed)...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = getattr(processor, "tokenizer", processor)

# ───────────────────────────────────────────────────────────────
# 2. 预计算后缀长度（复现 __init__ 中的逻辑）
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 2: Computing suffix lengths...")

_dummy = [{"role": "user", "content": "a"}]
_ids_with    = tokenizer.encode(
    processor.apply_chat_template(_dummy, tokenize=False, add_generation_prompt=True),
    add_special_tokens=False)
_ids_without = tokenizer.encode(
    processor.apply_chat_template(_dummy, tokenize=False, add_generation_prompt=False),
    add_special_tokens=False)
generation_prompt_len = len(_ids_with) - len(_ids_without)

_text_a = processor.apply_chat_template(
    [{"role": "user", "content": "ab"}], tokenize=False, add_generation_prompt=False)
_text_b = processor.apply_chat_template(
    [{"role": "user", "content": "abc"}], tokenize=False, add_generation_prompt=False)
_suffix_a = _text_a[_text_a.rfind("ab")  + len("ab") :]
_suffix_b = _text_b[_text_b.rfind("abc") + len("abc"):]
assert _suffix_a == _suffix_b, f"suffix mismatch: {repr(_suffix_a)} vs {repr(_suffix_b)}"
user_end_len = len(tokenizer.encode(_suffix_a, add_special_tokens=False))
teacher_suffix_len = user_end_len + generation_prompt_len

print(f"  generation_prompt_len = {generation_prompt_len}  "
      f"({repr(tokenizer.decode(_ids_with[-generation_prompt_len:]))})")
print(f"  user_end_len          = {user_end_len}  ({repr(_suffix_a)})")
print(f"  teacher_suffix_len    = {teacher_suffix_len}")

# ───────────────────────────────────────────────────────────────
# 3. 加载第一条样本
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 3: Loading sample...")
with open(DATA_PATH) as f:
    sample = json.load(f)[0]

print(f"  sample id    = {sample.get('id')}")
print(f"  video        = {sample.get('video')}")
print(f"  temporal_grounding = {json.dumps(sample.get('temporal_grounding'), ensure_ascii=False, indent=4)}")

# ───────────────────────────────────────────────────────────────
# 4. 构建学生 prompt（复现 make_conversation_video 逻辑）
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 4: Building student prompt messages...")

conversations = sample.get("conversations", [])
raw_question  = conversations[0].get("value", "") if conversations else ""
options       = sample.get("options", [])
options_text  = "\n".join(options) if isinstance(options, list) else str(options)
duration      = sample.get("duration", "unknown")
prompt_text   = VIDEO_QA_PROMPT.format(
    question=raw_question, options=options_text, duration=duration)

video_rel = sample.get("video", "")
video_path = os.path.join(VIDEO_BASE_DIR, video_rel) if VIDEO_BASE_DIR and not os.path.isabs(video_rel) else video_rel
print(f"  full video path = {video_path}")
print(f"  video exists    = {os.path.exists(video_path)}")

prompt_messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 1, "max_frames": 32,
             "max_pixels": 128 * 32 * 32},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# ───────────────────────────────────────────────────────────────
# 5. 处理学生输入（processor）
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 5: Processing student inputs...")

image_inputs, video_inputs, video_kwargs = process_vision_info(
    prompt_messages,
    return_video_kwargs=True,
    return_video_metadata=IS_QWEN3_VL,
)
student_text = processor.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True)

if IS_QWEN3_VL:
    videos, video_metas = zip(*video_inputs) if video_inputs else ([], [])
    student_inputs = processor(
        text=[student_text],
        images=image_inputs,
        videos=list(videos) or None,
        video_metadata=list(video_metas) or None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        **video_kwargs,
    )
else:
    student_inputs = processor(
        text=[student_text],
        images=image_inputs,
        videos=video_inputs or None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        **video_kwargs,
    )

print_tensor_dict(student_inputs, "STUDENT inputs structure")
decode_ids(tokenizer, student_inputs["input_ids"][0], "STUDENT decoded text")

# ───────────────────────────────────────────────────────────────
# 6. 构建教师额外上下文
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 6: Building teacher extra context...")

config = TeacherContextConfig(
    include_answer        = INCLUDE_ANSWER,
    include_temporal_text = INCLUDE_TEMPORAL_TEXT,
    include_temporal_video= INCLUDE_TEMPORAL_VIDEO,
    include_reasoning     = INCLUDE_REASONING,
    temporal_fps          = TEMPORAL_FPS,
    temporal_max_frames   = TEMPORAL_MAX_FRAMES,
    max_temporal_segments = MAX_TEMPORAL_SEGMENTS,
)
builder = TeacherContextBuilder(config)
print(f"  config: {builder.get_context_description()}")

# question_text 从 student prompt 的 text 部分提取，用于解析选项
qt_for_options = next(
    (item["text"] for item in prompt_messages[0]["content"] if item.get("type") == "text"), "")

extra_msg = builder.build_extra_context_only(
    conversations    = sample.get("conversations", []),
    question_text    = qt_for_options,
    temporal_grounding = sample.get("temporal_grounding"),
    video_path       = video_path,   # 已是完整路径
    video_base_dir   = None,         # Bug 修复：不再二次拼接
)

if not extra_msg or not extra_msg[0].get("content"):
    print("  [WARNING] extra_msg is empty — check temporal_grounding / video path")
    sys.exit(1)

content_summary = [(item["type"], item.get("text", "<image>")[:60] if item.get("type") == "text" else "<PIL Image>")
                   for item in extra_msg[0]["content"]]
print(f"  extra content items ({len(extra_msg[0]['content'])}):")
for t, v in content_summary:
    print(f"    [{t}] {v!r}")

# ───────────────────────────────────────────────────────────────
# 7. 处理额外上下文（裸文本，不走 apply_chat_template）
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 7: Tokenizing extra context (raw text, no chat_template)...")

extra_image_inputs, _, _ = process_vision_info(
    extra_msg,
    return_video_kwargs=True,
    return_video_metadata=IS_QWEN3_VL,
)

raw_text = ""
for item in extra_msg[0]["content"]:
    if item.get("type") == "text":
        raw_text += item.get("text", "")
    elif item.get("type") == "image":
        raw_text += "<|vision_start|><|image_pad|><|vision_end|>"

print(f"  raw_text (first 200 chars): {raw_text[:200]!r}")

extra_inputs = processor(
    text=[raw_text],
    images=extra_image_inputs,
    videos=None,
    return_tensors="pt",
    padding=True,
    padding_side="left",
)

print_tensor_dict(extra_inputs, "EXTRA CONTEXT inputs structure")
decode_ids(tokenizer, extra_inputs["input_ids"][0], "EXTRA CONTEXT decoded text")

# ───────────────────────────────────────────────────────────────
# 8. 拼接：将额外上下文插入 user 消息内部
# ───────────────────────────────────────────────────────────────
print(SEP)
print("Step 8: Concatenating student + extra context...")

s_ids    = student_inputs["input_ids"][0]
s_mask   = student_inputs["attention_mask"][0].bool()
s_no_pad = s_ids[s_mask]

e_ids    = extra_inputs["input_ids"][0]
e_mask   = extra_inputs["attention_mask"][0].bool()
e_no_pad = e_ids[e_mask]

print(f"  student tokens (no pad): {s_no_pad.shape[0]}")
print(f"  extra   tokens (no pad): {e_no_pad.shape[0]}")
print(f"  teacher_suffix_len      : {teacher_suffix_len}")

# 验证后缀 token 是否符合预期
suffix_tokens = s_no_pad[-teacher_suffix_len:]
print(f"  suffix decoded          : {tokenizer.decode(suffix_tokens)!r}")

# 拼接：[问题文本] + [额外上下文] + [<|im_end|>\n<|im_start|>assistant\n]
student_main = s_no_pad[:-teacher_suffix_len]
suffix       = s_no_pad[-teacher_suffix_len:]
concat_ids   = torch.cat([student_main, e_no_pad, suffix], dim=0)
concat_mask  = torch.ones(concat_ids.size(0), dtype=torch.long)

teacher_input_ids       = concat_ids.unsqueeze(0)
teacher_attention_mask  = concat_mask.unsqueeze(0)

# 组装教师 tensor 字典
teacher_prompt_inputs = {
    "input_ids":       teacher_input_ids,
    "attention_mask":  teacher_attention_mask,
}
for key in ["pixel_values_videos", "video_grid_thw"]:   # 视频视觉信息来自学生
    if key in student_inputs:
        teacher_prompt_inputs[key] = student_inputs[key]
for key in ["pixel_values", "image_grid_thw"]:           # 图像视觉信息来自额外上下文
    if key in extra_inputs:
        teacher_prompt_inputs[key] = extra_inputs[key]

print_tensor_dict(teacher_prompt_inputs, "TEACHER inputs structure")

# ───────────────────────────────────────────────────────────────
# 9. 解码教师输入，验证结构
# ───────────────────────────────────────────────────────────────
decode_ids(tokenizer, teacher_input_ids[0], "TEACHER decoded text (verify concatenation)")

# 额外校验：对比学生/教师长度差异
print(SEP)
print("Summary:")
print(f"  Student prompt tokens : {s_no_pad.shape[0]}")
print(f"  Extra context tokens  : {e_no_pad.shape[0]}")
print(f"  Teacher prompt tokens : {concat_ids.shape[0]}")
print(f"  Difference            : {concat_ids.shape[0] - s_no_pad.shape[0]} "
      f"(should equal extra context tokens = {e_no_pad.shape[0]})")
assert concat_ids.shape[0] == s_no_pad.shape[0] + e_no_pad.shape[0], "Length mismatch!"
print("  [PASS] Token count assertion passed.")
print(SEP)
