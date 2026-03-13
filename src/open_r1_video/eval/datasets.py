"""
VideoQA 评估数据集适配器

每个数据集提供一个 build_messages 函数，将 JSON 样本转换为模型输入所需的 messages 格式。

所有函数签名统一为:
    build_messages(sample, video_base_dir, fps, max_frames, max_pixels, answer_mode)
    -> (messages: list, gt_answer: str, sample_id: str)

answer_mode:
    "think"  - 先思考后回答，模型输出 <think>...</think><answer>X</answer>，从 XML 标签提取答案
    "direct" - 直接回答，模型直接输出选项字母或数字

支持的数据集（均位于 src/open_r1_video/eval/ 目录）:
    mmvu        - 科学领域视频多选题（A-E，5选项）
    mvbench     - 动作/运动理解视频多选题（A-C，3选项）
    tempcompass - 时序理解视频二选题（A-B，维度: action/direction/...）
    videomme    - 通用视频理解多选题（A-D，short/medium/long 时长分类）
    videommmu   - 学术领域视频多选题（A-J，最多10选项）
    vsibench    - 空间理解视频回归题（物体计数，无选项，数值答案）
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ======================== 提示词模板 ========================

# 思考模式：多选题
MC_THINK_PROMPT = """\
Please analyze the provided video carefully and answer the multiple-choice question strictly based on the content of the video.

You MUST first provide a detailed step-by-step reasoning process within the <think> and </think> tags.
You MUST then provide your final answer within the <answer> and </answer> tags.
The <answer> tag must contain ONLY the single option letter (e.g., A, B, C, or D) with no other text or explanation.

You MUST follow this exact format:
<think>
Your detailed step-by-step reasoning process here...
</think>
<answer>
X
</answer>
where X is the single option letter (A, B, C, D, etc.) with no additional text.

Question: {question}
Options:
{options}"""

# 直接回答模式：多选题
MC_DIRECT_PROMPT = """\
Please analyze the provided video and answer the multiple-choice question strictly based on the content of the video.

You MUST follow this exact format:
<think>
Your analysis here...
</think>
<answer>
X
</answer>
where X is the single option letter (A, B, C, D, etc.) with no additional text.

Question: {question}
Options:
{options}"""

# 思考模式：回归/计数题
REGRESSION_THINK_PROMPT = """\
Please analyze the provided video carefully and answer the question strictly based on the content of the video.

You MUST follow this exact format:
<think>
Your detailed step-by-step reasoning process here...
</think>
<answer>
[your numeric answer]
</answer>
Output only a single number in the answer tags, with no additional text.

Question: {question}"""

# 直接回答模式：回归/计数题
REGRESSION_DIRECT_PROMPT = """\
Please analyze the provided video carefully and answer the question with a number strictly based on the content of the video.

You MUST follow this exact format:
<think>
Your analysis here...
</think>
<answer>
[your numeric answer]
</answer>
Output only a single number in the answer tags, with no additional text.

Question: {question}"""


# ======================== 答案提取与评估 ========================

def extract_gt_answer(solution: str) -> str:
    """从 solution 字段提取标准答案（去除 XML 标签）"""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", solution, re.DOTALL)
    if m:
        return m.group(1).strip()
    return solution.strip()


def extract_pred_answer(response: str, answer_mode: str, problem_type: str = "multiple choice") -> str:
    """
    从模型输出中提取预测答案

    think 模式：优先从 <answer> 标签提取，fallback 到 </think> 之后的文本
    direct 模式：先尝试 <answer> 标签，再 fallback 到首个大写字母或数字
    """
    # 优先尝试 <answer> 标签（think 和 direct 模式都支持）
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    if answer_mode == "think":
        # fallback: 从 </think> 之后的文本中提取答案
        think_end = re.search(r"</think>\s*", response, re.IGNORECASE)
        if think_end:
            after_think = response[think_end.end():]
            if problem_type == "regression":
                m = re.search(r"\b(\d+(?:\.\d+)?)\b", after_think)
                if m:
                    return m.group(1)
            else:
                m = re.search(r"\b([A-J])\b", after_think)
                if m:
                    return m.group(1)

    if answer_mode == "direct":
        # fallback: 首个大写字母（多选题）
        m = re.search(r"\b([A-J])\b", response)
        if m:
            return m.group(1)
        # fallback: 首个数字（回归题）
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if m:
            return m.group(1)

    return ""


def compute_accuracy(pred: str, gt: str, problem_type: str = "multiple choice") -> float:
    """
    计算单样本准确率

    multiple choice: 字母精确匹配（大小写不敏感）
    regression:      数值精确匹配（容忍 ±0.5，即整数计数完全匹配）
    """
    if not pred or not gt:
        return 0.0

    if problem_type == "regression":
        try:
            return 1.0 if abs(float(pred) - float(gt)) <= 0.5 else 0.0
        except ValueError:
            return 0.0
    else:
        return 1.0 if pred.strip().upper() == gt.strip().upper() else 0.0


# ======================== 通用工具函数 ========================

def _resolve_video_path(raw_path: str, video_base_dir: Optional[str], check_exists: bool = False) -> str:
    """
    解析视频路径：若为相对路径且指定了 base_dir，则拼接并规范化。

    数据集中的 path 字段通常以 "./" 开头（如 "./Evaluation/VideoMME/xxx.mp4"），
    直接 os.path.join 会产生 "base/./Evaluation/..." 形式的冗余路径。
    使用 os.path.normpath 消除中间的 "./" 和 ".."，确保路径干净。

    check_exists: 若为 True，检查路径是否存在，不存在则抛出 FileNotFoundError。
    """
    if video_base_dir and not os.path.isabs(raw_path):
        resolved = os.path.normpath(os.path.join(video_base_dir, raw_path))
    else:
        resolved = raw_path

    if check_exists and not os.path.exists(resolved):
        raise FileNotFoundError(f"视频文件不存在: {resolved}")

    return resolved


def _video_content(video_path: str, fps: float, max_frames: int, max_pixels: Optional[int]) -> Dict:
    """构造 messages 中的视频内容字典"""
    content: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "fps": fps,
        "max_frames": max_frames,
    }
    if max_pixels is not None:
        content["max_pixels"] = max_pixels
    return content


def _mc_messages(
    video_path: str,
    prompt_text: str,
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
) -> List[Dict]:
    """构造标准多选题单轮对话 messages"""
    return [
        {
            "role": "user",
            "content": [
                _video_content(video_path, fps, max_frames, max_pixels),
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _regression_messages(
    video_path: str,
    prompt_text: str,
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
) -> List[Dict]:
    """构造回归/计数题单轮对话 messages（与多选题结构相同，仅语义不同）"""
    return [
        {
            "role": "user",
            "content": [
                _video_content(video_path, fps, max_frames, max_pixels),
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


# ======================== 各数据集构建函数 ========================

def build_mmvu_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    MMVU — 科学领域视频多选题（最多 5 选项 A-E）

    样本格式:
        problem_id, problem, options (list), solution (<answer>X</answer>), path
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)
    options_text = "\n".join(sample["options"])

    tmpl = MC_THINK_PROMPT if answer_mode == "think" else MC_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"], options=options_text)

    messages = _mc_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


def build_mvbench_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    MVBench — 动作/运动理解视频多选题（A-C，3 选项）

    样本格式:
        problem_id, problem, options (list), solution (<answer>X</answer>), path
        额外字段: answer（选项全文）, subtitle
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)
    options_text = "\n".join(sample["options"])

    tmpl = MC_THINK_PROMPT if answer_mode == "think" else MC_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"], options=options_text)

    messages = _mc_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


def build_tempcompass_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    TempCompass — 时序理解视频二选题（A-B）

    样本格式:
        problem_id, problem, options (list), solution (<answer>X</answer>), path
        额外字段: dim（action/direction/...），用于按维度分析结果
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)
    options_text = "\n".join(sample["options"])

    tmpl = MC_THINK_PROMPT if answer_mode == "think" else MC_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"], options=options_text)

    messages = _mc_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


def build_videomme_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    VideoMME — 通用视频理解多选题（A-D，分 short/medium/long 时长类别）

    样本格式:
        problem_id, problem, options (list), solution (<answer>X</answer>), path
        额外字段: duration, domain, sub_category, task_type
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)
    options_text = "\n".join(sample["options"])

    tmpl = MC_THINK_PROMPT if answer_mode == "think" else MC_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"], options=options_text)

    messages = _mc_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


def build_videommmu_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    VideoMMMU — 学术领域视频多选题（最多 10 选项 A-J）

    样本格式:
        problem_id, problem, options (list), solution (<answer>X</answer>), path
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)
    options_text = "\n".join(sample["options"])

    tmpl = MC_THINK_PROMPT if answer_mode == "think" else MC_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"], options=options_text)

    messages = _mc_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


def build_vsibench_messages(
    sample: Dict[str, Any],
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> Tuple[List[Dict], str, str]:
    """
    VSIBench — 空间理解视频回归题（物体计数，无选项，数值答案）

    样本格式:
        problem_id, problem, options (null), solution (<answer>N</answer>), path
        problem_type = "regression"，答案为整数

    准确率计算: |pred - gt| <= 0.5（即整数精确匹配）
    """
    video_path = _resolve_video_path(sample["path"], video_base_dir)

    tmpl = REGRESSION_THINK_PROMPT if answer_mode == "think" else REGRESSION_DIRECT_PROMPT
    prompt_text = tmpl.format(question=sample["problem"])

    messages = _regression_messages(video_path, prompt_text, fps, max_frames, max_pixels)
    gt_answer = extract_gt_answer(sample["solution"])
    sample_id = str(sample.get("problem_id", ""))

    return messages, gt_answer, sample_id


# ======================== 数据集注册表 ========================

DATASET_BUILDERS = {
    "mmvu": build_mmvu_messages,
    "mvbench": build_mvbench_messages,
    "tempcompass": build_tempcompass_messages,
    "videomme": build_videomme_messages,
    "videommmu": build_videommmu_messages,
    "vsibench": build_vsibench_messages,
}


def check_video_paths(dataset: List[Dict], video_base_dir: Optional[str], output_dir: Optional[str] = None) -> None:
    """
    预检查数据集中所有视频路径是否存在。

    在模型加载之前调用，提前发现缺失的视频文件，
    避免模型加载完成后才因文件不存在而报错。

    若指定 output_dir，会将所有缺失路径写入 output_dir/missing_videos.txt。
    """
    missing = []
    for sample in dataset:
        resolved = _resolve_video_path(sample["path"], video_base_dir)
        if not os.path.exists(resolved):
            missing.append(resolved)

    if missing:
        # 将完整的缺失路径列表写入文件
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            missing_file = os.path.join(output_dir, "missing_videos.txt")
            with open(missing_file, "w", encoding="utf-8") as f:
                for p in missing:
                    f.write(p + "\n")
            print(f"完整缺失路径列表已写入: {missing_file}")

        msg = f"视频路径检查失败: {len(missing)} 个文件不存在\n"
        for p in missing[:20]:
            msg += f"  - {p}\n"
        if len(missing) > 20:
            msg += f"  ... 及其他 {len(missing) - 20} 个文件\n"
        raise FileNotFoundError(msg)
