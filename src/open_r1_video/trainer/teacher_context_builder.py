"""
教师上下文构建模块

本模块用于构建教师模型的额外输入上下文，支持以下四种类型的额外信息：
1. 标准答案（选项及对应内容）
2. temporal_grounding 项中的内容
3. temporal_grounding 时间段对应的视频片段
4. 推理流程

每种信息都可以通过配置参数独立控制是否包含。
"""

import re
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class TeacherContextConfig:
    """
    教师上下文配置类
    
    用于控制教师模型额外输入的各部分是否包含
    
    属性:
        include_answer: 是否包含标准答案
        include_temporal_text: 是否包含 temporal_grounding 的文本描述
        include_temporal_video: 是否包含 temporal_grounding 时间段的视频片段
        include_reasoning: 是否包含推理流程
        temporal_frames_count: 时间段采样时的帧数（等间隔采样）
        point_frames_count: 时间点采样时的帧数（前后采样）
        point_frames_range: 时间点采样的前后范围（秒）
        max_temporal_segments: 最大时间片段数量（避免过长）
    """
    include_answer: bool = True
    include_temporal_text: bool = True
    include_temporal_video: bool = False
    include_reasoning: bool = True
    
    temporal_frames_count: int = 4
    point_frames_count: int = 4
    point_frames_range: float = 1.0
    max_temporal_segments: int = 5


def parse_timestamp(timestamp_str: str) -> Tuple[float, float]:
    """
    解析时间戳字符串，返回开始时间和结束时间（秒）
    
    支持的格式：
    - "00:20" -> 单个时间点，返回 (20.0, 20.0)
    - "00:20-00:48" -> 时间段，返回 (20.0, 48.0)
    - "01:06" -> 单个时间点，返回 (66.0, 66.0)
    
    参数:
        timestamp_str: 时间戳字符串
    
    返回:
        (start_time, end_time): 开始时间和结束时间（秒）
    """
    def time_str_to_seconds(time_str: str) -> float:
        """将时间字符串转换为秒数"""
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    
    if '-' in timestamp_str:
        start_str, end_str = timestamp_str.split('-')
        return time_str_to_seconds(start_str), time_str_to_seconds(end_str)
    else:
        time_val = time_str_to_seconds(timestamp_str)
        return time_val, time_val


def extract_answer_from_conversation(conversations: List[Dict]) -> Tuple[str, str]:
    """
    从对话中提取标准答案
    
    参数:
        conversations: 对话列表，格式为 [{"from": "human/assistant", "value": "..."}]
    
    返回:
        (answer_text, answer_letter): 答案文本和答案字母（如 "A", "B" 等）
    """
    answer_text = ""
    answer_letter = ""
    
    for conv in conversations:
        if conv.get("from") == "assistant":
            value = conv.get("value", "")
            
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', value, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                answer_text = value.strip()
            
            letter_match = re.search(r'\b([A-E])\b', answer_text)
            if letter_match:
                answer_letter = letter_match.group(1)
            
            break
    
    return answer_text, answer_letter


def extract_question_and_options(question_text: str) -> Tuple[str, Dict[str, str]]:
    """
    从问题文本中提取问题和选项
    
    参数:
        question_text: 问题文本，包含问题和选项
    
    返回:
        (question, options): 问题和选项字典 {"A": "选项内容", "B": "选项内容", ...}
    """
    options = {}
    
    lines = question_text.strip().split('\n')
    question_lines = []
    option_pattern = re.compile(r'^([A-E])[\.\、\)\s]+(.+)$')
    
    for line in lines:
        line = line.strip()
        match = option_pattern.match(line)
        if match:
            option_letter = match.group(1)
            option_content = match.group(2).strip()
            options[option_letter] = option_content
        elif not line.startswith('Options') and not line.startswith('Please think'):
            if not options:
                question_lines.append(line)
    
    question = ' '.join(question_lines).strip()
    
    return question, options


def extract_reasoning_from_conversation(conversations: List[Dict]) -> str:
    """
    从对话中提取推理流程
    
    参数:
        conversations: 对话列表
    
    返回:
        reasoning: 推理流程文本
    """
    reasoning = ""
    
    for conv in conversations:
        if conv.get("from") == "assistant":
            value = conv.get("value", "")
            
            think_match = re.search(r'<think[^>]*>(.*?)</think[^>]*>', value, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
            
            break
    
    return reasoning


def extract_temporal_grounding_text(temporal_grounding: Dict[str, str]) -> List[Dict[str, str]]:
    """
    提取 temporal_grounding 的文本内容
    
    参数:
        temporal_grounding: 时间定位字典，格式为 {"00:20-00:48": "描述", ...}
    
    返回:
        segments: 时间片段列表，格式为 [{"timestamp": "00:20-00:48", "description": "描述"}, ...]
    """
    if not temporal_grounding:
        return []
    
    segments = []
    for timestamp, description in temporal_grounding.items():
        if description is not None:
            segments.append({
                "timestamp": timestamp,
                "description": description
            })
    
    return segments


def sample_frames_from_video(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int,
    is_point: bool = False,
    point_range: float = 1.0
) -> List[Tuple[np.ndarray, float]]:
    """
    从视频中采样帧，返回帧和对应的时间戳
    
    参数:
        video_path: 视频文件路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        num_frames: 采样帧数
        is_point: 是否为时间点采样
        point_range: 时间点采样的前后范围（秒）
    
    返回:
        frames_with_timestamps: 采样帧列表，每个元素为 (frame, timestamp) 元组
    """
    if not os.path.exists(video_path):
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    frames_with_timestamps = []
    
    if is_point:
        center_time = start_time
        actual_start = max(0, center_time - point_range)
        actual_end = center_time + point_range
        
        time_points = np.linspace(actual_start, actual_end, num_frames).tolist()
    else:
        if start_time == end_time:
            time_points = [start_time]
        else:
            time_points = np.linspace(start_time, end_time, num_frames).tolist()
    
    for time_point in time_points:
        frame_number = int(time_point * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            frames_with_timestamps.append((frame, time_point))
    
    cap.release()
    
    return frames_with_timestamps


def sample_temporal_video_segments(
    video_path: str,
    temporal_grounding: Dict[str, str],
    config: TeacherContextConfig
) -> List[Dict[str, Any]]:
    """
    采样 temporal_grounding 时间段的视频帧
    
    参数:
        video_path: 视频文件路径
        temporal_grounding: 时间定位字典
        config: 教师上下文配置
    
    返回:
        video_segments: 视频片段列表，每个片段包含时间戳、描述和采样帧（带时间戳）
    """
    if not temporal_grounding or not os.path.exists(video_path):
        return []
    
    video_segments = []
    count = 0
    
    for timestamp, description in temporal_grounding.items():
        if description is None:
            continue
        
        if count >= config.max_temporal_segments:
            break
        
        start_time, end_time = parse_timestamp(timestamp)
        is_point = start_time == end_time
        
        if is_point:
            num_frames = config.point_frames_count
            frames_with_ts = sample_frames_from_video(
                video_path, start_time, end_time, num_frames,
                is_point=True, point_range=config.point_frames_range
            )
        else:
            num_frames = config.temporal_frames_count
            frames_with_ts = sample_frames_from_video(
                video_path, start_time, end_time, num_frames,
                is_point=False
            )
        
        if frames_with_ts:
            video_segments.append({
                "timestamp": timestamp,
                "description": description,
                "frames_with_timestamps": frames_with_ts,
                "is_point": is_point
            })
            count += 1
    
    return video_segments


def build_teacher_prompt(
    original_prompt_messages: List[Dict],
    config: TeacherContextConfig,
    conversations: List[Dict],
    temporal_grounding: Optional[Dict[str, str]] = None,
    video_path: Optional[str] = None,
    video_base_dir: Optional[str] = None
) -> List[Dict]:
    """
    构建教师模型的输入 prompt
    
    参数:
        original_prompt_messages: 原始 prompt 消息列表
        config: 教师上下文配置
        conversations: 对话列表
        temporal_grounding: 时间定位字典
        video_path: 视频文件路径（相对路径或绝对路径）
        video_base_dir: 视频文件基础目录（如果 video_path 是相对路径）
    
    返回:
        teacher_messages: 教师模型的输入消息列表
    """
    teacher_messages = copy.deepcopy(original_prompt_messages)
    
    context_parts = []
    
    for msg in teacher_messages:
        if msg.get("role") == "user":
            original_content = msg.get("content", "")
            if isinstance(original_content, str):
                question_text = original_content
            elif isinstance(original_content, list):
                text_parts = [item.get("text", "") for item in original_content if item.get("type") == "text"]
                question_text = " ".join(text_parts)
            else:
                question_text = ""
            break
    else:
        question_text = ""
    
    if config.include_answer:
        answer_text, answer_letter = extract_answer_from_conversation(conversations)
        if answer_text:
            _, options = extract_question_and_options(question_text)
            
            if answer_letter and answer_letter in options:
                context_parts.append(f"[Correct Answer]\nThe correct answer is provided for your reference: {answer_letter}. {options[answer_letter]}")
            else:
                context_parts.append(f"[Correct Answer]\nThe correct answer is provided for your reference: {answer_text}")
    
    if config.include_temporal_text and temporal_grounding:
        temporal_segments = extract_temporal_grounding_text(temporal_grounding)
        if temporal_segments:
            temporal_text = "[Key Temporal Moments]\nThe following are key moments with their descriptions. Use them to guide your analysis:"
            for i, seg in enumerate(temporal_segments[:config.max_temporal_segments], 1):
                temporal_text += f"\n{i}. [{seg['timestamp']}]: {seg['description']}"
            context_parts.append(temporal_text)
    
    if config.include_reasoning:
        reasoning = extract_reasoning_from_conversation(conversations)
        if reasoning:
            reasoning_text = "[Reference Reasoning]\nThe following is a reference reasoning process. Learn from it to improve your reasoning:"
            reasoning_text += f"\n{reasoning}"
            context_parts.append(reasoning_text)
    
    if config.include_temporal_video and temporal_grounding and video_path:
        if video_base_dir and not os.path.isabs(video_path):
            full_video_path = os.path.join(video_base_dir, video_path)
        else:
            full_video_path = video_path
        
        video_segments = sample_temporal_video_segments(
            full_video_path, temporal_grounding, config
        )
        
        if video_segments:
            from PIL import Image
            
            for msg in teacher_messages:
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), list):
                        msg["content"].append({
                            "type": "text",
                            "text": "\n\n[Key Temporal Frames]\nThe following frames are sampled from key moments. Use them as reference:"
                        })
                        
                        for seg in video_segments:
                            for frame, timestamp in seg['frames_with_timestamps']:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_image = Image.fromarray(frame_rgb)
                                
                                msg["content"].append({
                                    "type": "text",
                                    "text": f"<{timestamp:.1f} seconds>"
                                })
                                msg["content"].append({
                                    "type": "image",
                                    "image": pil_image
                                })
                    break
    
    if context_parts:
        full_context = "\n\n".join(context_parts)
        
        for msg in teacher_messages:
            if msg.get("role") == "user":
                if isinstance(msg.get("content"), list):
                    msg["content"].append({
                        "type": "text",
                        "text": f"\n\n{full_context}"
                    })
                else:
                    msg["content"] = msg["content"] + f"\n\n{full_context}"
                break
    
    return teacher_messages


def build_extra_context_only(
    config: TeacherContextConfig,
    conversations: List[Dict],
    question_text: str = "",
    temporal_grounding: Optional[Dict[str, str]] = None,
    video_path: Optional[str] = None,
    video_base_dir: Optional[str] = None
) -> List[Dict]:
    """
    只构建教师模型的额外上下文部分（不包含原始视频和问题）
    
    用于优化教师输入处理：避免重复处理视频数据，只处理额外信息，
    然后将处理后的额外信息与学生输入进行拼接。
    
    参数:
        config: 教师上下文配置
        conversations: 对话列表
        question_text: 原始问题文本（用于提取选项）
        temporal_grounding: 时间定位字典
        video_path: 视频文件路径
        video_base_dir: 视频文件基础目录
    
    返回:
        extra_messages: 只包含额外上下文的消息列表
            格式: [{"role": "user", "content": [...]}]
            content 只包含额外的文本和图像信息
    """
    extra_content = []
    
    _, options = extract_question_and_options(question_text)
    
    if config.include_answer:
        answer_text, answer_letter = extract_answer_from_conversation(conversations)
        if answer_text:
            if answer_letter and answer_letter in options:
                extra_content.append({
                    "type": "text",
                    "text": f"\n\n[Correct Answer]\nThe correct answer is provided for your reference: {answer_letter}. {options[answer_letter]}"
                })
            else:
                extra_content.append({
                    "type": "text",
                    "text": f"\n\n[Correct Answer]\nThe correct answer is provided for your reference: {answer_text}"
                })
    
    if config.include_temporal_text and temporal_grounding:
        temporal_segments = extract_temporal_grounding_text(temporal_grounding)
        if temporal_segments:
            temporal_text = "\n\n[Key Temporal Moments]\nThe following are key moments with their descriptions. Use them to guide your analysis:"
            for i, seg in enumerate(temporal_segments[:config.max_temporal_segments], 1):
                temporal_text += f"\n{i}. [{seg['timestamp']}]: {seg['description']}"
            extra_content.append({
                "type": "text",
                "text": temporal_text
            })
    
    if config.include_reasoning:
        reasoning = extract_reasoning_from_conversation(conversations)
        if reasoning:
            reasoning_text = "\n\n[Reference Reasoning]\nThe following is a reference reasoning process. Learn from it to improve your reasoning:"
            reasoning_text += f"\n{reasoning}"
            extra_content.append({
                "type": "text",
                "text": reasoning_text
            })
    
    if config.include_temporal_video and temporal_grounding and video_path:
        if video_base_dir and not os.path.isabs(video_path):
            full_video_path = os.path.join(video_base_dir, video_path)
        else:
            full_video_path = video_path
        
        video_segments = sample_temporal_video_segments(
            full_video_path, temporal_grounding, config
        )
        
        if video_segments:
            from PIL import Image
            
            extra_content.append({
                "type": "text",
                "text": "\n\n[Key Temporal Frames]\nThe following frames are sampled from key moments. Use them as reference:"
            })
            
            for seg in video_segments:
                for frame, timestamp in seg['frames_with_timestamps']:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    extra_content.append({
                        "type": "text",
                        "text": f"<{timestamp:.1f} seconds>"
                    })
                    extra_content.append({
                        "type": "image",
                        "image": pil_image
                    })
    
    if not extra_content:
        return []
    
    return [{"role": "user", "content": extra_content}]


def build_teacher_context_dict(
    config: TeacherContextConfig,
    conversations: List[Dict],
    temporal_grounding: Optional[Dict[str, str]] = None,
    video_path: Optional[str] = None,
    video_base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    构建教师上下文字典
    
    返回一个结构化的字典，包含所有启用的上下文信息。
    可用于 _inject_teacher_context 方法的字典类型输入。
    
    参数:
        config: 教师上下文配置
        conversations: 对话列表
        temporal_grounding: 时间定位字典
        video_path: 视频文件路径
        video_base_dir: 视频文件基础目录
    
    返回:
        context_dict: 教师上下文字典
    """
    context_dict = {}
    
    if config.include_answer:
        answer_text, answer_letter = extract_answer_from_conversation(conversations)
        if answer_text:
            context_dict["answer"] = {
                "text": answer_text,
                "letter": answer_letter
            }
    
    if config.include_temporal_text and temporal_grounding:
        temporal_segments = extract_temporal_grounding_text(temporal_grounding)
        if temporal_segments:
            context_dict["temporal_text"] = temporal_segments
    
    if config.include_reasoning:
        reasoning = extract_reasoning_from_conversation(conversations)
        if reasoning:
            context_dict["reasoning"] = reasoning
    
    if config.include_temporal_video and temporal_grounding and video_path:
        if video_base_dir and not os.path.isabs(video_path):
            full_video_path = os.path.join(video_base_dir, video_path)
        else:
            full_video_path = video_path
        
        video_segments = sample_temporal_video_segments(
            full_video_path, temporal_grounding, config
        )
        if video_segments:
            context_dict["temporal_video"] = video_segments
    
    return context_dict


class TeacherContextBuilder:
    """
    教师上下文构建器
    
    主要类，用于构建教师模型的额外输入上下文。
    
    使用示例:
    ```python
    config = TeacherContextConfig(
        include_answer=True,
        include_temporal_text=True,
        include_temporal_video=False,
        include_reasoning=True
    )
    
    builder = TeacherContextBuilder(config)
    
    teacher_messages = builder.build_teacher_prompt(
        original_prompt_messages=prompt_messages,
        conversations=sample["conversations"],
        temporal_grounding=sample.get("temporal_grounding"),
        video_path=sample["video"],
        video_base_dir="./Video-R2"
    )
    ```
    """
    
    def __init__(self, config: Optional[TeacherContextConfig] = None):
        """
        初始化教师上下文构建器
        
        参数:
            config: 教师上下文配置，如果为 None 则使用默认配置
        """
        self.config = config or TeacherContextConfig()
    
    def update_config(self, **kwargs) -> None:
        """
        更新配置参数
        
        参数:
            **kwargs: 要更新的配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def build_teacher_prompt(
        self,
        original_prompt_messages: List[Dict],
        conversations: List[Dict],
        temporal_grounding: Optional[Dict[str, str]] = None,
        video_path: Optional[str] = None,
        video_base_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        构建教师模型的输入 prompt
        
        参数:
            original_prompt_messages: 原始 prompt 消息列表
            conversations: 对话列表
            temporal_grounding: 时间定位字典
            video_path: 视频文件路径
            video_base_dir: 视频文件基础目录
        
        返回:
            teacher_messages: 教师模型的输入消息列表
        """
        return build_teacher_prompt(
            original_prompt_messages=original_prompt_messages,
            config=self.config,
            conversations=conversations,
            temporal_grounding=temporal_grounding,
            video_path=video_path,
            video_base_dir=video_base_dir
        )
    
    def build_context_dict(
        self,
        conversations: List[Dict],
        temporal_grounding: Optional[Dict[str, str]] = None,
        video_path: Optional[str] = None,
        video_base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        构建教师上下文字典
        
        参数:
            conversations: 对话列表
            temporal_grounding: 时间定位字典
            video_path: 视频文件路径
            video_base_dir: 视频文件基础目录
        
        返回:
            context_dict: 教师上下文字典
        """
        return build_teacher_context_dict(
            config=self.config,
            conversations=conversations,
            temporal_grounding=temporal_grounding,
            video_path=video_path,
            video_base_dir=video_base_dir
        )
    
    def build_extra_context_only(
        self,
        conversations: List[Dict],
        question_text: str = "",
        temporal_grounding: Optional[Dict[str, str]] = None,
        video_path: Optional[str] = None,
        video_base_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        只构建教师模型的额外上下文部分（不包含原始视频和问题）
        
        用于优化教师输入处理：避免重复处理视频数据，只处理额外信息，
        然后将处理后的额外信息与学生输入进行拼接。
        
        参数:
            conversations: 对话列表
            question_text: 原始问题文本（用于提取选项）
            temporal_grounding: 时间定位字典
            video_path: 视频文件路径
            video_base_dir: 视频文件基础目录
        
        返回:
            extra_messages: 只包含额外上下文的消息列表
        """
        return build_extra_context_only(
            config=self.config,
            conversations=conversations,
            question_text=question_text,
            temporal_grounding=temporal_grounding,
            video_path=video_path,
            video_base_dir=video_base_dir
        )
    
    def get_context_description(self) -> str:
        """
        获取当前配置的上下文描述
        
        返回:
            description: 配置描述字符串
        """
        parts = []
        if self.config.include_answer:
            parts.append("Correct Answer")
        if self.config.include_temporal_text:
            parts.append("Temporal Text")
        if self.config.include_temporal_video:
            parts.append("Temporal Video")
        if self.config.include_reasoning:
            parts.append("Reasoning")
        
        if parts:
            return "Teacher context includes: " + ", ".join(parts)
        else:
            return "Teacher context is empty"


def convert_video_r2_to_sdpo_format(
    sample: Dict[str, Any],
    config: Optional[TeacherContextConfig] = None,
    video_base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    将 Video-R2 数据集样本转换为 SDPO 训练格式
    
    参数:
        sample: Video-R2 数据集样本
        config: 教师上下文配置
        video_base_dir: 视频文件基础目录
    
    返回:
        sdpo_sample: SDPO 格式的样本
    """
    if config is None:
        config = TeacherContextConfig()
    
    builder = TeacherContextBuilder(config)
    
    question_text = sample["conversations"][0]["value"]
    
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": sample["video"]},
                {"type": "text", "text": question_text}
            ]
        }
    ]
    
    teacher_messages = builder.build_teacher_prompt(
        original_prompt_messages=prompt_messages,
        conversations=sample["conversations"],
        temporal_grounding=sample.get("temporal_grounding"),
        video_path=sample["video"],
        video_base_dir=video_base_dir
    )
    
    return {
        "id": sample.get("id", ""),
        "prompt": prompt_messages,
        "teacher_prompt": teacher_messages,
        "video": sample["video"],
        "teacher_context": builder.build_context_dict(
            conversations=sample["conversations"],
            temporal_grounding=sample.get("temporal_grounding"),
            video_path=sample["video"],
            video_base_dir=video_base_dir
        )
    }


if __name__ == "__main__":
    import json
    
    sample = {
        "id": "test-001",
        "video": "test.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "Why does the person use a white cloth?\nA. To cover\nB. To clean\nC. To dry\nD. To signal\nPlease think about this question."
            },
            {
                "from": "assistant",
                "value": "<think\nOkay, I need to figure out why the person is using a white cloth.\nLet me analyze the video carefully.\n</think\n<answer\nB. To clean the area and maintain cleanliness\n</answer>"
            }
        ],
        "temporal_grounding": {
            "00:20-00:48": "He actively wipes internal surfaces with a white cloth.",
            "01:06": "He begins reassembling the components."
        }
    }
    
    config = TeacherContextConfig(
        include_answer=True,
        include_temporal_text=True,
        include_temporal_video=False,
        include_reasoning=True
    )
    
    builder = TeacherContextBuilder(config)
    
    print("=" * 60)
    print("配置描述:", builder.get_context_description())
    print("=" * 60)
    
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": sample["video"]},
                {"type": "text", "text": sample["conversations"][0]["value"]}
            ]
        }
    ]
    
    teacher_messages = builder.build_teacher_prompt(
        original_prompt_messages=prompt_messages,
        conversations=sample["conversations"],
        temporal_grounding=sample.get("temporal_grounding")
    )
    
    print("\n教师模型输入:")
    print(json.dumps(teacher_messages, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("上下文字典:")
    context_dict = builder.build_context_dict(
        conversations=sample["conversations"],
        temporal_grounding=sample.get("temporal_grounding")
    )
    print(json.dumps(context_dict, indent=2, ensure_ascii=False))
