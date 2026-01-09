"""
evaluate_model_official_standard.py: 严格遵循Qwen3-VL官方处理流程
"""
import sys
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from PIL import Image
import time

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ==================== 严格导入官方工具 ====================
try:
    from qwen_vl_utils import process_vision_info
    print("✓ Imported official process_vision_info")
except ImportError:
    sys.path.append(str(Path(__file__).parent / "qwen-vl-utils"))
    from qwen_vl_utils import process_vision_info
    print("✓ Imported local process_vision_info")

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

# ==================== 配置参数 ====================
USE_FLASH_ATTENTION = True
USE_TORCH_COMPILE = False
DTYPE = torch.bfloat16

def load_model_with_acceleration(model_path: str, device: str = "auto"):
    """加载模型并应用加速优化"""
    print("Loading Qwen3-VL model with acceleration optimizations...")
    
    # ==================== 修复1：转换路径为绝对路径 ====================
    model_path = str(Path(model_path).resolve())  # 转换为绝对路径
    
    load_config = {
        "torch_dtype": DTYPE,  # 暂时保留，后面兼容处理
        "device_map": device,
        "trust_remote_code": True,
    }
    
    # 兼容新旧版transformers
    try:
        # 新版使用dtype参数
        from packaging import version
        import transformers
        if version.parse(transformers.__version__) >= version.parse("4.45.0"):
            load_config["dtype"] = DTYPE
            del load_config["torch_dtype"]
    except:
        pass
    
    if USE_FLASH_ATTENTION:
        try:
            load_config["attn_implementation"] = "flash_attention_2"
            print("✓ Flash Attention 2 enabled")
        except:
            print("⚠ Flash Attention 2 not available, using default attention")
    
    # ==================== 修复2：确保路径有效 ====================
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # from_pretrained可以接受str或Path对象
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),  # 确保传入字符串路径
        **load_config
    )
    
    if USE_TORCH_COMPILE:
        print("Applying torch.compile (this may take a few minutes on first run)...")
        model = torch.compile(model, mode="reduce-overhead")
    
    processor = AutoProcessor.from_pretrained(
        str(model_path),  # 同样使用规范路径
        trust_remote_code=True
    )
    
    return model, processor


def evaluate_model_on_dataset(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    device: str = "auto",
    max_frames: int = 16,
    batch_size: int = 1
) -> pd.DataFrame:
    """
    在数据集上评估模型并计算奖励（严格官方流程）
    """
    model, processor = load_model_with_acceleration(model_path, device)
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    inference_times = []
    
    print(f"Evaluating {len(dataset)} samples...")
    
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        start_time = time.time()
        
        try:
            # ==================== 核心：使用标准官方流程 ====================
            result = process_single_sample_official(
                sample, model, processor, max_frames
            )
            if result:
                results.append(result)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if (idx + 1) % 10 == 0:
                    avg_time = np.mean(inference_times[-10:])
                    print(f"  Average inference time (last 10): {avg_time:.2f}s")
                    
        except Exception as e:
            print(f"❌ Error processing sample {idx} (ID: {sample.get('problem_id')}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_json(output_path / "evaluation_results.json", orient="records", indent=2)
    results_df.to_csv(output_path / "evaluation_results.csv", index=False)
    
    if inference_times:
        time_stats = {
            "mean_inference_time": float(np.mean(inference_times)),
            "std_inference_time": float(np.std(inference_times)),
            "total_inference_time": float(np.sum(inference_times)),
            "samples_per_second": len(inference_times) / np.sum(inference_times)
        }
        with open(output_path / "inference_stats.json", 'w') as f:
            json.dump(time_stats, f, indent=2)
        
        print(f"\nInference speed: {time_stats['samples_per_second']:.2f} samples/sec")
    
    print(f"✓ Evaluation completed. Results saved to {output_path}")
    return results_df


def process_single_sample_official(
    sample: Dict, 
    model: Any, 
    processor: Any, 
    max_frames: int
) -> Optional[Dict]:
    """严格遵循官方示例的处理流程"""
    video_path = sample["path"]
    
    # ==================== 步骤1: 构造标准messages ====================
    options_text = "\n".join(sample["options"])
    prompt = VIDEO_GROUNDED_QA_PROMPT.format(
        question=sample["problem"],
        options=options_text,
        duration=sample["duration"]
    )
    
    # 官方格式的messages（关键改动）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,  # 直接传路径，由官方工具处理
                    # 可以在这里指定视频处理参数（如果process_vision_info支持）
                    # "max_frames": max_frames,
                    "fps": 0.5,
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # ==================== 步骤2: 应用chat template ====================
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # ==================== 步骤3: 处理视觉信息（核心官方函数） ====================
    try:
        # process_vision_info会自动处理视频解码、采样、时间戳
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            # max_frames=max_frames,  # 假设process_vision_info支持此参数
            return_video_kwargs=True,
            return_video_metadata=True
        )
    except Exception as e:
        print(f"⚠️  process_vision_info failed for {video_path}: {e}")
        return None
    print('=============videos============= ', '\n', videos)
    print('=============video_kwargs=============', '\n', video_kwargs)
    # ==================== 步骤4: 处理视频元数据 ====================
    video_metadata = None
    if videos is not None:
        # 解压videos和metadatas（与官方示例完全一致）
        videos, video_metadatas = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadatas)
        
        print(f"✓ Video processed: {len(videos)} frames with metadata")
        print(f"  Video kwargs: {video_kwargs}")
    
    # ==================== 步骤5: 构造processor输入 ====================
    # 严格遵循官方示例的参数
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,  # 官方示例明确设置
        # 其他参数由video_kwargs自动传递
        **video_kwargs
    )
    print("inputs keys=", list(inputs.keys()))# inputs keys= ['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw']
    print("inputs['input_ids'] shape= ", inputs['input_ids'].shape)
    print("inputs['pixel_values_videos'] shape= ", inputs['pixel_values_videos'].shape)
    print("inputs['attention_mask'] shape= ", inputs['attention_mask'].shape)
    # print("inputs['video_grid_thw']= ", inputs['video_grid_thw'])
    print("inputs['video_grid_thw']shape= ", inputs['video_grid_thw'].shape)
    # print('=============inputs============= ', '\n', inputs)
    # inputs = inputs.to(model.device)
    
    # ==================== 步骤6: 模型推理 ====================
    # with torch.no_grad():
    #     generation_config = {
    #         "max_new_tokens": 512,
    #         "do_sample": False,
    #         "temperature": 0.0,
    #         "use_cache": True,
    #     }
        
    #     generated_ids = model.generate(**inputs, **generation_config)
    
    # # ==================== 步骤7: 解码输出 ====================
    # generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    # response = processor.batch_decode(
    #     generated_ids_trimmed,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )[0]
    
    # # ==================== 步骤8: 计算奖励 ====================
    # accuracy_reward = calculate_accuracy(response, sample["solution"])
    # pred_time_ranges = extract_multiple_time_ranges(response)
    # temporal_iou = calculate_temporal_iou_multi(pred_time_ranges, sample["relevant_windows"])
    # format_reward = calculate_format_reward(response)
    # total_reward = 0.4 * accuracy_reward + 0.4 * temporal_iou + 0.2 * format_reward
    
    # # 提取实际使用的时间戳（从video_metadata）
    # actual_timestamps = []
    # if video_metadata and len(video_metadata) > 0:
    #     # video_metadata中通常包含frame_seconds信息
    #     actual_timestamps = video_metadata[0].get("frame_seconds", [])
    
    # return {
    #     "problem_id": sample["problem_id"],
    #     "question": sample["problem"],
    #     "ground_truth_solution": sample["solution"],
    #     "ground_truth_windows": sample["relevant_windows"],
    #     "model_response": response,
    #     "predicted_time_ranges": pred_time_ranges,
    #     "video_timestamps_used": actual_timestamps,
    #     "video_frames_count": len(videos) if videos else 0,
    #     "video_kwargs_used": video_kwargs,  # 记录使用的参数
    #     "accuracy_reward": accuracy_reward,
    #     "temporal_iou_reward": temporal_iou,
    #     "format_reward": format_reward,
    #     "total_reward": total_reward,
    #     "data_source": sample.get("data_source", ""),
    #     "video_path": video_path
    # }




def verify_local_utils():
    """验证本地qwen-vl-utils是否可用"""
    print("🔍 Verifying local qwen-vl-utils installation...")
    
    try:
        from qwen_vl_utils import process_vision_info
        print("✓ qwen_vl_utils.process_vision_info imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def main():
    # ==================== 配置参数 ====================
    MODEL_PATH = "Qwen3-VL-2B-Thinking"   # 本地模型路径
    DATASET_PATH = "./dataset/video-mtr/qv-nextgqa_merge_8k.json"    # 数据集路径
    OUTPUT_DIR = "./evaluation_results_offical_standard"
    MAX_SAMPLES = 1  # 评估样本数
    MAX_FRAMES = 64   # 每视频最大采样帧数
    
    # 验证本地工具
    if not verify_local_utils():
        print("\n⚠️  Please ensure qwen-vl-utils is installed locally:")
        print("   git clone https://github.com/QwenLM/qwen-vl-utils.git")
        print("   cd qwen-vl-utils && pip install -e .")
        return
    
    # 运行评估
    results_df = evaluate_model_on_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_samples=MAX_SAMPLES,
        max_frames=MAX_FRAMES,
        batch_size=1
    )
    
    # 绘制分布图
    # plot_reward_distributions(results_df, OUTPUT_DIR)
    
    # print("✅ All done! Strictly following official Qwen3-VL processing pipeline.")


if __name__ == "__main__":
    main()