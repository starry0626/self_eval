import torch
import torch_npu  # 【修改1】必须导入，注册 NPU 设备
import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
# Qwen3-VL 架构与 Qwen2.5-VL 兼容，Transformers 库通常使用此作为加载类
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from rouge_score import rouge_scorer
from qwen_vl_utils import process_vision_info

# ================= 配置项 =================
MODEL_ID = "models/qwen3-vl-2b-thinking"
DATASET_ID = "ReXTime/data/rextime_train_filtered.jsonl"
VIDEO_ROOT_PATH = "dataset/rextime/v1-2/train"  # 【请修改】视频路径
OUTPUT_FILE = "scored_dataset.jsonl"
PLOT_FILE = "reward_distribution.png"

# NPU 上建议显式控制 Batch Size 为 1，防止动态 Shape 导致的内存波动
BATCH_SIZE = 1 

def find_video_path(vid, video_root_path):
    """智能查找视频文件，支持多种格式"""
    if vid.lower().endswith(('.mp4', '.mkv', '.avi')):
        video_path = os.path.join(video_root_path, vid)
        if os.path.exists(video_path):
            return video_path
    
    for ext in ['.mp4', '.mkv', '.avi']:
        video_filename = vid + ext
        video_path = os.path.join(video_root_path, video_filename)
        if os.path.exists(video_path):
            return video_path
    
    return None

# ================= 奖励函数工具 (保持不变) =================
def calculate_iou(pred_span, gt_span):
    if not pred_span or not gt_span:
        return 0.0
    # 假设 gt_span 和 pred_span 都是 [start, end]
    # 如果数据集中的 span 是 [[s1,e1], [s2,e2]] 这种多段格式，需根据实际情况修改
    # 这里按单段处理：
    g_start, g_end = gt_span[0], gt_span[1]
    p_start, p_end = pred_span[0], pred_span[1]
    
    intersection_start = max(g_start, p_start)
    intersection_end = min(g_end, p_end)
    
    intersection = max(0, intersection_end - intersection_start)
    union = (g_end - g_start) + (p_end - p_start) - intersection
    
    if union <= 0: return 0.0
    return intersection / union

def calculate_rouge(pred_text, gt_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt_text, pred_text)
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0

def parse_output(output_text):
    # 提取时间戳 <timestamp>s-e</timestamp>
    time_pattern = r"<timestamp>\s*([\d\.]+)\s*-\s*([\d\.]+)\s*</timestamp>"
    time_match = re.search(time_pattern, output_text)
    pred_span = None
    if time_match:
        try:
            pred_span = [float(time_match.group(1)), float(time_match.group(2))]
        except:
            pred_span = None
            
    # 提取答案 <answer>...</answer>
    ans_pattern = r"<answer>(.*?)</answer>"
    ans_match = re.search(ans_pattern, output_text, re.DOTALL)
    pred_ans = ans_match.group(1).strip() if ans_match else ""
    
    return pred_span, pred_ans

# ================= 主流程 =================

def main():
    # 【修改2】显式定义 NPU 设备
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 【修改3】精度选择
    # Ascend 910B (Pro) 支持 bfloat16。
    # 如果你是 Ascend 910A，请将下面改为 torch.float16
    dtype = torch.bfloat16 

    print(f"Loading model: {MODEL_ID}")
    
    # 【修改4】加载模型时的 NPU 适配配置
    # 1. 不使用 device_map="auto"
    # 2. 使用 "sdpa" (Scaled Dot Product Attention) 获得最佳兼容性和速度
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="sdpa", 
        device_map=None, 
    )
    model.to(device) # 手动移动到 NPU
    model.eval() # 设置为评估模式

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading dataset from JSONL: {DATASET_ID}")
    ds = load_dataset("json", data_files=DATASET_ID, split="train")
    ds = ds.select(range(10)) # 调试时取消注释

    results = []
    
    print("Starting inference...")
    for item in tqdm(ds):
        vid = item['vid'] 
            
        video_path = find_video_path(vid, VIDEO_ROOT_PATH)
        
        if not video_path:
            missing_count += 1
            if missing_count <= 5:
                print(f"⚠️  视频不存在，跳过: {vid}")
            continue

        question = item['question']
        gt_answer = item['answer']
        gt_span = item['span'] 

        # 构造 Prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        # 如果显存紧张，可以限制最大帧数
                        # "max_pixels": 360 * 420, 
                        # "fps": 1.0, 
                    },
                    {"type": "text", "text": f"你是一个智能视频助手。请先思考，然后给出该问题相关的时间片段和答案。\n输出格式要求：\n<think>你的思考过程</think>\n<timestamp>开始秒数-结束秒数</timestamp>\n<answer>答案文本</answer>\n\nQuestion: {question}"},
                ],
            }
        ]

        # 预处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 【修改5】将 inputs 字典中的所有 tensor 移动到 NPU
        inputs = inputs.to(device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                use_cache=True # NPU 上通常开启 KV cache 没问题
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 解析与计算
        pred_span, pred_ans = parse_output(output_text)
        
        reward_format = 0.0
        if pred_span is not None: reward_format += 0.5
        if pred_ans != "": reward_format += 0.5
        
        reward_iou = calculate_iou(pred_span, gt_span)
        reward_accuracy = calculate_rouge(pred_ans, gt_answer)
        total_reward = (reward_format + reward_iou + reward_accuracy) / 3.0

        record = {
            "qid": item['qid'],
            "question": question,
            "gt_answer": gt_answer,
            "gt_span": gt_span,
            "pred_raw": output_text,
            "pred_span": pred_span,
            "pred_ans": pred_ans,
            "reward_format": reward_format,
            "reward_iou": reward_iou,
            "reward_accuracy": reward_accuracy,
            "total_reward": total_reward
        }
        results.append(record)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        # 【修改6】手动清理显存 (可选，视显存压力而定)
        # torch.npu.empty_cache() 

    # 绘图逻辑保持不变，因为这是在 CPU 上运行的 Matplotlib
    print(f"Plotting distribution to {PLOT_FILE}...")
    # ... (绘图代码与之前相同，省略以节省空间) ...
    # 建议直接把上一段回答里的绘图代码 copy 过来放在这里
    
    rewards_fmt = [r['reward_format'] for r in results]
    rewards_iou = [r['reward_iou'] for r in results]
    rewards_acc = [r['reward_accuracy'] for r in results]
    rewards_total = [r['total_reward'] for r in results]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].hist(rewards_fmt, bins=10, color='skyblue', edgecolor='black'); axs[0, 0].set_title('Format Reward')
    axs[0, 1].hist(rewards_iou, bins=20, color='lightgreen', edgecolor='black'); axs[0, 1].set_title('IoU Reward')
    axs[1, 0].hist(rewards_acc, bins=20, color='salmon', edgecolor='black'); axs[1, 0].set_title('Accuracy Reward')
    axs[1, 1].hist(rewards_total, bins=20, color='gold', edgecolor='black'); axs[1, 1].set_title('Total Reward')
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print("Done!")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    main()