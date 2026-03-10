"""
VideoQA 评估脚本

在多个视频问答基准（MMVU / MVBench / TempCompass / VideoMME / VideoMMMU / VSIBench）
上评估 Qwen3-VL 系列模型。

支持两种回答模式：
  think  - 先思考后回答，从 <answer> XML 标签中提取最终答案
  direct - 直接回答，优先尝试 <answer> 标签，再 fallback 到首个字母/数字

各数据集的 messages 构造逻辑见:
  src/open_r1_video/eval/datasets.py

使用方法:
    python src/open_r1_video/eval_videoqa.py \\
        --model_path ./Qwen3-VL-2B-Thinking \\
        --dataset_path ./src/open_r1_video/eval/eval_videomme.json \\
        --dataset_type videomme \\
        --video_base_dir ./video_data \\
        --output_dir ./output/eval/videomme \\
        --answer_mode think
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from open_r1_video.eval.datasets import (
    DATASET_BUILDERS,
    check_video_paths,
    compute_accuracy,
    extract_pred_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VideoQA 基准评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 路径 ----
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（本地目录或 HuggingFace Hub ID）")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="评估数据集 JSON 文件路径")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=list(DATASET_BUILDERS.keys()),
                        help=f"数据集类型，可选: {list(DATASET_BUILDERS.keys())}")
    parser.add_argument("--video_base_dir", type=str, default=None,
                        help="视频文件根目录，用于解析数据集中的相对路径（None 表示路径已是绝对路径）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果保存目录")

    # ---- 评估配置 ----
    parser.add_argument("--answer_mode", type=str, choices=["think", "direct"], default="think",
                        help="回答模式：think（先思考后回答）或 direct（直接回答）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数（None 表示评估全部）")
    parser.add_argument("--check_video_paths", action="store_true", default=False,
                        help="加载模型前预检查所有视频路径是否存在，不存在则报错退出")

    # ---- 视频处理配置 ----
    parser.add_argument("--fps", type=float, default=1.0,
                        help="视频采样帧率（帧/秒）")
    parser.add_argument("--max_frames", type=int, default=32,
                        help="每个视频最大采样帧数")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="视频帧最大像素数（None 表示使用处理器默认值）")

    # ---- 模型推理配置 ----
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="最大生成 token 数（think 模式建议适当调大）")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="注意力实现方式")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="模型加载精度")

    return parser.parse_args()


def load_model_and_processor(
    model_path: str,
    attn_implementation: str,
    torch_dtype_str: str,
):
    """加载 Qwen3-VL 模型和处理器"""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[torch_dtype_str]

    print(f"Loading model from {model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded. Devices: {set(str(p.device) for p in model.parameters())}")
    return model, processor


def run_inference(
    model,
    processor,
    messages: list,
    max_new_tokens: int,
    answer_mode: str = "think",
) -> str:
    """对单个样本运行推理，返回模型生成的文本（已去除输入部分）"""
    from qwen_vl_utils import process_vision_info

    # direct 模式关闭模型内置思考能力，避免生成 <think> 块
    enable_thinking = answer_mode != "direct"

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


def main():
    args = parse_args()

    print("=" * 60)
    print("VideoQA 评估配置")
    print("=" * 60)
    print(f"  模型:          {args.model_path}")
    print(f"  数据集:        {args.dataset_path}")
    print(f"  数据集类型:    {args.dataset_type}")
    print(f"  回答模式:      {args.answer_mode}")
    print(f"  视频根目录:    {args.video_base_dir}")
    print(f"  输出目录:      {args.output_dir}")
    print(f"  最大样本数:    {args.max_samples}")
    print(f"  视频 FPS:      {args.fps}")
    print(f"  最大帧数:      {args.max_frames}")
    print(f"  最大像素:      {args.max_pixels}")
    print(f"  最大生成长度:  {args.max_new_tokens}")
    print(f"  注意力实现:    {args.attn_implementation}")
    print(f"  精度:          {args.torch_dtype}")
    print(f"  路径预检查:    {args.check_video_paths}")
    print("=" * 60)

    # 加载数据集（在模型之前加载，以便预检查路径）
    print(f"\nLoading dataset from {args.dataset_path} ...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]
    print(f"Dataset loaded: {len(dataset)} samples")

    # 视频路径预检查（在加载模型之前，避免模型加载完才发现路径错误）
    if args.check_video_paths:
        print("\nChecking video paths ...")
        check_video_paths(dataset, args.video_base_dir)
        print(f"All {len(dataset)} video paths verified.")

    # 加载模型
    model, processor = load_model_and_processor(
        args.model_path, args.attn_implementation, args.torch_dtype
    )

    build_messages = DATASET_BUILDERS[args.dataset_type]

    # 逐样本评估
    results = []
    inference_times = []

    for sample in tqdm(dataset, desc="Evaluating"):
        t0 = time.time()
        problem_type = sample.get("problem_type", "multiple choice")
        # 提前获取 sample_id，方便异常时记录
        sample_id = str(sample.get("problem_id", sample.get("video_id", "")))
        gt_answer = ""

        try:
            messages, gt_answer, sample_id = build_messages(
                sample,
                video_base_dir=args.video_base_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=args.max_pixels,
                answer_mode=args.answer_mode,
            )

            response = run_inference(model, processor, messages, args.max_new_tokens, args.answer_mode)
            pred_answer = extract_pred_answer(response, args.answer_mode)
            correct = compute_accuracy(pred_answer, gt_answer, problem_type)

            elapsed = time.time() - t0
            inference_times.append(elapsed)

            results.append({
                "id": sample_id,
                "problem_type": problem_type,
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "correct": correct,
                "response": response,
                "inference_time": elapsed,
            })

        except Exception as e:
            import traceback
            print(f"\n❌ Error on sample {sample_id}: {e}")
            traceback.print_exc()
            results.append({
                "id": sample_id,
                "problem_type": problem_type,
                "ground_truth": gt_answer,
                "prediction": "",
                "correct": 0.0,
                "response": f"ERROR: {e}",
                "inference_time": time.time() - t0,
            })

    # 统计
    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / total if total > 0 else 0.0
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  总样本数:      {total}")
    print(f"  正确数:        {int(correct_count)}")
    print(f"  准确率:        {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  平均推理时间:  {avg_time:.2f} s/sample")
    print("=" * 60)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": args.model_path,
        "dataset_type": args.dataset_type,
        "dataset_path": args.dataset_path,
        "answer_mode": args.answer_mode,
        "total_samples": total,
        "correct": int(correct_count),
        "accuracy": accuracy,
        "avg_inference_time_sec": avg_time,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到 {output_dir}")
    print(f"    summary.json — 评估摘要")
    print(f"    results.json — 逐样本详细结果")


if __name__ == "__main__":
    main()
