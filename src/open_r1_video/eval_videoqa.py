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

    # ---- vLLM 推理配置 ----
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="使用 vLLM 进行推理（离线批量推理，速度更快）")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM 张量并行 GPU 数（默认 1）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80,
                        help="vLLM GPU 显存利用率（0-1）")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM 最大序列长度（None 表示使用模型默认值）")
    parser.add_argument("--vllm_batch_size", type=int, default=16,
                        help="vLLM 批量推理时每批样本数")
    parser.add_argument("--vllm_prefetch", action="store_true", default=False,
                        help="启用后台线程预取：在 GPU 推理当前 batch 时，后台线程预处理下一个 batch 的视频")

    return parser.parse_args()


def load_model_and_processor(
    model_path: str,
    attn_implementation: str,
    torch_dtype_str: str,
):
    """加载 Qwen3-VL 模型和处理器（HuggingFace Transformers）"""
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

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    print(f"Model loaded. Devices: {set(str(p.device) for p in model.parameters())}")
    return model, processor


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
):
    """加载 Qwen3-VL 模型（vLLM 离线推理）"""
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoProcessor
    from vllm import LLM

    print(f"Loading vLLM model from {model_path} ...")
    llm_kwargs = {
        "model": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "limit_mm_per_prompt": {"video": 1},
        "mm_processor_cache_gb": 0,
        "enable_prefix_caching": False,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    print(f"vLLM model loaded. tensor_parallel_size={tensor_parallel_size}")
    return llm, processor


def _append_skip_thinking(text: str) -> str:
    """在 apply_chat_template 生成的 prompt 末尾追加已完成的思考块，
    诱导模型跳过思考阶段直接输出答案。

    apply_chat_template(enable_thinking=True, add_generation_prompt=True)
    生成的 prompt 以 ``<|im_start|>assistant\\n<think>\\n`` 结尾，
    本函数在其后追加一段简短的"思考完毕"文本并关闭 </think> 标签，
    使模型认为思考已经结束，从而直接生成最终答案。
    """
    return text + "Okay, this is straightforward. Here is the final answer.\n</think>\n\n"


def run_inference(
    model,
    processor,
    messages: list,
    max_new_tokens: int,
    answer_mode: str = "think",
) -> str:
    """对单个样本运行推理，返回模型生成的文本（已去除输入部分）"""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if answer_mode == "direct":
        text = _append_skip_thinking(text)

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


def prepare_vllm_input(
    processor,
    messages: list,
    answer_mode: str = "think",
) -> dict:
    """将单个样本的 messages 转换为 vLLM 离线推理输入格式"""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if answer_mode == "direct":
        text = _append_skip_thinking(text)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=processor.image_processor.patch_size,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def run_vllm_batch_inference(
    llm,
    vllm_inputs: list,
    max_new_tokens: int,
) -> list:
    """使用 vLLM 对一批样本进行离线推理，返回生成文本列表"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


def _preprocess_chunk(
    chunk_samples: list,
    build_messages_fn,
    processor,
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
) -> tuple:
    """预处理单个 chunk：视频解码 + 构建 vLLM 输入。

    Returns:
        (vllm_inputs, chunk_meta, error_results) 三元组
    """
    chunk_inputs = []
    chunk_meta = []   # list of (sample_id, gt_answer, problem_type)
    error_results = []

    for sample in chunk_samples:
        problem_type = sample.get("problem_type", "multiple choice")
        sample_id = str(sample.get("problem_id", sample.get("video_id", "")))
        gt_answer = ""

        try:
            messages, gt_answer, sample_id = build_messages_fn(
                sample,
                video_base_dir=video_base_dir,
                fps=fps,
                max_frames=max_frames,
                max_pixels=max_pixels,
                answer_mode=answer_mode,
            )
            vllm_input = prepare_vllm_input(processor, messages, answer_mode)
            chunk_inputs.append(vllm_input)
            chunk_meta.append((sample_id, gt_answer, problem_type))
        except Exception as e:
            import traceback
            print(f"\n Error preparing sample {sample_id}: {e}")
            traceback.print_exc()
            error_results.append({
                "id": sample_id,
                "problem_type": problem_type,
                "ground_truth": gt_answer,
                "prediction": "",
                "correct": 0.0,
                "response": f"ERROR: {e}",
                "inference_time": 0.0,
            })

    return chunk_inputs, chunk_meta, error_results


def _prefetch_chunks(
    dataset: list,
    batch_size: int,
    build_messages_fn,
    processor,
    video_base_dir: Optional[str],
    fps: float,
    max_frames: int,
    max_pixels: Optional[int],
    answer_mode: str,
    prefetch_count: int = 1,
):
    """后台线程预取生成器：预处理 chunk N+1 的视频数据，与 chunk N 的 GPU 推理并行。

    类似训练脚本中的 BatchPrefetcher，使用生产者-消费者模式：

        [后台线程] 视频解码+预处理 ──► Queue ──► [主线程] vLLM GPU 推理
              chunk N+1                              chunk N

    Args:
        prefetch_count: 预取队列容量。1 = 提前准备 1 个 chunk，
                        峰值内存为 2 个 chunk 的视频数据。

    Yields:
        (chunk_inputs, chunk_meta, error_results) 三元组
    """
    import queue as _queue
    import threading

    total = len(dataset)
    q = _queue.Queue(maxsize=prefetch_count)
    stop_event = threading.Event()
    _DONE = object()

    def producer():
        try:
            for start in range(0, total, batch_size):
                if stop_event.is_set():
                    return
                chunk_samples = dataset[start : start + batch_size]
                try:
                    result = _preprocess_chunk(
                        chunk_samples, build_messages_fn, processor,
                        video_base_dir, fps, max_frames, max_pixels, answer_mode,
                    )
                    q.put(result)
                except Exception as e:
                    q.put(e)
                    return
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                stop_event.set()
                raise item
            yield item
    finally:
        stop_event.set()
        while not q.empty():
            try:
                q.get_nowait()
            except _queue.Empty:
                break
        thread.join(timeout=10)


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
    if args.use_vllm:
        print(f"  推理后端:      vLLM")
        print(f"  张量并行:      {args.tensor_parallel_size}")
        print(f"  显存利用率:    {args.gpu_memory_utilization}")
        print(f"  最大序列长度:  {args.max_model_len}")
        print(f"  批量大小:      {args.vllm_batch_size}")
    else:
        print(f"  推理后端:      HuggingFace Transformers")
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
        check_video_paths(dataset, args.video_base_dir, output_dir=args.output_dir)
        print(f"All {len(dataset)} video paths verified.")

    # 加载模型
    if args.use_vllm:
        llm, processor = load_vllm_model(
            args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        model, processor = load_model_and_processor(
            args.model_path, args.attn_implementation, args.torch_dtype
        )

    build_messages = DATASET_BUILDERS[args.dataset_type]

    # 逐样本评估
    results = []
    inference_times = []

    if args.use_vllm:
        # ---- vLLM 分块推理路径 ----
        import gc

        batch_size = args.vllm_batch_size
        total_samples = len(dataset)
        num_chunks = (total_samples + batch_size - 1) // batch_size

        if args.vllm_prefetch:
            mode_desc = "pipelined (prefetch=1)"
        else:
            mode_desc = "chunked (sequential)"
        print(f"\nRunning vLLM {mode_desc} inference: {total_samples} samples, "
              f"batch_size={batch_size}, {num_chunks} chunks")

        t_total_start = time.time()
        total_inferred = 0

        if args.vllm_prefetch:
            # 后台线程预处理 chunk N+1，同时主线程对 chunk N 进行 GPU 推理
            chunk_iter = _prefetch_chunks(
                dataset, batch_size, build_messages, processor,
                args.video_base_dir, args.fps, args.max_frames,
                args.max_pixels, args.answer_mode,
            )
        else:
            # 串行：预处理完当前 chunk 后再推理
            chunk_iter = (
                _preprocess_chunk(
                    dataset[start : start + batch_size],
                    build_messages, processor,
                    args.video_base_dir, args.fps, args.max_frames,
                    args.max_pixels, args.answer_mode,
                )
                for start in range(0, total_samples, batch_size)
            )

        for chunk_inputs, chunk_meta, error_results in tqdm(
            chunk_iter, desc="vLLM inference", total=num_chunks
        ):
            # 收集预处理阶段的错误
            results.extend(error_results)

            if chunk_inputs:
                chunk_responses = run_vllm_batch_inference(llm, chunk_inputs, args.max_new_tokens)
                total_inferred += len(chunk_responses)

                for j, response in enumerate(chunk_responses):
                    sample_id, gt_answer, problem_type = chunk_meta[j]
                    pred_answer = extract_pred_answer(response, args.answer_mode, problem_type)
                    correct = compute_accuracy(pred_answer, gt_answer, problem_type)
                    results.append({
                        "id": sample_id,
                        "problem_type": problem_type,
                        "ground_truth": gt_answer,
                        "prediction": pred_answer,
                        "correct": correct,
                        "response": response,
                    })

            # 释放当前 chunk 的视频数据，防止内存堆积
            del chunk_inputs, chunk_meta, error_results
            gc.collect()

        t_total_elapsed = time.time() - t_total_start
        avg_per_sample = t_total_elapsed / total_inferred if total_inferred else 0.0

        # 回填推理时间
        for r in results:
            if "inference_time" not in r:
                r["inference_time"] = avg_per_sample
        inference_times = [avg_per_sample] * total_inferred

    else:
        # ---- HuggingFace Transformers 逐样本推理路径 ----
        for sample in tqdm(dataset, desc="Evaluating"):
            t0 = time.time()
            problem_type = sample.get("problem_type", "multiple choice")
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
                pred_answer = extract_pred_answer(response, args.answer_mode, problem_type)
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
    no_answer_count = sum(1 for r in results if not r["prediction"])

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  总样本数:      {total}")
    print(f"  正确数:        {int(correct_count)}")
    print(f"  未提取到答案:  {no_answer_count}")
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
        "inference_backend": "vllm" if args.use_vllm else "transformers",
        "total_samples": total,
        "correct": int(correct_count),
        "no_answer": no_answer_count,
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
