"""
分析 results.json 中 prediction 为空（未提取到答案）的样本

功能：
  1. 统计 prediction 为空的样本数量
  2. 从原始数据集中查找对应的视频路径、问题、选项，补充到记录中
  3. 将这些样本单独写入 no_answer_samples.json
  4. (可选) --export_dataset: 从原始数据集中提取对应条目，
     生成可直接用于重新测试的数据集 JSON 文件

使用方法：
    python scripts/analyze_no_answer.py results.json

    # 指定原始数据集路径（默认从 results.json 的 summary.dataset_path 读取）
    python scripts/analyze_no_answer.py results.json --dataset_path path/to/eval_xxx.json

    # 指定输出文件路径（默认为 results.json 同目录下的 no_answer_samples.json）
    python scripts/analyze_no_answer.py results.json -o output.json

    # 导出可重新测试的数据集子集
    python scripts/analyze_no_answer.py results.json --dataset_path path/to/eval_xxx.json --export_dataset
    python scripts/analyze_no_answer.py results.json --dataset_path path/to/eval_xxx.json --export_dataset --export_path retest.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="分析 results.json 中未提取到答案的样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results_json", type=str,
                        help="待分析的 results.json 文件路径")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="原始数据集 JSON 路径（默认从 summary.dataset_path 读取）")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出文件路径（默认为同目录下 no_answer_samples.json）")
    parser.add_argument("--export_dataset", action="store_true", default=False,
                        help="从原始数据集中提取无答案样本，生成可重新测试的数据集 JSON")
    parser.add_argument("--export_path", type=str, default=None,
                        help="导出数据集的路径（默认为同目录下 eval_<type>_no_answer.json）")
    args = parser.parse_args()

    # 读取 results.json
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    results = data.get("results", [])
    total = len(results)

    if total == 0:
        print("results.json 中没有样本记录")
        sys.exit(0)

    # 确定原始数据集路径
    dataset_path_str = args.dataset_path or summary.get("dataset_path", "")
    dataset_path = Path(dataset_path_str)

    # 如果是相对路径，尝试相对于 results.json 所在目录解析
    if not dataset_path.is_absolute() and not dataset_path.exists():
        candidate = results_path.parent / dataset_path
        if candidate.exists():
            dataset_path = candidate

    # 构建 id → 数据集样本信息 的映射
    id_to_sample = {}
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for sample in dataset:
            sid = str(sample.get("problem_id", sample.get("video_id", "")))
            id_to_sample[sid] = sample
        print(f"已加载数据集: {dataset_path} ({len(dataset)} 条)")
    else:
        print(f"Warning: 数据集文件未找到 ({dataset_path_str})，无法补充视频路径/问题/选项")

    # 筛选 prediction 为空的样本
    no_answer = []
    for r in results:
        if not r.get("prediction"):
            record = dict(r)
            sid = str(r.get("id", ""))
            ds_sample = id_to_sample.get(sid, {})
            # 补充视频路径
            if "video_path" not in record or not record["video_path"]:
                record["video_path"] = ds_sample.get("path", "")
            # 补充问题和选项
            if "question" not in record:
                record["question"] = ds_sample.get("problem", ds_sample.get("question", ""))
            if "options" not in record:
                record["options"] = ds_sample.get("options", [])
            no_answer.append(record)

    no_answer_count = len(no_answer)

    # 打印统计
    print(f"\n{'=' * 50}")
    print(f"  结果文件:       {results_path}")
    print(f"  模型:           {summary.get('model_path', 'N/A')}")
    print(f"  数据集类型:     {summary.get('dataset_type', 'N/A')}")
    print(f"  回答模式:       {summary.get('answer_mode', 'N/A')}")
    print(f"  总样本数:       {total}")
    print(f"  正确数:         {summary.get('correct', 'N/A')}")
    print(f"  准确率:         {summary.get('accuracy', 'N/A')}")
    print(f"  未提取到答案:   {no_answer_count} ({no_answer_count / total * 100:.1f}%)")
    print(f"{'=' * 50}")

    if no_answer_count == 0:
        print("\n所有样本均成功提取到答案，无需生成输出文件。")
        return

    # 写入输出文件
    output_path = Path(args.output) if args.output else results_path.parent / "no_answer_samples.json"
    output_data = {
        "summary": {
            "source_file": str(results_path),
            "model_path": summary.get("model_path", ""),
            "dataset_type": summary.get("dataset_type", ""),
            "answer_mode": summary.get("answer_mode", ""),
            "total_samples": total,
            "no_answer_count": no_answer_count,
            "no_answer_ratio": no_answer_count / total,
        },
        "samples": no_answer,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n已将 {no_answer_count} 条未提取到答案的样本写入: {output_path}")

    # 打印前几个样本的摘要
    print(f"\n前 {min(5, no_answer_count)} 条样本预览:")
    for i, r in enumerate(no_answer[:5]):
        video = r.get("video_path", "N/A")
        response_preview = r.get("response", "")[:80] + "..." if len(r.get("response", "")) > 80 else r.get("response", "")
        print(f"  [{i+1}] id={r.get('id', 'N/A')}, gt={r.get('ground_truth', 'N/A')}, video={video}")
        print(f"       response: {response_preview}")

    # 导出可重新测试的数据集子集
    if args.export_dataset:
        if not id_to_sample:
            print("\nError: 需要原始数据集才能导出，请通过 --dataset_path 指定")
            sys.exit(1)

        no_answer_ids = {str(r.get("id", "")) for r in no_answer}
        export_samples = [
            id_to_sample[sid] for sid in no_answer_ids if sid in id_to_sample
        ]

        if not export_samples:
            print("\nWarning: 未能从数据集中匹配到任何无答案样本，跳过导出")
        else:
            dataset_type = summary.get("dataset_type", "unknown")
            export_path = (
                Path(args.export_path) if args.export_path
                else results_path.parent / f"eval_{dataset_type}_no_answer.json"
            )
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_samples, f, indent=2, ensure_ascii=False)
            print(f"\n已导出 {len(export_samples)} 条数据集记录用于重新测试: {export_path}")


if __name__ == "__main__":
    main()
