#!/usr/bin/env python3
"""
从 results.json 重新提取答案并计算正确率。

兼容两种模型输出格式：
  1. 答案在 <answer>...</answer> 标签中
  2. 答案直接跟在 </think> 标签之后（无 <answer> 标签）
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def extract_pred_answer(response: str, problem_type: str = "multiple choice") -> str:
    """
    从模型输出中提取预测答案，兼容有无 <answer> 标签两种情况。

    提取优先级：
      1. <answer>...</answer> 标签内容
      2. </think> 之后的文本中提取答案
      3. 整个 response 中最后出现的独立选项字母 / 数字
    """
    # 1) 优先尝试 <answer> 标签
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) 尝试从 </think> 之后的文本中提取
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

    # 3) 最终 fallback：从整段 response 的末尾部分查找
    #    取最后 100 个字符，寻找最后一个独立的选项字母或数字
    tail = response[-100:] if len(response) > 100 else response
    if problem_type == "regression":
        matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", tail)
        if matches:
            return matches[-1]
    else:
        matches = re.findall(r"\b([A-J])\b", tail)
        if matches:
            return matches[-1]

    return ""


def compute_accuracy(pred: str, gt: str, problem_type: str = "multiple choice") -> float:
    """计算单样本准确率"""
    if not pred or not gt:
        return 0.0
    if problem_type == "regression":
        try:
            return 1.0 if abs(float(pred) - float(gt)) <= 0.5 else 0.0
        except ValueError:
            return 0.0
    else:
        return 1.0 if pred.strip().upper() == gt.strip().upper() else 0.0


def main():
    parser = argparse.ArgumentParser(description="从 results.json 重新计算正确率")
    parser.add_argument("input", nargs="?", default="results.json",
                        help="输入的 results.json 文件路径 (默认: results.json)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出修正后的 JSON 文件路径 (默认: results_recalc.json)")
    parser.add_argument("--detail", action="store_true",
                        help="打印每个修正样本的详情")
    args = parser.parse_args()

    output_path = args.output or args.input.replace(".json", "_recalc.json")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    results = data.get("results", [])
    dataset_type = summary.get("dataset_type", "unknown")

    total = len(results)
    old_correct = 0
    new_correct = 0
    fixed_count = 0

    # 按维度/类别统计（如 tempcompass 的 dimension）
    dim_stats = defaultdict(lambda: {"total": 0, "old_correct": 0, "new_correct": 0})

    for item in results:
        response = item.get("response", "")
        gt = item.get("ground_truth", "")
        problem_type = item.get("problem_type", "multiple choice")
        old_pred = item.get("prediction", "")
        old_acc = item.get("correct", 0.0)

        # 重新提取
        new_pred = extract_pred_answer(response, problem_type)
        new_acc = compute_accuracy(new_pred, gt, problem_type)

        old_correct += old_acc
        new_correct += new_acc

        # 维度统计
        dimension = item.get("dimension", "overall")
        dim_stats[dimension]["total"] += 1
        dim_stats[dimension]["old_correct"] += old_acc
        dim_stats[dimension]["new_correct"] += new_acc

        if new_pred != old_pred:
            fixed_count += 1
            if args.detail:
                status = "FIXED" if new_acc > old_acc else ("SAME" if new_acc == old_acc else "WORSE")
                print(f"[{status}] id={item['id']}  gt={gt}  "
                      f"old_pred='{old_pred}'  new_pred='{new_pred}'  "
                      f"old_acc={old_acc}  new_acc={new_acc}")

        # 更新 item
        item["prediction"] = new_pred
        item["correct"] = new_acc
        item["old_prediction"] = old_pred
        item["old_correct"] = old_acc

    old_accuracy = old_correct / total if total > 0 else 0
    new_accuracy = new_correct / total if total > 0 else 0

    # 更新 summary
    summary["old_accuracy"] = old_accuracy
    summary["old_correct"] = int(old_correct)
    summary["accuracy"] = new_accuracy
    summary["correct"] = int(new_correct)
    summary["recalculated"] = True
    summary["fixed_predictions"] = fixed_count

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  数据集: {dataset_type}")
    print(f"  总样本数: {total}")
    print(f"  原始正确数: {int(old_correct)} / {total}  ({old_accuracy:.4%})")
    print(f"  修正正确数: {int(new_correct)} / {total}  ({new_accuracy:.4%})")
    print(f"  修正预测数: {fixed_count}")
    print(f"  准确率提升: {new_accuracy - old_accuracy:+.4%}")
    print(f"{'='*60}")

    # 按维度打印
    if len(dim_stats) > 1:
        print(f"\n按维度/类别统计:")
        print(f"{'维度':<25} {'总数':>6} {'原始正确':>8} {'修正正确':>8} {'原始率':>8} {'修正率':>8}")
        print("-" * 70)
        for dim in sorted(dim_stats.keys()):
            s = dim_stats[dim]
            t = s["total"]
            oa = s["old_correct"] / t if t > 0 else 0
            na = s["new_correct"] / t if t > 0 else 0
            print(f"{dim:<25} {t:>6} {int(s['old_correct']):>8} {int(s['new_correct']):>8} "
                  f"{oa:>7.2%} {na:>7.2%}")

    # 保存修正后的结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n修正后的结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
