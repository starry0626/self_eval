#!/usr/bin/env python3
"""
随机采样过滤测试数据集，生成缩小后的 JSON 文件。

用法:
  python filter_dataset.py eval_tempcompass.json -n 500          # 保留 500 条
  python filter_dataset.py eval_tempcompass.json -r 0.1          # 保留 10%
  python filter_dataset.py eval_tempcompass.json -n 500 -s 42    # 指定随机种子
  python filter_dataset.py eval_tempcompass.json -n 500 -o out.json  # 指定输出路径
"""

import argparse
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="随机采样过滤测试数据集")
    parser.add_argument("input", help="输入数据集 JSON 文件路径")
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="保留的样本数量")
    parser.add_argument("-r", "--ratio", type=float, default=None,
                        help="保留的比例 (0~1), 如 0.1 表示保留 10%%")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件路径 (默认: 原文件名_filtered.json)")
    args = parser.parse_args()

    if args.num is None and args.ratio is None:
        parser.error("请指定 -n (数量) 或 -r (比例) 中的至少一个")
    if args.num is not None and args.ratio is not None:
        parser.error("-n 和 -r 不能同时使用")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    if args.ratio is not None:
        if not 0 < args.ratio <= 1:
            parser.error("-r 比例必须在 (0, 1] 之间")
        keep = max(1, int(total * args.ratio))
    else:
        keep = args.num
        if keep > total:
            print(f"警告: 请求 {keep} 条，但数据集仅有 {total} 条，将保留全部数据")
            keep = total

    random.seed(args.seed)
    sampled = random.sample(data, keep)

    # 按原始顺序排列（如果有 problem_id 字段则按其排序，否则保持采样顺序）
    if sampled and "problem_id" in sampled[0]:
        sampled.sort(key=lambda x: x["problem_id"])

    input_path = Path(args.input)
    output_path = args.output or str(input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=4)

    print(f"原始样本数: {total}")
    print(f"保留样本数: {keep}")
    print(f"随机种子:   {args.seed}")
    print(f"输出文件:   {output_path}")


if __name__ == "__main__":
    main()
