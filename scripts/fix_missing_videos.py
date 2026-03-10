#!/usr/bin/env python3
"""
修复缺失视频路径工具

基于 missing_videos.txt 诊断每个缺失路径的原因并自动修复:
  1. 格式不匹配（同名文件存在但扩展名不同）
     - 解码器支持该格式 → 修改数据集中的视频路径
     - 解码器不支持     → 用 ffmpeg 转换为原路径期望的格式
  2. 文件完全缺失 → 从数据集中删除对应数据项

修复结果完整写入 output_dir/fix_report.txt。

用法:
    python scripts/fix_missing_videos.py \
        --missing_file  ./output/eval/videomme/Qwen3-VL-2B-Thinking/missing_videos.txt \
        --dataset_path  ./src/open_r1_video/eval/eval_videomme.json \
        --video_base_dir ./video_data \
        --output_dir    ./output/eval/videomme/Qwen3-VL-2B-Thinking
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 所有后端（torchcodec / decord / torchvision）底层均基于 FFmpeg，
# 以下为 FFmpeg 常见支持的视频容器格式
# ---------------------------------------------------------------------------
FFMPEG_SUPPORTED_EXTENSIONS: Set[str] = {
    ".mp4", ".avi", ".mkv", ".webm", ".mov", ".flv", ".wmv",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob",
}


# ======================== 工具函数 ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 missing_videos.txt 诊断并修复数据集中的缺失视频路径",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--missing_file", type=str, required=True,
                        help="missing_videos.txt 文件路径（由 eval_videoqa 生成）")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="原始数据集 JSON 文件路径")
    parser.add_argument("--video_base_dir", type=str, default=None,
                        help="视频文件根目录，用于将数据集中的相对路径解析为绝对路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录（保存修复后数据集与报告）")
    return parser.parse_args()


def detect_backend() -> str:
    """检测当前环境使用的视频解码后端（与 qwen_vl_utils 逻辑一致）"""
    import importlib.util

    force = os.getenv("FORCE_QWENVL_VIDEO_READER")
    if force:
        return force
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    if importlib.util.find_spec("decord"):
        return "decord"
    return "torchvision"


def get_supported_extensions(backend: str) -> Set[str]:
    """返回指定后端支持的视频扩展名集合

    三个后端底层均为 FFmpeg，支持的容器格式相同。
    """
    return FFMPEG_SUPPORTED_EXTENSIONS


def resolve_video_path(raw_path: str, video_base_dir: Optional[str]) -> str:
    """与 datasets.py 中 _resolve_video_path 逻辑一致"""
    if video_base_dir and not os.path.isabs(raw_path):
        return os.path.normpath(os.path.join(video_base_dir, raw_path))
    return raw_path


def find_alternative_files(resolved_path: str) -> List[Path]:
    """在同目录下查找同名但不同扩展名的文件"""
    p = Path(resolved_path)
    if not p.parent.exists():
        return []
    return sorted(
        f for f in p.parent.iterdir()
        if f.is_file() and f.stem == p.stem and f.suffix.lower() != p.suffix.lower()
    )


def convert_video(src: str, dst: str) -> None:
    """使用 ffmpeg 将视频转换为目标格式"""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:500]}")


# ======================== 主流程 ========================

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. 读取缺失路径 ----
    with open(args.missing_file, "r", encoding="utf-8") as f:
        missing_set: Set[str] = {line.strip() for line in f if line.strip()}
    print(f"读取到 {len(missing_set)} 个缺失视频路径")

    # ---- 2. 加载数据集 ----
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)
    print(f"原始数据集: {len(dataset)} 条样本")

    # ---- 3. 检测后端 & 支持格式 ----
    backend = detect_backend()
    supported_exts = get_supported_extensions(backend)
    print(f"视频解码后端: {backend}")

    # ---- 4. 对每个唯一缺失路径确定修复策略 ----
    # action 类型:
    #   ("format_fix", new_suffix, alt_resolved_path)
    #   ("convert",    src_path,  dst_resolved_path)
    #   ("remove",)
    actions: Dict[str, tuple] = {}

    for sample in dataset:
        resolved = resolve_video_path(sample["path"], args.video_base_dir)
        if resolved not in missing_set or resolved in actions:
            continue

        alternatives = find_alternative_files(resolved)
        if alternatives:
            # 优先选择解码器支持的格式
            supported_alts = [a for a in alternatives if a.suffix.lower() in supported_exts]
            if supported_alts:
                alt = supported_alts[0]
                actions[resolved] = ("format_fix", alt.suffix, str(alt))
            else:
                # 解码器不支持，需要转换为原路径期望的格式
                alt = alternatives[0]
                actions[resolved] = ("convert", str(alt), resolved)
        else:
            actions[resolved] = ("remove",)

    # ---- 5. 执行视频转换 ----
    for resolved, action in list(actions.items()):
        if action[0] != "convert":
            continue
        src, dst = action[1], action[2]
        try:
            convert_video(src, dst)
            print(f"  [转换成功] {src} -> {dst}")
        except Exception as e:
            print(f"  [转换失败] {src}: {e}，该视频将被删除")
            actions[resolved] = ("remove",)

    # ---- 6. 应用修复策略，构建新数据集 ----
    fixed_dataset: List[Dict[str, Any]] = []
    # 逐条记录，用于写入报告（同一视频可能对应多条样本）
    report_lines: List[str] = []

    for sample in dataset:
        raw_path = sample["path"]
        resolved = resolve_video_path(raw_path, args.video_base_dir)

        if resolved not in actions:
            # 路径正常，原样保留
            fixed_dataset.append(sample)
            continue

        action = actions[resolved]
        sample_id = str(sample.get("problem_id", sample.get("video_id", "")))

        if action[0] == "format_fix":
            new_suffix = action[1]
            new_raw_path = str(Path(raw_path).with_suffix(new_suffix))
            report_lines.append(
                f"[FORMAT_FIX] sample={sample_id}  {raw_path} -> {new_raw_path}"
            )
            fixed_dataset.append({**sample, "path": new_raw_path})

        elif action[0] == "convert":
            report_lines.append(
                f"[CONVERTED]  sample={sample_id}  {action[1]} -> {action[2]}"
            )
            # 转换后文件位于原路径，数据集无需修改
            fixed_dataset.append(sample)

        else:  # remove
            report_lines.append(
                f"[REMOVED]    sample={sample_id}  {resolved}"
            )
            # 不加入 fixed_dataset，即删除

    # ---- 7. 统计 ----
    n_format_fix = sum(1 for l in report_lines if l.startswith("[FORMAT_FIX]"))
    n_converted  = sum(1 for l in report_lines if l.startswith("[CONVERTED]"))
    n_removed    = sum(1 for l in report_lines if l.startswith("[REMOVED]"))
    n_unique_fix = sum(1 for a in actions.values() if a[0] == "format_fix")
    n_unique_cvt = sum(1 for a in actions.values() if a[0] == "convert")
    n_unique_rm  = sum(1 for a in actions.values() if a[0] == "remove")

    print(f"\n{'='*60}")
    print(f"修复结果")
    print(f"{'='*60}")
    print(f"  路径修正: {n_unique_fix} 个视频 ({n_format_fix} 条样本)")
    print(f"  格式转换: {n_unique_cvt} 个视频 ({n_converted} 条样本)")
    print(f"  删除缺失: {n_unique_rm} 个视频 ({n_removed} 条样本)")
    print(f"  数据集:   {len(dataset)} -> {len(fixed_dataset)} 条样本")
    print(f"{'='*60}")

    # ---- 8. 保存修复后的数据集 ----
    fixed_path = os.path.join(args.output_dir, "dataset_fixed.json")
    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed_dataset, f, indent=2, ensure_ascii=False)

    # ---- 9. 写入完整修复报告 ----
    report_path = os.path.join(args.output_dir, "fix_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("视频路径修复报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"原始数据集:      {args.dataset_path}\n")
        f.write(f"缺失路径文件:    {args.missing_file}\n")
        f.write(f"视频解码后端:    {backend}\n")
        f.write(f"支持格式:        {', '.join(sorted(supported_exts))}\n")
        f.write(f"原始样本数:      {len(dataset)}\n")
        f.write(f"修复后样本数:    {len(fixed_dataset)}\n")
        f.write(f"删除样本数:      {n_removed}\n")
        f.write("=" * 70 + "\n\n")

        for tag, title in [
            ("[FORMAT_FIX]", "路径修正（扩展名不匹配，已修改数据集路径）"),
            ("[CONVERTED]",  "格式转换（解码器不支持，已用 ffmpeg 转换）"),
            ("[REMOVED]",    "已删除（视频文件完全缺失）"),
        ]:
            entries = [l for l in report_lines if l.startswith(tag)]
            f.write(f"--- {title} [{len(entries)} 条] ---\n")
            if entries:
                for line in entries:
                    f.write(f"  {line}\n")
            else:
                f.write("  (无)\n")
            f.write("\n")

    print(f"\n修复后数据集: {fixed_path}")
    print(f"修复报告:     {report_path}")


if __name__ == "__main__":
    main()
