#!/usr/bin/env python
# filter_json.py
import json
import os
import argparse
from pathlib import Path

EXTS = (".mp4", ".mkv", ".avi")

def find_video(vid: str, root: Path):
    """返回存在的视频路径，否则 None"""
    if vid.lower().endswith(EXTS):
        f = root / vid
        return str(f) if f.exists() else None
    for e in EXTS:
        f = root / (vid + e)
        if f.exists():
            return str(f)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="原始 JSON 文件")
    parser.add_argument("--video-root", required=True, help="视频根目录")
    parser.add_argument("--out", required=True, help="输出 JSONL")
    args = parser.parse_args()

    video_root = Path(args.video_root).expanduser().resolve()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)          # 一次性读成 list[dict]

    kept, miss = 0, 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for sample in data:
            vid = sample["vid"]
            if find_video(vid, video_root):
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            else:
                miss += 1
                if miss <= 5:
                    print(f"❌ 缺失: {vid}")

    print(f"保留 {kept} 条 | 缺失 {miss} 条 | 已写入 {args.out}")

if __name__ == "__main__":
    main()