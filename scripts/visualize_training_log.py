#!/usr/bin/env python3
"""
Training Log KL Divergence Visualizer

读取训练日志 JSONL 文件，生成交互式 HTML 可视化页面。
- 将子词 token 合并为自然语言词，聚合 KL 散度（同时提供 max 和 mean）
- 词级着色：绿色（低 KL）→ 黄色 → 红色（高 KL）
- 悬浮提示显示精确 KL 数值
- 支持按 KL 排序、展开/折叠、颜色模式切换（max/mean）

Usage:
    python scripts/visualize_training_log.py training_log_rank0.jsonl
    python scripts/visualize_training_log.py training_log_rank0.jsonl training_log_rank1.jsonl -o viz.html
"""

import argparse
import json
import math
import os
import sys
from html import escape


def merge_subwords(tokens, kl_values):
    """
    将子词 token 合并为自然语言词，并聚合 KL 散度值。

    合并规则：
    - Ġ 前缀 = 词边界（BPE 编码中表示前面有空格）
    - Ċ = 换行符，作为词边界
    - <...> 形式的特殊 token（如 <think>）独立成词
    - 其他 token 拼接到当前词

    返回:
        list[dict]: 每个词包含 text, kl_max, kl_mean, space_before
    """
    if not tokens or not kl_values:
        return []

    words = []
    buf_text = ""
    buf_kls = []
    buf_space = False

    def flush():
        nonlocal buf_text, buf_kls, buf_space
        if buf_text:
            words.append({
                "text": buf_text,
                "kl_max": max(buf_kls),
                "kl_mean": sum(buf_kls) / len(buf_kls),
                "space_before": buf_space,
            })
            buf_text = ""
            buf_kls = []
            buf_space = False

    for tok, kl in zip(tokens, kl_values):
        # 特殊 token（如 <think>, </think>, <|im_end|>）独立成词
        if tok.startswith("<") and ">" in tok:
            flush()
            words.append({
                "text": tok,
                "kl_max": kl,
                "kl_mean": kl,
                "space_before": True,
            })
            continue

        # Ġ 前缀 = 词边界（新词，前面有空格）
        if tok.startswith("Ġ"):
            flush()
            buf_space = True
            buf_text = tok[1:]
            buf_kls = [kl]
        # Ċ = 换行符，作为词边界
        elif tok.startswith("Ċ"):
            flush()
            buf_space = True
            buf_text = "\n" + tok[1:]
            buf_kls = [kl]
        elif not buf_text:
            # 序列第一个 token
            buf_text = tok
            buf_kls = [kl]
            buf_space = False
        else:
            # 当前词的延续（子词拼接）
            buf_text += tok
            buf_kls.append(kl)

    flush()
    return words


def generate_html(records, output_path):
    """生成交互式 HTML 可视化文件"""

    # ── 步骤 1: 合并子词 & 收集统计数据 ──
    all_word_kl_maxes = []
    for rec in records:
        rec["_words"] = merge_subwords(rec["tokens"], rec["kl_per_token"])
        all_word_kl_maxes.extend(w["kl_max"] for w in rec["_words"])

    # 计算归一化阈值（95th percentile, log scale）
    all_word_kl_maxes.sort()
    n = len(all_word_kl_maxes)
    if n > 0:
        p95 = all_word_kl_maxes[min(int(n * 0.95), n - 1)]
        median_word_kl = all_word_kl_maxes[n // 2]
    else:
        p95 = 1.0
        median_word_kl = 0.0
    log_p95 = math.log(1 + p95) if p95 > 0 else 1.0

    # 样本级统计
    total = len(records)
    mean_sample_kl = (
        sum(r["mean_kl_divergence"] for r in records) / total if total else 0
    )
    median_sample_kl = (
        sorted(r["mean_kl_divergence"] for r in records)[total // 2] if total else 0
    )

    # ── 步骤 2: 按 mean KL 降序排列 ──
    records.sort(key=lambda r: r["mean_kl_divergence"], reverse=True)

    # ── 步骤 3: 生成每个样本的 HTML 卡片 ──
    cards_html_parts = []
    for rank, rec in enumerate(records):
        mean_kl = rec["mean_kl_divergence"]
        # 卡片头部 KL badge 颜色
        norm_badge = (
            min(math.log(1 + mean_kl) / log_p95, 1.0) if log_p95 > 0 else 0
        )
        badge_h = 120 * (1 - norm_badge)
        badge_l = 88 - 30 * norm_badge
        badge_bg = f"hsl({badge_h:.0f},70%,{badge_l:.0f}%)"
        badge_tc = "#fff" if norm_badge > 0.7 else "#333"

        q = escape(rec.get("question", "") or "")
        opts_html = "  ".join(
            f"<b>{escape(k)}.</b> {escape(v)}"
            for k, v in rec.get("options", {}).items()
        )
        video = escape(rec.get("video_path", "") or "")
        video_basename = escape(os.path.basename(rec.get("video_path", "") or ""))
        sample_id = escape(str(rec.get("sample_id", "")))
        step = rec.get("global_step", "")
        comp_len = rec.get("completion_length", "")

        # 生成词级别的 <span> 元素
        word_spans = []
        for w in rec["_words"]:
            text = escape(w["text"])
            has_br = "\n" in w["text"]
            if has_br:
                text = text.replace("\n", "<br>")
            prefix = " " if w["space_before"] else ""
            kl_max_str = f"{w['kl_max']:.4f}"
            kl_mean_str = f"{w['kl_mean']:.4f}"
            title = f"KL max: {kl_max_str} | mean: {kl_mean_str}"
            word_spans.append(
                f'{prefix}<span class="w" '
                f'data-x="{kl_max_str}" data-y="{kl_mean_str}" '
                f'title="{title}">{text}</span>'
            )

        answer_html = "".join(word_spans)

        q_short = q[:80] + ("..." if len(q) > 80 else "")

        # 新字段：正确答案和教师额外信息（旧日志无这些字段时 .get() 返回空值）
        correct_answer_text = escape(rec.get("correct_answer_text", "") or "")
        teacher_extra = rec.get("teacher_extra_info", {}) or {}

        # 构建 Correct Answer 行（仅有内容时显示）
        correct_answer_html = ""
        if correct_answer_text:
            correct_answer_html = (
                f'    <div class="mt" style="background:#e6f9e6;padding:4px 8px;border-radius:4px">'
                f'<span class="lb" style="color:#2a7a2a">Correct Answer:</span> '
                f'<span style="color:#2a7a2a;font-weight:600">{correct_answer_text}</span></div>\n'
            )

        # 构建 Teacher Extra Info 面板
        teacher_info_html = ""
        if teacher_extra:
            parts = []
            if teacher_extra.get("answer"):
                parts.append(
                    f'<div class="mt"><span class="lb">Answer:</span> {escape(str(teacher_extra["answer"]))}</div>'
                )
            if teacher_extra.get("temporal_text"):
                tl = teacher_extra["temporal_text"]
                lines = "".join(
                    f'<div style="margin-left:12px">[{escape(s.get("timestamp",""))}] {escape(s.get("description",""))}</div>'
                    for s in tl
                )
                parts.append(f'<div class="mt"><span class="lb">Temporal Text:</span>{lines}</div>')
            if teacher_extra.get("reasoning"):
                r_text = escape(teacher_extra["reasoning"])
                # 截断过长的推理文本，避免卡片过大
                if len(r_text) > 600:
                    r_text = r_text[:600] + "..."
                parts.append(
                    f'<div class="mt"><span class="lb">Reasoning:</span>'
                    f'<div style="margin-left:12px;white-space:pre-wrap;font-size:12px;max-height:200px;overflow-y:auto">{r_text}</div></div>'
                )
            if teacher_extra.get("temporal_video_frames"):
                tvf = teacher_extra["temporal_video_frames"]
                lines = "".join(
                    f'<div style="margin-left:12px">[{escape(s.get("segment",""))}] '
                    f'{escape(s.get("description",""))} '
                    f'({s.get("num_frames",0)} frames: {escape(str(s.get("frame_timestamps",[])))})</div>'
                    for s in tvf
                )
                parts.append(f'<div class="mt"><span class="lb">Temporal Video Frames:</span>{lines}</div>')

            if parts:
                inner = "\n".join(parts)
                teacher_info_html = (
                    f'    <div style="background:#f5f5f5;padding:10px;border-radius:6px;'
                    f'border:1px solid #e0e0e0;margin-top:6px;margin-bottom:6px">\n'
                    f'      <div class="lb" style="margin-bottom:4px">Teacher Extra Info:</div>\n'
                    f'      {inner}\n'
                    f'    </div>\n'
                )

        cards_html_parts.append(
            f'<div class="card" data-kl="{mean_kl:.6f}" data-step="{step}">\n'
            f'  <div class="hdr" onclick="T(this)">\n'
            f'    <span class="rk">#{rank + 1}</span>\n'
            f'    <span class="bd" style="background:{badge_bg};color:{badge_tc}">KL {mean_kl:.4f}</span>\n'
            f'    <span class="st">Step {step}</span>\n'
            f'    <span class="qt">{q_short}</span>\n'
            f'    <span class="vd">{video_basename}</span>\n'
            f'    <span class="ar">&#9654;</span>\n'
            f"  </div>\n"
            f'  <div class="bd2">\n'
            f'    <div class="mt"><span class="lb">ID:</span> {sample_id}</div>\n'
            f'    <div class="mt"><span class="lb">Question:</span> {q}</div>\n'
            f'    <div class="mt"><span class="lb">Options:</span> {opts_html}</div>\n'
            f'{correct_answer_html}'
            f'    <div class="mt"><span class="lb">Video:</span> {video}</div>\n'
            f'    <div class="mt"><span class="lb">Length:</span> {comp_len} tokens | '
            f'<span class="lb">Mean KL:</span> {mean_kl:.4f}</div>\n'
            f'{teacher_info_html}'
            f'    <div class="lb" style="margin-top:8px">Student Answer:</div>\n'
            f'    <div class="ans">{answer_html}</div>\n'
            f"  </div>\n"
            f"</div>"
        )

    samples_html = "\n".join(cards_html_parts)

    # ── 步骤 4: 组装完整 HTML ──
    html = _HTML_TEMPLATE
    html = html.replace("__LOG_P95__", f"{log_p95:.6f}")
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__MEAN_KL__", f"{mean_sample_kl:.4f}")
    html = html.replace("__MEDIAN_KL__", f"{median_sample_kl:.4f}")
    html = html.replace("__P95__", f"{p95:.4f}")
    html = html.replace("__SAMPLES__", samples_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ═══════════════════════════════════════════════════════════════════
# HTML 模板（CSS + JS 内联，零外部依赖）
# ═══════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>KL Divergence Visualization</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#f0f2f5;padding:20px;color:#333}
h1{text-align:center;margin-bottom:6px;color:#1a1a2e;font-size:22px}
.info{text-align:center;color:#888;margin-bottom:14px;font-size:13px}
.ctl{display:flex;gap:8px;justify-content:center;align-items:center;
  flex-wrap:wrap;margin-bottom:14px}
.ctl button,.ctl select{padding:5px 14px;border:1px solid #ccc;border-radius:6px;
  background:#fff;cursor:pointer;font-size:13px}
.ctl button:hover{background:#e8e8e8}
.ctl button.on{background:#4a90d9;color:#fff;border-color:#4a90d9}
.ctl select{padding:4px 8px}
.leg{display:flex;justify-content:center;align-items:center;gap:6px;
  margin-bottom:18px;font-size:12px;color:#777}
.leg-bar{width:220px;height:14px;border-radius:3px;
  background:linear-gradient(to right,hsl(120,75%,85%),hsl(60,75%,75%),hsl(0,75%,60%))}
.card{background:#fff;border-radius:8px;margin-bottom:6px;
  box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
.hdr{display:flex;align-items:center;gap:10px;padding:10px 16px;
  cursor:pointer;user-select:none;font-size:13px}
.hdr:hover{background:#fafafa}
.rk{font-weight:700;color:#888;min-width:40px}
.bd{padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
.st{color:#999;font-size:12px;min-width:60px}
.qt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#555}
.vd{color:#aaa;font-size:11px;max-width:180px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.ar{transition:transform .2s;color:#ccc;font-size:11px}
.card.open .ar{transform:rotate(90deg)}
.bd2{display:none;padding:16px;border-top:1px solid #f0f0f0}
.card.open .bd2{display:block}
.mt{margin-bottom:6px;font-size:14px;line-height:1.6}
.lb{font-weight:600;color:#555}
.ans{font-family:"SF Mono","Fira Code",Consolas,monospace;font-size:13px;
  line-height:2.0;padding:12px;background:#fafafa;border-radius:6px;
  border:1px solid #eee;white-space:pre-wrap;word-break:break-word;margin-top:6px}
.w{display:inline;padding:1px 2px;border-radius:2px;cursor:help}
</style>
</head>
<body>
<h1>KL Divergence Visualization</h1>
<div class="info">
  Total: __TOTAL__ samples &nbsp;|&nbsp;
  Median KL: __MEDIAN_KL__ &nbsp;|&nbsp;
  Mean KL: __MEAN_KL__ &nbsp;|&nbsp;
  P95 (word max): __P95__
</div>
<div class="ctl">
  <button class="on" onclick="S('kl-desc',this)">KL &#8595;</button>
  <button onclick="S('kl-asc',this)">KL &#8593;</button>
  <button onclick="S('step',this)">Step Order</button>
  <button onclick="E()">Expand All</button>
  <button onclick="C()">Collapse All</button>
  <span style="margin-left:12px;font-size:13px;color:#666">Color by:</span>
  <select id="cm" onchange="A(this.value)">
    <option value="max">Max KL</option>
    <option value="mean">Mean KL</option>
  </select>
</div>
<div class="leg">
  <span>Low KL</span>
  <div class="leg-bar"></div>
  <span>High KL</span>
</div>
<div id="ss">
__SAMPLES__
</div>
<script>
var LP=__LOG_P95__;
function K(raw){
  var n=LP>0?Math.min(Math.log(1+raw)/LP,1.0):0;
  var h=120*(1-n),l=92-37*n;
  return["hsl("+h+",75%,"+l+"%)",n>.75?"#fff":"#222"]
}
function A(mode){
  var a=mode==="max"?"x":"y";
  var els=document.querySelectorAll(".w");
  for(var i=0;i<els.length;i++){
    var r=parseFloat(els[i].dataset[a]);
    var c=K(r);
    els[i].style.background=c[0];
    els[i].style.color=c[1]
  }
}
function T(h){h.parentElement.classList.toggle("open")}
function E(){var c=document.querySelectorAll(".card");for(var i=0;i<c.length;i++)c[i].classList.add("open")}
function C(){var c=document.querySelectorAll(".card");for(var i=0;i<c.length;i++)c[i].classList.remove("open")}
function S(mode,btn){
  if(btn){var bs=document.querySelectorAll(".ctl button");for(var i=0;i<bs.length;i++)bs[i].classList.remove("on");btn.classList.add("on")}
  var p=document.getElementById("ss"),cs=[].slice.call(p.children);
  cs.sort(function(a,b){
    if(mode==="kl-desc")return parseFloat(b.dataset.kl)-parseFloat(a.dataset.kl);
    if(mode==="kl-asc")return parseFloat(a.dataset.kl)-parseFloat(b.dataset.kl);
    return parseInt(a.dataset.step)-parseInt(b.dataset.step)
  });
  for(var i=0;i<cs.length;i++){cs[i].querySelector(".rk").textContent="#"+(i+1);p.appendChild(cs[i])}
}
A("max");
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="将训练日志中的 KL 散度可视化为交互式 HTML 热力图"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="训练日志 JSONL 文件路径（支持多个文件合并，如 rank0 + rank1）",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 HTML 文件路径（默认：输入文件同目录下 _viz.html）",
    )
    args = parser.parse_args()

    # 加载所有记录
    records = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("Error: No records found in input file(s).", file=sys.stderr)
        sys.exit(1)

    # 确定输出路径
    output = args.output or os.path.splitext(args.input[0])[0] + "_viz.html"

    generate_html(records, output)
    print(f"Generated: {output}")
    print(f"  Samples: {len(records)}")
    print(f"  Open in browser to view.")


if __name__ == "__main__":
    main()
