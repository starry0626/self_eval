#!/bin/bash

# ================= VideoQA 评估脚本 =================
# 在视频问答基准上评估 Qwen3-VL 系列模型
# 支持数据集: mmvu / mvbench / tempcompass / videomme / videommmu / vsibench
# 支持模式:   think（先思考后回答）/ direct（直接回答）

# ================= GPU 配置 =================
# 参与推理的 GPU 编号（逗号分隔）；多卡时 device_map="auto" 自动分片
CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

# ================= 路径配置 =================
MODEL_PATH="./Qwen3-VL-2B-Thinking"

# 数据集类型: mmvu / mvbench / tempcompass / videomme / videommmu / vsibench
DATASET_TYPE="videomme"
DATASET_PATH="./src/open_r1_video/eval/eval_${DATASET_TYPE}.json"

# 视频根目录，用于解析数据集中的相对路径（如 ./Evaluation/VideoMME/xxx.mp4）
VIDEO_BASE_DIR="./video_data"

OUTPUT_DIR="./output/eval/${DATASET_TYPE}/$(basename $MODEL_PATH)"

mkdir -p "$OUTPUT_DIR"

# ================= 评估配置 =================
# think:  先思考后回答，从 <answer> XML 标签提取最终答案
# direct: 直接输出字母/数字，优先尝试 <answer> 标签，再 fallback
ANSWER_MODE=think

# 留空则评估全部样本
MAX_SAMPLES=

# ================= 视频处理配置 =================
FPS=1.0
MAX_FRAMES=32

# 留空则使用处理器默认值
MAX_PIXELS=

# ================= 模型推理配置 =================
# think 模式推理链较长，建议适当调大（如 2048）
MAX_NEW_TOKENS=1024
ATTN_IMPLEMENTATION=flash_attention_2
TORCH_DTYPE=bfloat16

# ================= 构建可选参数 =================
OPTIONAL_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_samples $MAX_SAMPLES"
fi
if [ -n "$MAX_PIXELS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_pixels $MAX_PIXELS"
fi

# ================= 打印启动信息 =================
echo "=========================================="
echo "  VideoQA 评估"
echo "  GPU:        $CUDA_VISIBLE_DEVICES"
echo "  模型:       $MODEL_PATH"
echo "  数据集:     $DATASET_TYPE"
echo "  数据集路径: $DATASET_PATH"
echo "  回答模式:   $ANSWER_MODE"
echo "  输出:       $OUTPUT_DIR"
echo "=========================================="

# ================= 启动命令 =================
python src/open_r1_video/eval_videoqa.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --dataset_type $DATASET_TYPE \
    --video_base_dir $VIDEO_BASE_DIR \
    --output_dir $OUTPUT_DIR \
    \
    `# 评估配置` \
    --answer_mode $ANSWER_MODE \
    \
    `# 视频处理` \
    --fps $FPS \
    --max_frames $MAX_FRAMES \
    \
    `# 模型推理` \
    --max_new_tokens $MAX_NEW_TOKENS \
    --attn_implementation $ATTN_IMPLEMENTATION \
    --torch_dtype $TORCH_DTYPE \
    \
    $OPTIONAL_ARGS

echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
