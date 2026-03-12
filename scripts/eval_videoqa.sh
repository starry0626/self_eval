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
MODEL_PATH="output/sdpo/checkpoint-63"

# 数据集类型: mmvu / mvbench / tempcompass / videomme / videommmu / vsibench
DATASET_TYPE="tempcompass"
DATASET_PATH="./src/open_r1_video/eval/eval_${DATASET_TYPE}.json"

# 视频根目录，用于解析数据集中的相对路径（如 ./Evaluation/VideoMME/xxx.mp4）
VIDEO_BASE_DIR="data"

OUTPUT_DIR="./output/eval/${DATASET_TYPE}/$(basename $MODEL_PATH)"

mkdir -p "$OUTPUT_DIR"

# ================= 评估配置 =================
# think:  先思考后回答，从 <answer> XML 标签提取最终答案
# direct: 直接输出字母/数字，优先尝试 <answer> 标签，再 fallback
ANSWER_MODE=direct

# 留空则评估全部样本
MAX_SAMPLES=

# 是否在加载模型前预检查所有视频路径（true/留空）
CHECK_VIDEO_PATHS=

# ================= 视频处理配置 =================
FPS=1.0
MAX_FRAMES=64

# 留空则使用处理器默认值
MAX_PIXELS=262144   # 256 * 32 * 32

# ================= 模型推理配置 =================
# think 模式推理链较长，建议适当调大（如 2048）
MAX_NEW_TOKENS=1024
ATTN_IMPLEMENTATION=flash_attention_2
TORCH_DTYPE=bfloat16

# ================= vLLM 推理配置 =================
# 设置为 true 启用 vLLM 离线批量推理（速度更快）
USE_VLLM=true
# vLLM 张量并行 GPU 数
TENSOR_PARALLEL_SIZE=1
# vLLM GPU 显存利用率（0-1）
GPU_MEMORY_UTILIZATION=0.80
# vLLM 最大序列长度（留空使用模型默认值）
MAX_MODEL_LEN=
# vLLM 每批推理样本数
VLLM_BATCH_SIZE=16

# ================= 构建可选参数 =================
OPTIONAL_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_samples $MAX_SAMPLES"
fi
if [ -n "$MAX_PIXELS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_pixels $MAX_PIXELS"
fi
if [ "$CHECK_VIDEO_PATHS" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --check_video_paths"
fi
if [ "$USE_VLLM" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --use_vllm"
    OPTIONAL_ARGS="$OPTIONAL_ARGS --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
    OPTIONAL_ARGS="$OPTIONAL_ARGS --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"
    OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_batch_size $VLLM_BATCH_SIZE"
    if [ -n "$MAX_MODEL_LEN" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --max_model_len $MAX_MODEL_LEN"
    fi
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
if [ "$USE_VLLM" = "true" ]; then
echo "  推理后端:   vLLM"
echo "  张量并行:   $TENSOR_PARALLEL_SIZE"
echo "  显存利用率: $GPU_MEMORY_UTILIZATION"
echo "  最大序列长度: $MAX_MODEL_LEN"
echo "  批量大小:   $VLLM_BATCH_SIZE"
else
echo "  推理后端:   HuggingFace Transformers"
fi
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
