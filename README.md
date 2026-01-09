# 数据集
## 原始视频下载
使用video-mtr数据集. 该数据集的原视频来自QVHighlights以及Next-gqa两个数据集. 两个数据集的原视频可以从Huggingfaces下载.
[QVHighlights原视频huggingface下载链接](https://huggingface.co/datasets/data-process/QVHighlights-zip)
[next-gqa原视频huggingface下载链接](https://huggingface.co/datasets/jinyoungkim/NExT-GQA)
下载后分别替换`./QVHighlights`与`./NExT-GQA`目录
## 下载可能遇到的问题
华为云服务器通过镜像可以连接到huggingface, 另外有些文件下载需要权限, 这时得去huggingface网站里申请一个token.
如果是503错误则为网络连接错误, 401错误则为权限错误.
最快的下载方法似乎还是huggingface-cli. 同时开启高速下载(甚至能达到百兆每秒): 记得配置huggingface国内镜像.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
# 环境配置
```
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
cd qwen-vl-utils
pip install -e .
cd ..
```
# 模型下载
下载后替换`./Qwen3-VL-2B-Thinking`目录