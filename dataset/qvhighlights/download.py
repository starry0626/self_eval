import os
import subprocess
import time

# --- 配置区域 ---
# 使用 hf-mirror 镜像加速下载
BASE_URL = "https://hf-mirror.com/datasets/data-process/QVHighlights-zip/resolve/main"

# 文件名前缀 (该数据集文件名为 QVHighlights.part1.rar, part2.rar ...)
FILE_PREFIX = "QVHighlights.part"
FILE_EXTENSION = ".rar"

# 该数据集共有 8 个分卷 (part1 - part8)
MAX_INDEX = 8 
# 起始索引 (该数据集从 part1 开始，而不是 part0)
START_INDEX = 1

# 下载保存目录 (建议改为当前目录下的子文件夹，避免权限问题)
TARGET_DIR = "./QVHighlights"
# ----------------

def check_is_html(filepath):
    """检查文件是否是网页报错文件（如404页面）"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(100)
            # 检查常见的 HTML 标签
            if b'<!DOCTYPE html>' in header or b'<html' in header:
                return True
            # HuggingFace 某些报错是纯文本 "Entry not found"
            if b'Entry not found' in header:
                return True
    except:
        pass
    return False

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    filepath = os.path.join(TARGET_DIR, filename)
    
    print(f"\n>>>>> 正在检查/下载: {filename}")

    # 1. 检查是否存在 HTML 错误文件或损坏文件
    if os.path.exists(filepath):
        if check_is_html(filepath):
            print(f"❌ 发现错误文件（HTML网页或404），正在删除并重下: {filename}")
            os.remove(filepath)
        # 注意：RAR分卷很大(20GB)，如果文件太小(小于10MB)通常也是不对的
        elif os.path.getsize(filepath) < 10 * 1024 * 1024: 
            print(f"⚠️ 文件过小，可能已损坏，重新下载: {filename}")
            os.remove(filepath)

    # 2. 调用 wget 下载
    # 尝试 3 次
    for attempt in range(1, 4):
        try:
            # -c: 断点续传
            # --no-check-certificate: 跳过证书检查
            # -O: 指定输出文件名
            cmd = [
                "wget",
                "-c",
                "--no-check-certificate",
                "-O", filepath,
                url
            ]
            
            # 打印实际执行的命令方便调试
            # print("Exec:", " ".join(cmd))
            
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                #再一次检查下载下来的文件是否变成了HTML（防止wget把404页面存下来返回0）
                if check_is_html(filepath):
                   print(f"❌ 下载似乎成功但内容是网页报错，删除重试...")
                   os.remove(filepath)
                   continue

                print(f"✅ {filename} 下载成功")
                return True
            else:
                print(f"⚠️ 下载中断 (尝试 {attempt}/3)，等待后重试...")
                time.sleep(3) # 等待3秒
                
        except Exception as e:
            print(f"执行出错: {e}")
    
    print(f"❌ {filename} 多次尝试下载失败，请检查网络或URL。")
    return False

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"创建目录: {TARGET_DIR}")
    
    print(f"开始下载 QVHighlights-zip 数据集...")
    print(f"目标目录: {TARGET_DIR}")
    print(f"预计分卷: part{START_INDEX} 到 part{MAX_INDEX}")
    
    success_count = 0
    fail_count = 0
    
    # 遍历分卷 (注意：range不包含结束值，所以要 +1)
    for i in range(START_INDEX, MAX_INDEX + 1):
        # 构造文件名: QVHighlights.part1.rar
        filename = f"{FILE_PREFIX}{i}{FILE_EXTENSION}"
        
        # 尝试下载
        if download_file(filename):
            success_count += 1
        else:
            print(f"❌ 文件 {filename} 下载失败。")
            fail_count += 1

    print("\n" + "="*50)
    print(f"任务结束. 成功: {success_count} 个分卷, 失败: {fail_count} 个")
    
    if fail_count == 0:
        print("🎉 所有分卷下载完成！")
        print("由于这是 RAR 分卷格式，请使用以下命令解压（只需指定第一个分卷）：")
        print("-" * 50)
        print(f"cd {TARGET_DIR}")
        print("# 需要安装 unrar (sudo apt install unrar) 或 7zip (sudo apt install p7zip-full)")
        print(f"unrar x {FILE_PREFIX}{START_INDEX}{FILE_EXTENSION}")
        print("-" * 50)
    else:
        print("❌ 仍有文件下载失败，请重新运行脚本。")

if __name__ == "__main__":
    main()