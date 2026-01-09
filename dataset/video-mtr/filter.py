import json
import os

# 原始文件路径
file_path = "/home/ma-user/work/self_eval/dataset/video-mtr/qv-nextgqa_merge_8k.json"
# 备份路径 (为了安全)
backup_path = file_path + ".bak"

print(f"读取文件: {file_path}")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 创建备份
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已创建备份: {backup_path}")
    
    # 2. 修复数据
    fixed_count = 0
    for item in data:
        # 获取当前的 problem_id
        pid = item.get('problem_id')
        
        # 强制转换为字符串
        if not isinstance(pid, str):
            item['problem_id'] = str(pid)
            fixed_count += 1
            
    print(f"修复完成！共将 {fixed_count} 个非字符串 id 转换为字符串。")
    
    # 3. 保存回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("文件已保存，现在可以重新运行训练代码了。")

except Exception as e:
    print(f"处理出错: {e}")