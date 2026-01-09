from transformers import AutoProcessor

model_id = "Qwen/Qwen2-VL-2B-Instruct" 
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("1. 刚加载时，尝试读取 pad_token_id ...")
try:
    print(f"Processor pad_id: {processor.pad_token_id}")
except AttributeError as e:
    print(f"--> 报错了 (符合预期): {e}")
    print("--> 结论：原生 Processor 确实没有这个属性。")

print("\n" + "="*40 + "\n")

print("2. 执行训练脚本里的'强行注入'操作 ...")
# --- 模拟 grpo_trainer.py 的那三行代码 ---
# 从 tokenizer 拿真值
real_pad_id = processor.tokenizer.pad_token_id
real_eos_id = processor.tokenizer.eos_token_id

# 强行赋值给 processor (动态添加属性)
processor.pad_token_id = real_pad_id
processor.eos_token_id = real_eos_id
# ---------------------------------------
print("--> 注入完成。")

print("\n" + "="*40 + "\n")

print("3. 再次尝试读取 ...")
try:
    print(f"现在 Processor pad_id 是: {processor.pad_token_id}")
    print(f"现在 Processor eos_id 是: {processor.eos_token_id}")
    print("--> 成功！现在它可以被 GRPOTrainer 正常使用了。")
except AttributeError as e:
    print(f"--> 依然报错: {e}")