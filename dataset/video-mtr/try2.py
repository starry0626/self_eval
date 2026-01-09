from transformers import AutoModelForImageTextToText, AutoProcessor
model_dir = "Qwen3-VL-2B-Thinking"# 不同尺寸模型的配置信息有所不同，需要甄别预处理部分的配置是否不同
messages = [    {"role": "system", 
                "content": [{"type": "text", "text": "You are a helpful assistant."}]},  
                {"role": "user",     
                 "content": [       
                     { "type": "video",          
                       "video": "data/Standard_Mode_Starting_from_her_iconic_over_th.mp4" },
                       {"type": "text", 
                        "text": "描述下这个视频"},     
                        ] }]
processor = AutoProcessor.from_pretrained(model_dir)# Preparation for inference
print(type(processor))
inputs = processor.apply_chat_template(    messages,    tokenize=True,    add_generation_prompt=True,    return_dict=True,    return_tensors="pt")
print("inputs keys=", list(inputs.keys()))# inputs keys= ['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw']
print("inputs['input_ids'] shape= ", inputs['input_ids'].shape)
print("inputs['pixel_values_videos'] shape= ", inputs['pixel_values_videos'].shape)
print("inputs['video_grid_thw']= ", inputs['video_grid_thw'])


