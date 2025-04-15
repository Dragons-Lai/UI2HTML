#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI2HTML - 推理脚本
用于使用基础模型和微调模型进行推理，生成HTML
"""

import os
import torch
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm

# 系统消息和提示
prompt = """Convert the provided UI screenshot into clean, semantic HTML code with inline CSS.
The code should be responsive and follow best practices for web development.
Do not include any JavaScript.

Please provide only the HTML code."""

system_message = "You are an expert UI developer specializing in converting design mockups into clean HTML code."

# 加载基础模型和处理器
def load_models():
    print("加载模型中...")
    # Hugging Face模型ID
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_path = "./qwen2-7b-instruct-ui2html"
    
    # 加载基础模型
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("模型加载完成")
    
    return base_model, processor, adapter_path

# 从图片生成HTML
def generate_html(image, model, processor):
    # 创建消息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]

    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 生成输出
    max_tokens = 1024

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        top_p=0.9,
        do_sample=True,
        temperature=0.7
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 提取HTML内容
def extract_html(text):
    """从文本中提取HTML内容"""
    import re

    # 查找具有不同模式的HTML内容
    patterns = [
        r'<html.*?>.*?</html>',  # 标准HTML
        r'<!DOCTYPE.*?>.*?</html>',  # DOCTYPE声明
        r'<[^>]+>.*</[^>]+>'  # 任何类似HTML的标签
    ]

    # 尝试每个模式直到找到匹配项
    html_content = None
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            html_content = match.group(0)
            break

    if not html_content:
        return "未找到HTML内容。"

    return html_content

# 加载测试数据集
def load_test_dataset():
    print("加载测试数据集...")
    # 加载Design2Code数据集
    dataset_id = "SALT-NLP/Design2Code-hf"
    full_dataset = load_dataset(dataset_id, split="train")
    
    # 加载测试索引
    try:
        with open("test_indices.json", "r") as f:
            test_indices = json.load(f)["test_indices"]
        
        # 创建测试数据集
        test_dataset = full_dataset.select(test_indices)
        print(f"测试集大小: {len(test_dataset)}")
        return test_dataset
    except FileNotFoundError:
        print("未找到test_indices.json文件。请先运行训练脚本生成测试索引。")
        return None

# 进行基础模型推理
def run_base_model_inference(model, processor, test_dataset):
    print("使用基础模型进行推理...")
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/original", exist_ok=True)
    os.makedirs("test_results/base_model", exist_ok=True)
    
    # 处理测试子集
    max_test_samples = min(len(test_dataset), 50)  # 根据资源调整
    test_subset = test_dataset.select(range(max_test_samples))
    
    base_outputs = []
    
    for i, test_sample in enumerate(tqdm(test_subset, desc="基础模型推理")):
        sample_id = f"sample_{i}"
        
        try:
            # 保存原始UI图像
            original_image = test_sample["image"]
            original_path = f"test_results/original/{sample_id}.png"
            
            # 如果是PIL图像，直接保存
            if hasattr(original_image, 'save'):
                original_image.save(original_path)
            
            # 使用基础模型生成HTML
            base_html_raw = generate_html(original_image, model, processor)
            base_html = extract_html(base_html_raw)
            
            # 保存基础模型HTML
            with open(f"test_results/base_model/{sample_id}.html", "w", encoding="utf-8") as f:
                f.write(base_html)
            
            # 存储输出以便稍后比较
            base_outputs.append({
                'sample_id': sample_id,
                'original_image': original_image,
                'original_path': original_path,
                'base_html': base_html
            })
            
        except Exception as e:
            print(f"处理样本 {i} 的基础模型时出错: {e}")
            base_outputs.append(None)
    
    # 保存base_outputs以便后续使用
    with open("test_results/base_outputs.json", "w") as f:
        # 处理不可序列化的内容
        serializable_outputs = []
        for output in base_outputs:
            if output is not None:
                serializable_outputs.append({
                    'sample_id': output['sample_id'],
                    'original_path': output['original_path'],
                    'base_html': output['base_html']
                })
        json.dump(serializable_outputs, f)
    
    return base_outputs

# 进行微调模型推理
def run_finetuned_model_inference(model, processor, adapter_path, base_outputs):
    print("加载适配器以创建微调模型...")
    model.load_adapter(adapter_path)
    
    os.makedirs("test_results/fine_tuned_model", exist_ok=True)
    
    ft_results = []
    
    for i, data in enumerate(tqdm(base_outputs, desc="微调模型推理")):
        if data is None:
            continue
        
        sample_id = data['sample_id']
        
        try:
            # 使用微调模型生成HTML
            ft_html_raw = generate_html(data['original_image'], model, processor)
            ft_html = extract_html(ft_html_raw)
            
            # 保存微调模型HTML
            with open(f"test_results/fine_tuned_model/{sample_id}.html", "w", encoding="utf-8") as f:
                f.write(ft_html)
            
            # 保存结果
            ft_results.append({
                'sample_id': sample_id,
                'ft_html': ft_html
            })
            
        except Exception as e:
            print(f"处理样本 {i} 的微调模型时出错: {e}")
            continue
    
    # 保存ft_results以便后续使用
    with open("test_results/ft_results.json", "w") as f:
        json.dump(ft_results, f)
    
    return ft_results

# 主函数
if __name__ == "__main__":
    print("开始UI2HTML推理脚本...")
    
    # 加载模型和测试数据集
    base_model, processor, adapter_path = load_models()
    test_dataset = load_test_dataset()
    
    if test_dataset is not None:
        # 运行基础模型推理
        base_outputs = run_base_model_inference(base_model, processor, test_dataset)
        
        # 运行微调模型推理
        ft_results = run_finetuned_model_inference(base_model, processor, adapter_path, base_outputs)
        
        print("推理脚本执行完毕")
    else:
        print("无法加载测试数据集，脚本终止")