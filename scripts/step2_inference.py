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
import time
import zipfile
from huggingface_hub import HfApi, login
from peft import PeftModel

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
    # 打印当前使用的模型类型
    print(f"生成HTML使用的模型类型: {type(model)}")
    print(f"模型是否有PEFT配置: {hasattr(model, 'peft_config')}")
    
    # 创建消息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
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

    # 打印输入信息
    print(f"推理输入形状: {inputs.input_ids.shape}")
    
    # 生成输出
    max_tokens = 1024

    print("开始生成...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    print("生成完成")

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
    # 使用与训练相同的数据集 (webcode2m_purified)
    dataset_id = "xcodemind/webcode2m_purified"
    
    # 检查是否存在测试索引文件
    test_indices_file = "test_indices.json"
    
    try:
        with open(test_indices_file, "r") as f:
            test_indices = json.load(f)["test_indices"]
        
        print(f"成功加载测试索引，共有 {len(test_indices)} 个样本")
    except Exception as e:
        print(f"加载测试索引时出错: {e}")
        return None
    
    try:
        # 使用流式加载数据集
        print(f"使用流式加载数据集 '{dataset_id}'...")
        stream_dataset = load_dataset(dataset_id, split="train", streaming=True)
        
        # 创建一个测试数据集
        from datasets import Dataset
        test_samples = []
        
        # 记录已处理的样本数量和找到的测试样本数量
        processed_count = 0
        found_count = 0
        
        # 对于流式数据集，我们需要遍历并找到匹配test_indices的样本
        print("筛选测试样本...")
        
        # 将test_indices转换为集合以便快速查找
        test_indices_set = set(test_indices)
        
        # 遍历流式数据集
        for example in stream_dataset:
            if processed_count in test_indices_set:
                test_samples.append(example)
                found_count += 1
                
                # 如果找到所有测试样本，可以提前结束
                if found_count == len(test_indices):
                    break
                
                # 每找到10个样本打印一次进度
                if found_count % 10 == 0:
                    print(f"已找到 {found_count}/{len(test_indices)} 个测试样本")
            
            processed_count += 1
            
            # 每处理1000个样本打印一次进度
            if processed_count % 1000 == 0:
                print(f"已处理 {processed_count} 个样本，找到 {found_count}/{len(test_indices)} 个测试样本")
        
        # 将收集的样本转换为正常数据集
        test_dataset = Dataset.from_list(test_samples)
        print(f"测试集大小: {len(test_dataset)} 个样本")
        
        return test_dataset
    except Exception as e:
        print(f"加载或处理数据集时出错: {e}")
        return None

# 加载HuggingFace令牌
def load_hf_token():
    try:
        with open("hf_token.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("未找到hf_token.txt文件，请确保该文件存在并包含有效的HuggingFace令牌")
        return None

# 进行基础模型推理
def run_base_model_inference(model, processor, test_dataset):
    print("使用基础模型进行推理...")
    print(f"基础模型类型: {type(model)}")
    print(f"基础模型是否有适配器: {hasattr(model, 'peft_config')}")
    
    # 确保模型处于评估模式
    model.eval()
    print(f"模型评估模式: {model.training == False}")
    
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/original", exist_ok=True)
    os.makedirs("test_results/base_model", exist_ok=True)
    
    # 处理测试子集
    max_test_samples = min(len(test_dataset), 500)  # 根据资源调整
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
            print(f"\n处理样本 {i} 使用基础模型生成HTML")
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
    
    # 检查适配器路径是否存在
    print(f"适配器路径: {adapter_path}")
    if not os.path.exists(adapter_path):
        print(f"警告: 适配器路径 {adapter_path} 不存在!")
    else:
        print(f"适配器路径存在!")
        # for file in os.listdir(adapter_path):
        #     file_path = os.path.join(adapter_path, file)
        #     if os.path.isdir(file_path):
        #         print(f"  - {file}/ (目录)")
        #     else:
        #         print(f"  - {file} ({os.path.getsize(file_path)} 字节)")
    
    # 打印加载前的参数数量和模型状态
    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"基础模型可训练参数数量: {base_params}")
    print(f"适配器加载前模型类型: {type(model)}")
    print(f"基础模型是否已有适配器: {hasattr(model, 'peft_config')}")
    
    # 加载适配器
    model = PeftModel.from_pretrained(model, adapter_path)
    print("使用PeftModel.from_pretrained成功加载适配器")

    # 打印加载后的参数数量
    ft_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"加载适配器后可训练参数数量: {ft_params}")
    
    # 打印模型状态信息，确认适配器是否正确加载
    print(f"适配器是否已加载: {hasattr(model, 'peft_config')}")
    if hasattr(model, 'peft_config'):
        print(f"PEFT配置: {model.peft_config}")
        # 修复配置访问方式
        if hasattr(model.peft_config, 'target_modules'):
            print(f"PEFT配置中的target_modules: {model.peft_config.target_modules}")
        else:
            # 对于多配置的情况
            if hasattr(model.peft_config, 'default'):
                print(f"PEFT默认配置: {model.peft_config.default}")
                if hasattr(model.peft_config.default, 'target_modules'):
                    print(f"PEFT配置中的target_modules: {model.peft_config.default.target_modules}")
        
        try:
            if hasattr(model, 'state_dict'):
                print(f"适配器模块信息: {list(model.state_dict().keys())[:10]} ...")
                # 检查是否有LoRA权重
                lora_weights = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
                print(f"发现LoRA权重的数量: {len(lora_weights)}")
                if lora_weights:
                    print(f"LoRA权重示例: {lora_weights[:5]}")
        except Exception as e:
            print(f"获取模型状态字典时出错: {e}")
    
    # 确保模型处于评估模式
    model.eval()
    print(f"模型评估模式: {model.training == False}")
    
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

# 上传结果到Hugging Face
def upload_to_huggingface(zip_path, hf_repo_id=None):
    """上传结果到Hugging Face Hub"""
    try:
        # 初始化API
        api = HfApi()
        
        # 获取当前登录的用户信息
        try:
            current_user = api.whoami()
            username = current_user['name']
            repo_name = "ui2html_results"
            hf_repo_id = f"{username}/{repo_name}"
            print(f"使用当前登录用户 '{username}' 上传到仓库: {hf_repo_id}")
        except Exception as e:
            print(f"获取用户信息失败: {e}")
            print("请确保您已通过 huggingface-cli login 登录")
            return False
        
        print(f"正在上传到Hugging Face: {hf_repo_id}")
        
        # 创建一个简单的README
        readme_content = f"""# UI2HTML 推理结果数据

这个仓库包含UI2HTML项目的推理结果数据，用于统计和可视化分析。

## 数据信息
- 创建时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
- 包含内容: 原始图像、基础模型HTML、微调模型HTML、结果数据

## 使用方法
1. 下载 `ui2html_results.zip`
2. 解压到工作目录
3. 使用统计脚本进行分析
"""
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # 检查仓库是否存在
        try:
            api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
            print(f"仓库 {hf_repo_id} 已存在")
        except Exception:
            print(f"创建新仓库 {hf_repo_id}")
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=False)
        
        # 上传文件
        print(f"上传 {zip_path}...")
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=os.path.basename(zip_path),
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        
        # 上传README
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        
        print(f"上传成功! 访问: https://huggingface.co/datasets/{hf_repo_id}")
        return True
    except Exception as e:
        print(f"上传过程中出错: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists("README.md"):
            os.remove("README.md")

# 主函数
if __name__ == "__main__":
    print("开始UI2HTML推理脚本...")
    
    # 登录HuggingFace
    token = load_hf_token()
    if token:
        login(token=token)
        print("已成功登录HuggingFace")
    else:
        print("警告：未设置HuggingFace令牌，可能无法访问模型或推送结果")
    
    # 确保结果目录存在
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/original", exist_ok=True)
    os.makedirs("test_results/base_model", exist_ok=True)
    os.makedirs("test_results/fine_tuned_model", exist_ok=True)
    
    # 加载模型和测试数据集
    base_model, processor, adapter_path = load_models()
    
    # 加载测试数据集
    test_dataset = load_test_dataset()
    
    if test_dataset is not None:
        print(f"成功加载测试数据集，共 {len(test_dataset)} 个样本")
        
        # 运行基础模型推理
        base_outputs = run_base_model_inference(base_model, processor, test_dataset)
        
        if not base_outputs:
            print("基础模型推理没有产生有效输出，脚本终止")
            exit(1)
        
        # 运行微调模型推理
        ft_results = run_finetuned_model_inference(base_model, processor, adapter_path, base_outputs)
        
        if not ft_results:
            print("微调模型推理没有产生有效输出，脚本终止")
            exit(1)
        
        # 比较输出结果
        print("比较基础模型和微调模型输出...")
        for i, base_output in enumerate(base_outputs):
            if base_output is None:
                continue
                
            sample_id = base_output['sample_id']
            # 找到对应的微调结果
            ft_item = next((item for item in ft_results if item['sample_id'] == sample_id), None)
            
            if ft_item:
                base_html = base_output['base_html']
                ft_html = ft_item['ft_html']
                
                # 比较输出是否相同
                if base_html == ft_html:
                    print(f"警告: 样本 {sample_id} 基础模型和微调模型输出完全相同!")
                else:
                    print(f"样本 {sample_id} 基础模型和微调模型输出不同")
        
        # 创建元数据文件
        metadata = {
            "description": "UI2HTML 推理结果数据",
            "version": "1.0",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("test_results/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 压缩结果目录
        zip_path = "ui2html_results.zip"
        print(f"压缩结果目录到 {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk("test_results"):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, start="."))
        
        # 上传到Hugging Face
        upload_to_huggingface(zip_path)
        
        print(f"结果已保存为: {zip_path}")
        print("推理脚本执行完毕")
    else:
        print("无法加载测试数据集，脚本终止")