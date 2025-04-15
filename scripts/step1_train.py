#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI2HTML - 训练脚本
用于微调Qwen2-VL-7B-Instruct模型将UI截图转换为HTML布局
"""

import json
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# 设置随机种子以保证可重复性
seed = 459
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 加载HuggingFace令牌
def load_hf_token():
    try:
        with open("hf_token.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("未找到hf_token.txt文件，请确保该文件存在并包含有效的HuggingFace令牌")
        return None

# 登录HuggingFace
token = load_hf_token()
if token:
    login(token=token)
else:
    print("警告：未设置HuggingFace令牌，可能无法访问模型或推送结果")

# 定义系统消息和提示
prompt = """Convert the provided UI screenshot into clean, semantic HTML code with inline CSS.
The code should be responsive and follow best practices for web development.
Do not include any JavaScript.

Please provide only the HTML code."""

system_message = "You are an expert UI developer specializing in converting design mockups into clean HTML code."

# 加载和预处理数据集
def load_and_preprocess_dataset():
    print("开始加载和预处理数据集...")
    # 加载Design2Code数据集
    dataset_id = "SALT-NLP/Design2Code-hf"
    full_dataset = load_dataset(dataset_id, split="train")
    
    # 定义HTML长度阈值
    MAX_HTML_LENGTH = 12000
    
    # 按HTML长度过滤函数
    def filter_by_html_length(example):
        return len(example["text"]) <= MAX_HTML_LENGTH
    
    # 过滤数据集
    filtered_indices = []
    for i, example in enumerate(full_dataset):
        if filter_by_html_length(example):
            filtered_indices.append(i)
    
    # 创建过滤后的数据集
    filtered_dataset = full_dataset.select(filtered_indices)
    print(f"原始数据集大小: {len(full_dataset)}")
    print(f"过滤后数据集大小: {len(filtered_dataset)}")
    print(f"移除了 {len(full_dataset) - len(filtered_dataset)} 个HTML过长的示例")
    
    # 将数据集分为训练集(80%)和测试集(20%)
    all_indices = list(range(len(filtered_dataset)))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=seed)
    
    # 创建训练集和测试集
    train_dataset_raw = filtered_dataset.select(train_indices)
    test_dataset_raw = filtered_dataset.select(test_indices)
    
    print(f"训练集大小: {len(train_dataset_raw)}")
    print(f"测试集大小: {len(test_dataset_raw)}")
    
    # 将结果保存为JSON以便后续使用
    test_indices_file = "test_indices.json"
    with open(test_indices_file, "w") as f:
        json.dump({"test_indices": test_indices}, f)
    
    return train_dataset_raw, test_dataset_raw

# 将数据转换为OAI消息格式
def format_data(sample):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },{
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ],
        }

# 创建数据整理器
def create_collate_fn(processor):
    def collate_fn(examples):
        # 获取文本和图像，并应用聊天模板
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
        
        # 标记化文本并处理图像
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        # 标签是input_ids，在损失计算中屏蔽填充标记
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # 在损失计算中忽略图像标记索引（特定于模型）
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
            
        batch["labels"] = labels
        return batch
    
    return collate_fn

# 训练模型
def train_model(train_dataset):
    print("开始训练模型...")
    # Hugging Face模型ID
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    # BitsAndBytesConfig int-4配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型和处理器
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # LoRA配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    # SFT配置
    args = SFTConfig(
        output_dir="qwen2-7b-instruct-ui2html",  # 保存目录和仓库ID
        num_train_epochs=1,                      # 训练轮数
        per_device_train_batch_size=1,           # 每设备训练批次大小
        gradient_accumulation_steps=1,           # 梯度累积步数
        gradient_checkpointing=True,             # 使用梯度检查点以节省内存
        optim="adamw_torch_fused",               # 使用融合的adamw优化器
        logging_steps=1,                          # 每1步记录一次日志
        save_strategy="epoch",                    # 每个轮次保存一次检查点
        learning_rate=2e-4,                      # 学习率
        bf16=True,                               # 使用bfloat16精度
        tf32=True,                               # 使用tf32精度
        max_grad_norm=0.3,                       # 最大梯度范数
        warmup_ratio=0.03,                       # 预热比例
        lr_scheduler_type="constant",            # 使用常量学习率调度器
        push_to_hub=True,                        # 推送模型到hub
        report_to="tensorboard",                 # 将指标报告给tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 使用非重入检查点
        dataset_text_field="",                   # 需要一个虚拟字段用于整理器
        dataset_kwargs={"skip_prepare_dataset": True},  # 对整理器很重要
    )
    args.remove_unused_columns = False
    
    # 将数据集转换为OAI消息格式
    train_dataset_formatted = [format_data(sample) for sample in train_dataset]
    
    # 创建整理函数
    collate_fn = create_collate_fn(processor)
    
    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset_formatted,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model(args.output_dir)
    print(f"训练完成！模型已保存到 {args.output_dir}")

# 主函数
if __name__ == "__main__":
    print("开始UI2HTML训练脚本...")
    train_dataset, test_dataset = load_and_preprocess_dataset()
    train_model(train_dataset)
    print("训练脚本执行完毕")