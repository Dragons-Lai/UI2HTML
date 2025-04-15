#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI2HTML - 统计和可视化脚本
用于渲染HTML为图像，计算相似度，并生成统计数据和可视化
"""

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 渲染HTML为图像
def render_html_to_image(html_str, output_path):
    """使用Selenium将HTML字符串渲染为图像"""
    html_file = f"test_results/temp_{int(time.time() * 1000)}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_str)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1280x1024')
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.get("file://" + os.path.abspath(html_file))
        time.sleep(1)  # 等待渲染
        driver.save_screenshot(output_path)
        driver.quit()
    except Exception as e:
        print(f"渲染HTML时出错: {e}")
        return None

    try:
        os.remove(html_file)  # 清理临时文件
    except:
        pass

    return output_path

# 加载CLIP模型进行相似度计算
def load_clip_model():
    print("加载CLIP模型用于图像相似度计算...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

# 获取图像嵌入
def get_image_embedding(image_path, clip_model, clip_processor):
    """获取图像的CLIP嵌入"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embeds = clip_model.get_image_features(**inputs)
        return F.normalize(image_embeds, p=2, dim=-1)
    except Exception as e:
        print(f"获取图像 {image_path} 的嵌入时出错: {e}")
        return None

# 计算余弦相似度
def cosine_similarity(a, b):
    """计算两个嵌入之间的余弦相似度"""
    if a is None or b is None:
        return 0.0
    return (a @ b.T).item()

# 计算相似度指标
def calculate_similarity_metrics():
    print("计算相似度指标...")
    
    # 确保渲染目录存在
    os.makedirs("test_results/rendered", exist_ok=True)
    
    # 加载base_outputs
    try:
        with open("test_results/base_outputs.json", "r") as f:
            base_data = json.load(f)
    except FileNotFoundError:
        print("未找到base_outputs.json。请先运行推理脚本。")
        return None
        
    # 加载ft_results
    try:
        with open("test_results/ft_results.json", "r") as f:
            ft_data = json.load(f)
    except FileNotFoundError:
        print("未找到ft_results.json。请先运行推理脚本。")
        return None
    
    # 将ft_data转换为字典以便快速查找
    ft_dict = {item['sample_id']: item for item in ft_data}
    
    # 加载CLIP模型
    clip_model, clip_processor = load_clip_model()
    
    # 准备结果字典
    results = {
        'sample_id': [],
        'original_path': [],
        'base_rendered_path': [],
        'ft_rendered_path': [],
        'base_similarity': [],
        'ft_similarity': [],
        'improvement': []
    }
    
    # 处理每个样本
    for item in tqdm(base_data, desc="处理样本"):
        sample_id = item['sample_id']
        
        # 检查是否有对应的微调结果
        if sample_id not in ft_dict:
            print(f"样本 {sample_id} 没有对应的微调结果，跳过")
            continue
            
        original_path = item['original_path']
        base_html = item['base_html']
        ft_html = ft_dict[sample_id]['ft_html']
        
        # 渲染HTML为图像
        base_rendered_path = render_html_to_image(
            base_html,
            f"test_results/rendered/{sample_id}_base.png"
        )
        
        ft_rendered_path = render_html_to_image(
            ft_html,
            f"test_results/rendered/{sample_id}_ft.png"
        )
        
        if base_rendered_path is None or ft_rendered_path is None:
            print(f"样本 {sample_id} 的渲染失败，跳过")
            continue
            
        # 获取嵌入
        e_orig = get_image_embedding(original_path, clip_model, clip_processor)
        e_base = get_image_embedding(base_rendered_path, clip_model, clip_processor)
        e_ft = get_image_embedding(ft_rendered_path, clip_model, clip_processor)
        
        # 计算相似度
        base_similarity = cosine_similarity(e_orig, e_base)
        ft_similarity = cosine_similarity(e_orig, e_ft)
        improvement = ft_similarity - base_similarity
        
        # 存储结果
        results['sample_id'].append(sample_id)
        results['original_path'].append(original_path)
        results['base_rendered_path'].append(base_rendered_path)
        results['ft_rendered_path'].append(ft_rendered_path)
        results['base_similarity'].append(base_similarity)
        results['ft_similarity'].append(ft_similarity)
        results['improvement'].append(improvement)
        
        print(f"  基础模型相似度: {base_similarity:.4f}")
        print(f"  微调模型相似度: {ft_similarity:.4f}")
        print(f"  改进: {improvement:.4f}")
    
    return results

# 生成统计和可视化
def generate_stats_and_visualizations(results):
    print("生成统计和可视化...")
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 保存结果到CSV
    results_df.to_csv("test_results/similarity_metrics.csv", index=False)
    
    # 计算平均指标
    avg_base_similarity = results_df['base_similarity'].mean()
    avg_ft_similarity = results_df['ft_similarity'].mean()
    avg_improvement = results_df['improvement'].mean()
    median_improvement = results_df['improvement'].median()
    improvement_percent = (results_df['improvement'] > 0).mean() * 100
    
    # 生成摘要统计
    print("\n=== 统计摘要 ===")
    print(f"测试样本数量: {len(results_df)}")
    print(f"基础模型平均相似度: {avg_base_similarity:.4f}")
    print(f"微调模型平均相似度: {avg_ft_similarity:.4f}")
    print(f"平均改进: {avg_improvement:.4f}")
    print(f"中位数改进: {median_improvement:.4f}")
    print(f"改进样本百分比: {improvement_percent:.2f}%")
    
    # 保存摘要到文件
    with open("test_results/summary_stats.txt", "w") as f:
        f.write("=== 统计摘要 ===\n")
        f.write(f"测试样本数量: {len(results_df)}\n")
        f.write(f"基础模型平均相似度: {avg_base_similarity:.4f}\n")
        f.write(f"微调模型平均相似度: {avg_ft_similarity:.4f}\n")
        f.write(f"平均改进: {avg_improvement:.4f}\n")
        f.write(f"中位数改进: {median_improvement:.4f}\n")
        f.write(f"改进样本百分比: {improvement_percent:.2f}%\n")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['base_similarity'], results_df['ft_similarity'])
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlabel('基础模型相似度')
    plt.ylabel('微调模型相似度')
    plt.title('基础模型与微调模型性能对比')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("test_results/similarity_comparison.png")
    
    # 绘制改进直方图
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['improvement'], bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('改进 (微调模型 - 基础模型)')
    plt.ylabel('频率')
    plt.title('改进直方图')
    plt.grid(True)
    plt.savefig("test_results/improvement_histogram.png")
    
    print("\n测试结果已保存到 test_results/")
    print("视觉比较已保存到 test_results/rendered/")
    print("指标已保存到 test_results/similarity_metrics.csv")
    print("图表已保存到 test_results/")

# 主函数
if __name__ == "__main__":
    print("开始UI2HTML统计和可视化脚本...")
    
    # 确保结果目录存在
    os.makedirs("test_results", exist_ok=True)
    
    # 计算相似度指标
    results = calculate_similarity_metrics()
    
    if results is not None:
        # 生成统计和可视化
        generate_stats_and_visualizations(results)
        print("统计和可视化脚本执行完毕")
    else:
        print("无法计算相似度指标，脚本终止")