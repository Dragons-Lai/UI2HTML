#!/bin/bash
#SBATCH --job-name=ui2html
#SBATCH --output=ui2html_%j.out
#SBATCH --error=ui2html_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# 加载模块
module load Miniforge3/24.1.2-0

# 激活conda环境
source activate ui2html

# 设置Hugging Face令牌（如果存在）
if [ -f hf_token.txt ]; then
    export HF_TOKEN=$(cat hf_token.txt)
    huggingface-cli login --token $HF_TOKEN
fi

# 创建结果目录
mkdir -p test_results

# 设置Chrome驱动路径（如果需要）
export PATH=$PATH:/path/to/chromedriver

# 运行训练脚本
echo "====== 开始训练 ======"
python scripts/step1_train.py

# 运行推理脚本
echo "====== 开始推理 ======"
python scripts/step2_inference.py

# 运行统计和可视化脚本
echo "====== 开始统计和可视化 ======"
python scripts/step3_statistics.py

echo "====== 任务完成 ======"