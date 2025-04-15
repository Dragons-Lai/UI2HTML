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

# 创建结果目录
mkdir -p test_results

# 运行训练脚本
echo "====== Train ======"
python scripts/step1_train.py

# 运行推理脚本
echo "====== Inference ======"
python scripts/step2_inference.py

# 运行统计和可视化脚本
echo "====== Statistics and Visualization ======"
python scripts/step3_statistics.py

echo "====== Task Completed ======"