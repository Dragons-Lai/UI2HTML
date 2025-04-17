#!/bin/bash
#SBATCH --job-name=ui2html
#SBATCH --output=ui2html_%j.out
#SBATCH --error=ui2html_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load modules
module load Miniforge3/24.1.2-0

# Activate conda environment
source activate ui2html

# Create results directory
mkdir -p test_results

# Run training script
echo "====== Train ======"
python scripts/step1_train.py

# Run inference script
echo "====== Inference ======"
python scripts/step2_inference.py

echo "====== Task Completed ======"