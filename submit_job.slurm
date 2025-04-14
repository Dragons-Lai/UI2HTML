#!/bin/bash
#SBATCH --job-name=ui2html
#SBATCH --output=ui2html_%j.out
#SBATCH --error=ui2html_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load modules
module load Miniforge3/24.1.2-0

# Activate conda environment
source activate ui2html

# Run the training script
python ui2html_training.py