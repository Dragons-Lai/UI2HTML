#!/bin/bash

# Load correct modules
module load Miniforge3/24.1.2-0
# module load CUDA/12.4.0

# Create conda environment from configuration file
conda env create -f environment.yml

# Activate the environment
conda activate ui2html

# Install huggingface-cli
pip install 'huggingface_hub[cli,torch]'

echo "Environment setup complete!"