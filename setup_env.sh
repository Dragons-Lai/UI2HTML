#!/bin/bash

# Load correct modules
module load Miniforge3/24.1.2-0
# module load CUDA/12.4.0

# Initialize conda (Miniforge version)
# source $CONDA_PREFIX/etc/profile.d/conda.sh

# Create conda environment from configuration file
conda env create -f environment.yml

# Activate the environment
conda activate ui2html

# 安裝 huggingface-cli（一定要）
pip install 'huggingface_hub[cli,torch]'

# Install wkhtmltopdf globally if needed
# Note: This might require sudo access, otherwise you'd need a local installation
# sudo apt-get install -y wkhtmltopdf

# # Or download the binary if you don't have admin rights
# if [ ! -f "wkhtmltopdf" ]; then
#     echo "Downloading wkhtmltopdf binary..."
#     wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.centos8.x86_64.rpm
#     # Extract the binary without installing the package
#     rpm2cpio wkhtmltox-0.12.6-1.centos8.x86_64.rpm | cpio -idmv
#     mv ./usr/local/bin/wkhtmltopdf ./
#     chmod +x wkhtmltopdf
#     echo "wkhtmltopdf downloaded and extracted"
# fi

# # Set the path for imgkit to use our local wkhtmltopdf
# export WKHTMLTOPDF=$(pwd)/wkhtmltopdf

# # Log in to Hugging Face
# echo "Please enter your Hugging Face token (Read and Write):"
# read -s HF_TOKEN
# export HF_TOKEN
# huggingface-cli login --token $HF_TOKEN

echo "Environment setup complete!"