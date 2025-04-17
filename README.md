# UI2HTML

## Introduction
UI2HTML leverages the Qwen/Qwen2-VL-7B-Instruct model, combined with LoRA technology, to perform efficient training and inference using 4-bit quantization during Supervised Fine Tuning. By inputting web screenshot (image), the model outputs the corresponding HTML code (text), automating the conversion from UI design to web layout. During the inference phase, the CLIP model is used to evaluate the visual similarity of the generated results, comparing the output quality between base model and fine-tuned model. 🌟

## Table of Contents
- [Quick Start](#quick-start)
- [Usage](#usage)
- [GPU Resource Usage](#gpu-resource-usage)
- [FAQ](#faq)

## Quick Start 🚀

You need to manually create a file named `hf_token.txt` in the root directory of the project. This file should contain your Hugging Face token with write access. Make sure to keep this token secure and do not share it publicly.

### One-Click Installation
```bash
# Add execution permission to the script
chmod +x setup_env.sh

# Run the installation script
./setup_env.sh
```
The script will automatically complete the following operations:
1. Create and configure the conda environment
2. Install required dependencies

## Usage

### Environment Management
```bash
# Activate environment
conda activate ui2html

# Exit environment
conda deactivate

# Remove environment (if needed)
conda remove --name ui2html --all
```

### Running scripts
```bash
python scripts/step1_train.py
python scripts/step2_inference.py
```

Note: The script `scripts/step3_statistics.ipynb` is runned in Colab notebook via this link: [Colab Link (For Visualization)](https://colab.research.google.com/drive/1--a2JUkBlN3Z26g1gaT_EglNLNuZJ6ZY?usp=sharing)

## GPU Resource Usage 💻

### Interactive Usage (Testing/Development)
```bash
srun --gres=gpu:1 --pty --time=02:00:00 --mem=60G bash
srun --partition=debug --gres=gpu:1 --pty --time=00:30:00 --mem=60G bash
```

### Check CUDA Version
```bash
# System CUDA version
nvidia-smi

# PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## Contribution 🤝

We welcome contributions of any kind! Please check our contribution guidelines to learn how to participate in the project.