# UI2HTML

UI2HTML 是一个用于将用户界面转换为 HTML 的项目。

## 目录
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [GPU 资源使用](#gpu-资源使用)
- [常见问题](#常见问题)

## 快速开始

### 一键安装（推荐）
```bash
# 给脚本添加执行权限
chmod +x setup_env.sh

# 运行安装脚本
./setup_env.sh
```
脚本会自动完成以下操作：
1. 创建并配置 conda 环境
2. 安装所需依赖
<!-- 3. 配置 Hugging Face 登录 -->

### 手动安装（如需要）
如果自动安装出现问题，您也可以按照以下步骤手动安装：

1. 创建并激活环境
```bash
conda env create -f environment.yml
conda activate ui2html
```

2. 安装额外依赖
```bash
pip install 'huggingface_hub[cli,torch]'
```

3. 登录 Hugging Face
```bash
huggingface-cli login
```

## 使用方法

### 环境管理
```bash
# 激活环境
conda activate ui2html

# 退出环境
conda deactivate

# 移除环境（如需要）
conda remove --name ui2html --all
```

### 运行训练
```bash
# 使用 SLURM 提交作业
sbatch submit_job.slurm

# 或直接运行
python ui2html_training.py
```

## GPU 资源使用

### 交互式使用（测试/开发）
```bash
srun --gres=gpu:1 --pty --time=02:00:00 --mem=60G bash
```

### 检查 CUDA 版本
```bash
# 系统 CUDA 版本
nvidia-smi

# PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```