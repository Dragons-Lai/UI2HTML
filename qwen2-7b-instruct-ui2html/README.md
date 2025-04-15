---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2-VL-7B-Instruct
tags:
- trl
- sft
- generated_from_trainer
model-index:
- name: qwen2-7b-instruct-ui2html
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2-7b-instruct-ui2html

This model is a fine-tuned version of [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.13.0
- Transformers 4.45.1
- Pytorch 2.6.0+cu124
- Datasets 3.0.1
- Tokenizers 0.20.3