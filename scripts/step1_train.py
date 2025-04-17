#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI2HTML - Training Script
Used for fine-tuning the Qwen2-VL-7B-Instruct model to convert UI screenshots to HTML layouts
"""

import json
import random
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# Set random seed to ensure reproducibility
seed = 459
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load HuggingFace token
def load_hf_token():
    try:
        with open("hf_token.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("hf_token.txt file not found, please ensure this file exists and contains a valid HuggingFace token")
        return None

# Login to HuggingFace
token = load_hf_token()
if token:
    login(token=token)
else:
    print("Warning: HuggingFace token not set, may not be able to access models or push results")

# Define system message and prompt
prompt = """Convert the provided UI screenshot into clean, semantic HTML code with inline CSS.
The code should be responsive and follow best practices for web development.
Do not include any JavaScript.

Please provide only the HTML code."""

system_message = "You are an expert UI developer specializing in converting design mockups into clean HTML code."

# Load and preprocess dataset
def load_and_preprocess_dataset():
    print("Starting to load and preprocess dataset...")
    # Load webcode2m_purified dataset in streaming mode and limit the quantity
    dataset_id = "xcodemind/webcode2m_purified"
    MAX_SAMPLES = 100
    
    print(f"Loading dataset '{dataset_id}' in streaming mode and limiting to max {MAX_SAMPLES} samples...")
    stream_dataset = load_dataset(dataset_id, split="train", streaming=True)
    
    # Define HTML length threshold and filtering conditions
    MAX_HTML_LENGTH = 5000
    MAX_TOKEN_COUNT = 1024
    
    # Filter function by HTML length, language and total token count
    def filter_dataset(example):
        # Check HTML length
        html_length_ok = len(example["text"]) <= MAX_HTML_LENGTH
        
        # Check if language is English
        lang_ok = example.get("lang") == "en"
        
        # Check token count
        tokens = example.get("tokens", [0, 0])
        if isinstance(tokens, list) and len(tokens) == 2:
            total_tokens = tokens[0] + tokens[1]  # CSS length + HTML length
            tokens_ok = total_tokens <= MAX_TOKEN_COUNT
        else:
            tokens_ok = False
            
        return html_length_ok and lang_ok and tokens_ok
    
    # Filter and collect data
    filtered_examples = []
    sample_count = 0
    total_checked = 0
    
    # Record the index of each filtered sample in the original dataset
    original_indices = []
    
    print("Starting to filter data...")
    for example in stream_dataset:
        total_checked += 1
        if filter_dataset(example):
            filtered_examples.append(example)
            original_indices.append(total_checked - 1)  # Record index in original dataset
            sample_count += 1
            
        if sample_count >= MAX_SAMPLES:
            break
        
        if total_checked % 1000 == 0:
            print(f"Checked {total_checked} samples, collected {sample_count} samples that meet the criteria")
    
    print(f"Checked a total of {total_checked} samples")
    print(f"Collected {sample_count} samples that meet the criteria")
    
    # Convert to regular dataset
    from datasets import Dataset
    filtered_dataset = Dataset.from_list(filtered_examples)
    
    # Split the dataset into training (80%) and test (20%) sets
    all_indices = list(range(len(filtered_dataset)))
    train_indices_local, test_indices_local = train_test_split(all_indices, test_size=0.2, random_state=seed)
    
    # Map local indices back to original dataset indices
    test_indices = [original_indices[i] for i in test_indices_local]
    
    # Create training and test datasets
    train_dataset_raw = filtered_dataset.select(train_indices_local)
    test_dataset_raw = filtered_dataset.select(test_indices_local)
    
    print(f"Training set size: {len(train_dataset_raw)}")
    print(f"Test set size: {len(test_dataset_raw)}")
    
    with open("test_indices.json", "w") as f:
        json.dump({"test_indices": test_indices}, f)
    
    print(f"Training and test indices saved to test_indices.json")
    
    # Save the filtered complete dataset (optional)
    # filtered_dataset.save_to_disk("filtered_dataset")
    
    return train_dataset_raw, test_dataset_raw

# Convert data to OAI message format
def format_data(sample):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },{
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ],
        }

# Create data collator
def create_collate_fn(processor):
    def collate_fn(examples):
        # Get text and images, and apply chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
        
        # Tokenize text and process images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        # Labels are input_ids, mask padding tokens in loss calculation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Ignore image token indices in loss calculation (model-specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
            
        batch["labels"] = labels
        return batch
    
    return collate_fn

# Train model
def train_model(train_dataset):
    print("Starting model training...")
    # Clear CUDA cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Hugging Face model ID
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    # BitsAndBytesConfig int-4 configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    # SFT configuration
    args = SFTConfig(
        output_dir="qwen2-7b-instruct-ui2html",  # Save directory and repository ID
        num_train_epochs=3,                      # Control epochs
        per_device_train_batch_size=1,           # Batch size per device for training
        gradient_accumulation_steps=4,           # Gradient accumulation steps
        gradient_checkpointing=True,             # Use gradient checkpointing to save memory
        optim="adamw_torch_fused",               # Use fused adamw optimizer
        logging_steps=1,                         # Log every 1 step
        save_strategy="epoch",                   # Save checkpoint every epoch
        learning_rate=2e-5,                      # Learning rate
        bf16=True,                               # Use bfloat16 precision
        tf32=True,                               # Use tf32 precision
        max_grad_norm=0.3,                       # Maximum gradient norm
        warmup_ratio=0.05,                       # Warmup ratio
        lr_scheduler_type="constant",            # Use constant learning rate scheduler
        push_to_hub=True,                        # Push model to hub
        report_to="tensorboard",                 # Report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant checkpointing
        dataset_text_field="",                   # Need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # Important for collator
    )
    args.remove_unused_columns = False
    
    # Convert dataset to OAI message format
    train_dataset_formatted = [format_data(sample) for sample in train_dataset]
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset_formatted,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model(args.output_dir)
    print(f"Training complete! Model saved to {args.output_dir}")

# Main function
if __name__ == "__main__":
    print("Starting UI2HTML training script...")
    train_dataset, test_dataset = load_and_preprocess_dataset()
    train_model(train_dataset)
    print("Training script execution completed")