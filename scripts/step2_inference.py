#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI2HTML - Inference Script
Used for inference with base model and fine-tuned model to generate HTML
"""

import os
import torch
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm
import time
import zipfile
from huggingface_hub import HfApi, login
from peft import PeftModel

# System message and prompt
prompt = """Convert the provided UI screenshot into clean, semantic HTML code with inline CSS.
The code should be responsive and follow best practices for web development.
Do not include any JavaScript.

Please provide only the HTML code."""

system_message = "You are an expert UI developer specializing in converting design mockups into clean HTML code."

# Load base model and processor
def load_models():
    print("Loading models...")
    # Hugging Face model ID
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_path = "./qwen2-7b-instruct-ui2html"
    
    # Load base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loading complete")
    
    return base_model, processor, adapter_path

# Generate HTML from image
def generate_html(image, model, processor):
    # Print current model type
    print(f"Model type used for HTML generation: {type(model)}")
    print(f"Model has PEFT configuration: {hasattr(model, 'peft_config')}")
    
    # Create messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]},
    ]

    # Prepare inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Print input information
    print(f"Inference input shape: {inputs.input_ids.shape}")
    
    # Generate output
    max_tokens = 1024

    print("Starting generation...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    print("Generation complete")

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Extract HTML content
def extract_html(text):
    """Extract HTML content from text"""
    import re

    # Find HTML content with different patterns
    patterns = [
        r'<html.*?>.*?</html>',  # Standard HTML
        r'<!DOCTYPE.*?>.*?</html>',  # DOCTYPE declaration
        r'<[^>]+>.*</[^>]+>'  # Any HTML-like tags
    ]

    # Try each pattern until a match is found
    html_content = None
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            html_content = match.group(0)
            break

    if not html_content:
        return "No HTML content found."

    return html_content

# Load test dataset
def load_test_dataset():
    print("Loading test dataset...")
    # Use the same dataset as training (webcode2m_purified)
    dataset_id = "xcodemind/webcode2m_purified"
    
    # Check if test indices file exists
    test_indices_file = "test_indices.json"
    
    try:
        with open(test_indices_file, "r") as f:
            test_indices = json.load(f)["test_indices"]
        
        print(f"Successfully loaded test indices, total of {len(test_indices)} samples")
    except Exception as e:
        print(f"Error loading test indices: {e}")
        return None
    
    try:
        # Load dataset in streaming mode
        print(f"Loading dataset '{dataset_id}' in streaming mode...")
        stream_dataset = load_dataset(dataset_id, split="train", streaming=True)
        
        # Create a test dataset
        from datasets import Dataset
        test_samples = []
        
        # Record the number of processed samples and found test samples
        processed_count = 0
        found_count = 0
        
        # For streaming dataset, we need to iterate and find samples matching test_indices
        print("Filtering test samples...")
        
        # Convert test_indices to a set for faster lookup
        test_indices_set = set(test_indices)
        
        # Iterate through streaming dataset
        for example in stream_dataset:
            if processed_count in test_indices_set:
                test_samples.append(example)
                found_count += 1
                
                # If all test samples are found, we can end early
                if found_count == len(test_indices):
                    break
                
                # Print progress every 10 samples found
                if found_count % 10 == 0:
                    print(f"Found {found_count}/{len(test_indices)} test samples")
            
            processed_count += 1
            
            # Print progress every 1000 samples processed
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} samples, found {found_count}/{len(test_indices)} test samples")
        
        # Convert collected samples to regular dataset
        test_dataset = Dataset.from_list(test_samples)
        print(f"Test set size: {len(test_dataset)} samples")
        
        return test_dataset
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return None

# Load HuggingFace token
def load_hf_token():
    try:
        with open("hf_token.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("hf_token.txt file not found, please ensure this file exists and contains a valid HuggingFace token")
        return None

# Run base model inference
def run_base_model_inference(model, processor, test_dataset):
    print("Running inference with base model...")
    print(f"Base model type: {type(model)}")
    print(f"Base model has adapter: {hasattr(model, 'peft_config')}")
    
    # Ensure model is in evaluation mode
    model.eval()
    print(f"Model evaluation mode: {model.training == False}")
    
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/original", exist_ok=True)
    os.makedirs("test_results/base_model", exist_ok=True)
    
    # Process test subset
    max_test_samples = min(len(test_dataset), 500)  # Adjust based on resources
    test_subset = test_dataset.select(range(max_test_samples))
    
    base_outputs = []
    
    for i, test_sample in enumerate(tqdm(test_subset, desc="Base model inference")):
        sample_id = f"sample_{i}"
        
        try:
            # Save original UI image
            original_image = test_sample["image"]
            original_path = f"test_results/original/{sample_id}.png"
            
            # If it's a PIL image, save directly
            if hasattr(original_image, 'save'):
                original_image.save(original_path)
            
            # Generate HTML using base model
            print(f"\nProcessing sample {i} generating HTML with base model")
            base_html_raw = generate_html(original_image, model, processor)
            base_html = extract_html(base_html_raw)
            
            # Save base model HTML
            with open(f"test_results/base_model/{sample_id}.html", "w", encoding="utf-8") as f:
                f.write(base_html)
            
            # Store output for later comparison
            base_outputs.append({
                'sample_id': sample_id,
                'original_image': original_image,
                'original_path': original_path,
                'base_html': base_html
            })
            
        except Exception as e:
            print(f"Error with base model on sample {i}: {e}")
            base_outputs.append(None)
    
    # Save base_outputs for later use
    with open("test_results/base_outputs.json", "w") as f:
        # Handle non-serializable content
        serializable_outputs = []
        for output in base_outputs:
            if output is not None:
                serializable_outputs.append({
                    'sample_id': output['sample_id'],
                    'original_path': output['original_path'],
                    'base_html': output['base_html']
                })
        json.dump(serializable_outputs, f)
    
    return base_outputs

# Run fine-tuned model inference
def run_finetuned_model_inference(model, processor, adapter_path, base_outputs):
    print("Loading adapter to create fine-tuned model...")
    
    # Check if adapter path exists
    print(f"Adapter path: {adapter_path}")
    if not os.path.exists(adapter_path):
        print(f"Warning: Adapter path {adapter_path} does not exist!")
    else:
        print(f"Adapter path exists!")

    # Print parameter count and model state before loading
    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Base model trainable parameter count: {base_params}")
    print(f"Model type before adapter loading: {type(model)}")
    print(f"Base model already has adapter: {hasattr(model, 'peft_config')}")
    
    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Successfully loaded adapter using PeftModel.from_pretrained")

    # Print parameter count after loading
    ft_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter count after loading adapter: {ft_params}")
    
    # Print model state information, confirming adapter was loaded correctly
    print(f"Adapter is loaded: {hasattr(model, 'peft_config')}")
    if hasattr(model, 'peft_config'):
        print(f"PEFT configuration: {model.peft_config}")
        # Fix configuration access method
        if hasattr(model.peft_config, 'target_modules'):
            print(f"target_modules in PEFT configuration: {model.peft_config.target_modules}")
        
        try:
            if hasattr(model, 'state_dict'):
                print(f"Adapter module info: {list(model.state_dict().keys())[:10]} ...")
                # Check for LoRA weights
                lora_weights = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
                print(f"Number of LoRA weights found: {len(lora_weights)}")
                if lora_weights:
                    print(f"LoRA weight examples: {lora_weights[:5]}")
        except Exception as e:
            print(f"Error getting model state dict: {e}")
    
    # Ensure model is in evaluation mode
    model.eval()
    print(f"Model evaluation mode: {model.training == False}")
    
    os.makedirs("test_results/fine_tuned_model", exist_ok=True)
    
    ft_results = []
    
    for i, data in enumerate(tqdm(base_outputs, desc="Fine-tuned model inference")):
        if data is None:
            continue
        
        sample_id = data['sample_id']
        
        try:
            # Generate HTML using fine-tuned model
            ft_html_raw = generate_html(data['original_image'], model, processor)
            ft_html = extract_html(ft_html_raw)
            
            # Save fine-tuned model HTML
            with open(f"test_results/fine_tuned_model/{sample_id}.html", "w", encoding="utf-8") as f:
                f.write(ft_html)
            
            # Save results
            ft_results.append({
                'sample_id': sample_id,
                'ft_html': ft_html
            })
            
        except Exception as e:
            print(f"Error with fine-tuned model on sample {i}: {e}")
            continue
    
    # Save ft_results for later use
    with open("test_results/ft_results.json", "w") as f:
        json.dump(ft_results, f)
    
    return ft_results

# Upload results to Hugging Face
def upload_to_huggingface(zip_path, hf_repo_id=None):
    """Upload results to Hugging Face Hub"""
    try:
        # Initialize API
        api = HfApi()
        
        # Get current logged-in user information
        try:
            current_user = api.whoami()
            username = current_user['name']
            repo_name = "ui2html_results"
            hf_repo_id = f"{username}/{repo_name}"
            print(f"Using current logged-in user '{username}' to upload to repository: {hf_repo_id}")
        except Exception as e:
            print(f"Failed to get user information: {e}")
            print("Please ensure you are logged in via huggingface-cli login")
            return False
        
        print(f"Uploading to Hugging Face: {hf_repo_id}")
        
        # Create a simple README
        readme_content = f"""# UI2HTML Inference Results Data

This repository contains inference results data for the UI2HTML project, for statistical and visual analysis.

## Data Information
- Creation time: {time.strftime("%Y-%m-%d %H:%M:%S")}
- Contents: Original images, base model HTML, fine-tuned model HTML, result data

## Usage Instructions
1. Download `ui2html_results.zip`
2. Extract to your working directory
3. Use statistical scripts for analysis
"""
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Check if repository exists
        try:
            api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
            print(f"Repository {hf_repo_id} already exists")
        except Exception:
            print(f"Creating new repository {hf_repo_id}")
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=False)
        
        # Upload files
        print(f"Uploading {zip_path}...")
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=os.path.basename(zip_path),
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        
        # Upload README
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        
        print(f"Upload successful! Visit: https://huggingface.co/datasets/{hf_repo_id}")
        return True
    except Exception as e:
        print(f"Error during upload process: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists("README.md"):
            os.remove("README.md")

# Main function
if __name__ == "__main__":
    print("Starting UI2HTML inference script...")
    
    # Login to HuggingFace
    token = load_hf_token()
    if token:
        login(token=token)
        print("Successfully logged into HuggingFace")
    else:
        print("Warning: HuggingFace token not set, may not be able to access models or push results")
    
    # Ensure results directories exist
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/original", exist_ok=True)
    os.makedirs("test_results/base_model", exist_ok=True)
    os.makedirs("test_results/fine_tuned_model", exist_ok=True)
    
    # Load models and test dataset
    base_model, processor, adapter_path = load_models()
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    if test_dataset is not None:
        print(f"Successfully loaded test dataset with {len(test_dataset)} samples")
        
        # Run base model inference
        base_outputs = run_base_model_inference(base_model, processor, test_dataset)
        
        if not base_outputs:
            print("Base model inference did not produce valid output, script terminating")
            exit(1)
        
        # Run fine-tuned model inference
        ft_results = run_finetuned_model_inference(base_model, processor, adapter_path, base_outputs)
        
        if not ft_results:
            print("Fine-tuned model inference did not produce valid output, script terminating")
            exit(1)
        
        # Compare output results
        print("Comparing base model and fine-tuned model outputs...")
        for i, base_output in enumerate(base_outputs):
            if base_output is None:
                continue
                
            sample_id = base_output['sample_id']
            # Find corresponding fine-tuned result
            ft_item = next((item for item in ft_results if item['sample_id'] == sample_id), None)
            
            if ft_item:
                base_html = base_output['base_html']
                ft_html = ft_item['ft_html']
                
                # Compare if outputs are the same
                if base_html == ft_html:
                    print(f"Warning: Sample {sample_id} base model and fine-tuned model outputs are identical!")
                else:
                    print(f"Sample {sample_id} base model and fine-tuned model outputs differ")
        
        # Create metadata file
        metadata = {
            "description": "UI2HTML Inference Results Data",
            "version": "1.0",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("test_results/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Compress results directory
        zip_path = "ui2html_results.zip"
        print(f"Compressing results directory to {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk("test_results"):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, start="."))
        
        # Upload to Hugging Face
        upload_to_huggingface(zip_path)
        
        print(f"Results saved as: {zip_path}")
        print("Inference script execution completed")
    else:
        print("Unable to load test dataset, script terminating")