#!/usr/bin/env python3
"""
Fine-tuning script for HERON Reviewer model.
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# Configuration from environment variables
MODEL_NAME = os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
NEW_MODEL_NAME = os.getenv('OUTPUT_MODEL', 'heron_reviewer_qwen1.5b_gguf')
DATASET_FILE = os.getenv('DATASET_PATH', './heron_reviewer_dataset.jsonl')
MAX_SEQ_LENGTH = int(os.getenv('MAX_SEQ_LENGTH', '2048'))


def load_base_model():
    """Load and configure the base model with 4-bit quantization."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def configure_lora(model):
    """Configure LoRA adapters for efficient fine-tuning."""
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def setup_tokenizer(tokenizer):
    """Configure chat template for the tokenizer."""
    return get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
        mapping={
            "role": "role",
            "content": "content",
            "user": "user",
            "assistant": "assistant"
        },
    )


def formatting_prompts_func(examples, tokenizer):
    """Format examples for training."""
    convos = examples["messages"]
    texts = []
    for convo in convos:
        if isinstance(convo, list) and len(convo) > 0:
            try:
                text = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            except Exception:
                texts.append("")
        else:
            texts.append("")
    return texts


def load_and_clean_dataset(dataset_file):
    """Load and filter malformed dataset entries."""
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    original_size = len(dataset)

    dataset = dataset.filter(
        lambda x: isinstance(x["messages"], list) and len(x["messages"]) > 0
    )

    print(f"Dataset: {original_size} -> {len(dataset)} entries (after filtering)")
    return dataset


def create_trainer(model, tokenizer, dataset):
    """Create SFT trainer with configuration."""

    def format_func(examples):
        return formatting_prompts_func(examples, tokenizer)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_func,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=600,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="checkpoints",
        ),
    )


def export_gguf(model, tokenizer, model_name):
    """Export model to GGUF format for inference."""
    try:
        model.save_pretrained_gguf(
            model_name, tokenizer, quantization_method="q4_k_m"
        )
        print(f"Model exported to: {os.path.abspath(model_name)}")
    except Exception as e:
        print(f"GGUF export failed: {e}")
        print("Falling back to LoRA format...")
        model.save_pretrained(f"{model_name}_lora")
        tokenizer.save_pretrained(f"{model_name}_lora")


def main():
    """Main training pipeline."""
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = load_base_model()

    print("Configuring LoRA adapters...")
    model = configure_lora(model)

    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer(tokenizer)

    print(f"Loading dataset: {DATASET_FILE}")
    dataset = load_and_clean_dataset(DATASET_FILE)

    print("Starting training...")
    trainer = create_trainer(model, tokenizer, dataset)
    trainer.train()

    print("Exporting to GGUF format...")
    export_gguf(model, tokenizer, NEW_MODEL_NAME)

    print("Training complete!")


if __name__ == "__main__":
    main()
