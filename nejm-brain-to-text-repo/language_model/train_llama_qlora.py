"""
QLoRA Fine-tuning for Llama 3.1 8B on Phoneme-to-Text Correction

This script fine-tunes Llama using QLoRA for improved language model rescoring
in the brain-to-text pipeline.

Training data format:
    - Input: Noisy decoded text (from acoustic model)
    - Output: Clean ground truth text

Usage:
    python train_llama_qlora.py --train_data train_pairs.jsonl --output_dir ./llama_adapter

Requirements:
    pip install torch transformers peft bitsandbytes datasets accelerate
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B"
    cache_dir: Optional[str] = None

    # LoRA parameters
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA alpha (scaling)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 256
    weight_decay: float = 0.01

    # Output
    output_dir: str = "./llama_qlora_adapter"
    save_steps: int = 500
    logging_steps: int = 50


def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSONL file.

    Expected format per line:
    {"noisy": "decoded text with errors", "clean": "ground truth text"}

    Args:
        data_path: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    logger.info(f"Loaded {len(data)} training examples")
    return Dataset.from_list(data)


def create_training_data_from_val_metrics(
    val_metrics_path: str,
    phoneme_to_text_map_path: str,
    output_path: str,
):
    """
    Create training data from validation metrics saved during acoustic model training.

    Args:
        val_metrics_path: Path to val_metrics.pkl from Conformer training
        phoneme_to_text_map_path: Path to phoneme-to-character mapping
        output_path: Where to save the JSONL training data
    """
    import pickle

    with open(val_metrics_path, 'rb') as f:
        val_metrics = pickle.load(f)

    # Create pairs of (decoded, ground_truth)
    training_pairs = []

    decoded_seqs = val_metrics.get('decoded_seqs', [])
    true_seqs = val_metrics.get('true_seq', [])
    transcriptions = val_metrics.get('transcription', [])

    for i, (decoded, true, transcription) in enumerate(zip(decoded_seqs, true_seqs, transcriptions)):
        # Convert phoneme sequences to text if needed
        # For now, use the transcription directly as the target
        if isinstance(transcription, (list, tuple)):
            transcription = transcription[0] if len(transcription) > 0 else ""

        if isinstance(decoded, (list, tuple)):
            decoded_text = " ".join(str(d) for d in decoded)
        else:
            decoded_text = str(decoded)

        if len(decoded_text) > 0 and len(str(transcription)) > 0:
            training_pairs.append({
                "noisy": decoded_text,
                "clean": str(transcription)
            })

    # Save to JSONL
    with open(output_path, 'w') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + '\n')

    logger.info(f"Created {len(training_pairs)} training pairs at {output_path}")


def format_training_example(example: Dict, tokenizer) -> str:
    """
    Format a training example as a prompt-completion pair.

    Uses a simple format that teaches the model to correct errors.
    """
    prompt = f"Correct this speech transcription:\n\nInput: {example['noisy']}\n\nOutput: {example['clean']}</s>"
    return prompt


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 256,
) -> Dataset:
    """Preprocess dataset for training."""

    def tokenize_function(examples):
        texts = [
            format_training_example({"noisy": n, "clean": c}, tokenizer)
            for n, c in zip(examples["noisy"], examples["clean"])
        ]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )


def setup_model_and_tokenizer(config: QLoRAConfig):
    """Load and configure model with QLoRA."""

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def train(
    config: QLoRAConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
):
    """Run QLoRA fine-tuning."""

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Preprocess datasets
    logger.info("Preprocessing training data...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, config.max_seq_length)

    if eval_dataset is not None:
        eval_dataset = preprocess_dataset(eval_dataset, tokenizer, config.max_seq_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final adapter
    logger.info(f"Saving adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Llama")

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data JSONL")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to evaluation data JSONL")

    # Model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--cache_dir", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=256)

    # Output
    parser.add_argument("--output_dir", type=str, default="./llama_qlora_adapter")

    args = parser.parse_args()

    # Create config
    config = QLoRAConfig(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )

    # Load data
    train_dataset = load_training_data(args.train_data)
    eval_dataset = load_training_data(args.eval_data) if args.eval_data else None

    # Train
    train(config, train_dataset, eval_dataset)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
