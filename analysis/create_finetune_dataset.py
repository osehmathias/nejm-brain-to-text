#!/usr/bin/env python3
"""
Create OpenAI fine-tuning dataset from phoneme predictions.

Stratifies samples by PER (Phoneme Error Rate) to ensure coverage of
different error levels.

Usage:
    python create_finetune_dataset.py
    python create_finetune_dataset.py --n-samples 150 --input rnn_baseline/phoneme_predictions.csv
"""

import argparse
import csv
import json
import random
import editdistance
from pathlib import Path


SYSTEM_PROMPT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

Output only the decoded sentence, nothing else."""


def calculate_per(prediction: str, ground_truth: str) -> float:
    """Calculate Phoneme Error Rate between prediction and ground truth."""
    pred_phonemes = prediction.split()
    true_phonemes = ground_truth.split()

    if len(true_phonemes) == 0:
        return 1.0 if len(pred_phonemes) > 0 else 0.0

    distance = editdistance.eval(pred_phonemes, true_phonemes)
    return distance / len(true_phonemes)


def load_predictions(csv_path: str, split_filter: str = None) -> list[dict]:
    """Load predictions from CSV file.

    Args:
        csv_path: Path to CSV file
        split_filter: If set, only load samples from this split ('train' or 'val')
    """
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip samples without ground truth text
            if not row['ground_truth_text'].strip():
                continue

            # Filter by split if specified
            if split_filter and row.get('split') != split_filter:
                continue

            per = calculate_per(row['prediction'], row['ground_truth_phonemes'])
            samples.append({
                'sample_id': row['sample_id'],
                'session': row['session'],
                'split': row.get('split', 'unknown'),
                'prediction': row['prediction'],
                'ground_truth_text': row['ground_truth_text'],
                'ground_truth_phonemes': row['ground_truth_phonemes'],
                'per': per,
            })

    return samples


def stratified_sample(samples: list[dict], n_samples: int, seed: int = 42) -> list[dict]:
    """
    Stratified sampling by PER.

    Strata:
        - Low PER: 0-7%
        - Medium PER: 7-12%
        - High PER: 12%+
    """
    random.seed(seed)

    # Define strata
    low = [s for s in samples if s['per'] < 0.07]
    medium = [s for s in samples if 0.07 <= s['per'] < 0.12]
    high = [s for s in samples if s['per'] >= 0.12]

    print(f"Stratum sizes:")
    print(f"  Low PER (0-7%):    {len(low):,} samples")
    print(f"  Medium PER (7-12%): {len(medium):,} samples")
    print(f"  High PER (12%+):   {len(high):,} samples")

    # Sample from each stratum
    n_per_stratum = n_samples // 3
    remainder = n_samples % 3

    selected = []

    # Sample from each stratum
    for i, (stratum, name) in enumerate([(low, 'low'), (medium, 'medium'), (high, 'high')]):
        n = n_per_stratum + (1 if i < remainder else 0)
        if len(stratum) < n:
            print(f"  Warning: {name} stratum has only {len(stratum)} samples, using all")
            selected.extend(stratum)
        else:
            selected.extend(random.sample(stratum, n))

    random.shuffle(selected)
    return selected


def create_finetune_example(sample: dict) -> dict:
    """Create a single fine-tuning example in OpenAI format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample['prediction']},
            {"role": "assistant", "content": sample['ground_truth_text']}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Create OpenAI fine-tuning dataset')
    parser.add_argument('--input', type=str, default='rnn_baseline/phoneme_predictions.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='rnn_baseline/finetune_train.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--n-samples', type=int, default=150,
                        help='Number of samples to include')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-remaining', type=str, default='rnn_baseline/finetune_test.jsonl',
                        help='Output JSONL for remaining samples (test set)')
    parser.add_argument('--train-from-split', type=str, default='train', choices=['train', 'val', 'both'],
                        help='Which split to sample training data from (default: train)')
    parser.add_argument('--test-from-split', type=str, default='val', choices=['train', 'val', 'both'],
                        help='Which split to use for test data (default: val)')
    args = parser.parse_args()

    # Load training candidates
    train_split_filter = None if args.train_from_split == 'both' else args.train_from_split
    test_split_filter = None if args.test_from_split == 'both' else args.test_from_split

    print(f"Loading predictions from {args.input}...")
    print(f"  Training samples from: {args.train_from_split} split")
    print(f"  Test samples from: {args.test_from_split} split")

    train_candidates = load_predictions(args.input, split_filter=train_split_filter)
    print(f"Loaded {len(train_candidates):,} training candidates")

    # Load test candidates (may be different split)
    test_candidates = load_predictions(args.input, split_filter=test_split_filter)
    print(f"Loaded {len(test_candidates):,} test candidates\n")

    # Calculate PER statistics for training candidates
    pers = [s['per'] for s in train_candidates]
    print(f"PER statistics (training candidates):")
    print(f"  Min:    {min(pers)*100:.2f}%")
    print(f"  Max:    {max(pers)*100:.2f}%")
    print(f"  Mean:   {sum(pers)/len(pers)*100:.2f}%")
    print(f"  Median: {sorted(pers)[len(pers)//2]*100:.2f}%\n")

    # Stratified sampling from training candidates
    print(f"Selecting {args.n_samples} stratified samples from {args.train_from_split} split...")
    train_samples = stratified_sample(train_candidates, args.n_samples, args.seed)

    # Test set: all test candidates minus any that ended up in training (if splits overlap)
    train_ids = {s['sample_id'] for s in train_samples}
    test_samples = [s for s in test_candidates if s['sample_id'] not in train_ids]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples):,} samples (from {args.train_from_split})")
    print(f"  Test:  {len(test_samples):,} samples (from {args.test_from_split})")

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write train JSONL
    print(f"\nWriting train set to {args.output}...")
    with open(args.output, 'w') as f:
        for sample in train_samples:
            example = create_finetune_example(sample)
            f.write(json.dumps(example) + '\n')

    # Write test JSONL
    print(f"Writing test set to {args.output_remaining}...")
    with open(args.output_remaining, 'w') as f:
        for sample in test_samples:
            example = create_finetune_example(sample)
            f.write(json.dumps(example) + '\n')

    # Show sample examples
    print(f"\n{'='*60}")
    print("Sample training examples:")
    print(f"{'='*60}")
    for i, sample in enumerate(train_samples[:3]):
        print(f"\nExample {i+1} (PER: {sample['per']*100:.1f}%):")
        print(f"  Input:  {sample['prediction'][:80]}...")
        print(f"  Output: {sample['ground_truth_text']}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Train file: {args.output}")
    print(f"Test file:  {args.output_remaining}")
    print(f"\nTo upload for fine-tuning:")
    print(f"  openai api files.create -f {args.output} -p fine-tune")


if __name__ == '__main__':
    main()
