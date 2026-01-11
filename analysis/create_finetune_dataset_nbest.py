#!/usr/bin/env python3
"""
Create OpenAI fine-tuning dataset from N-best beam search phoneme predictions.

Trains the LLM to select the best hypothesis or synthesize the correct output
from multiple candidate phoneme sequences.

Usage:
    python create_finetune_dataset_nbest.py --input rnn_baseline/phoneme_predictions_beam.csv
    python create_finetune_dataset_nbest.py --input rnn_baseline/phoneme_predictions_beam.csv --n-samples 150
"""

import argparse
import csv
import json
import random
import editdistance
from pathlib import Path


SYSTEM_PROMPT_NBEST = """You are a phoneme-to-text decoder. You are given multiple candidate phoneme sequences from a speech decoder, ranked by acoustic score.

Your task: Output the most likely English sentence based on all candidates.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

The candidates may contain errors. Use linguistic knowledge to determine the correct sentence. Output only the decoded sentence, nothing else."""


SYSTEM_PROMPT_NBEST_SCORED = """You are a phoneme-to-text decoder. You are given multiple candidate phoneme sequences from a speech decoder, each with a log-probability score (higher/less negative = more likely according to acoustics).

Your task: Consider all candidates and output the most likely English sentence.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

The acoustic scores indicate decoder confidence, but the highest-scored candidate may still contain errors. Use linguistic knowledge to determine the correct sentence. Output only the decoded sentence, nothing else."""


def calculate_per(prediction: str, ground_truth: str) -> float:
    """Calculate Phoneme Error Rate between prediction and ground truth."""
    pred_phonemes = prediction.split()
    true_phonemes = ground_truth.split()

    if len(true_phonemes) == 0:
        return 1.0 if len(pred_phonemes) > 0 else 0.0

    distance = editdistance.eval(pred_phonemes, true_phonemes)
    return distance / len(true_phonemes)


def load_beam_predictions(csv_path: str, split_filter: str = None) -> list[dict]:
    """Load N-best predictions from beam search CSV file."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Detect number of hypotheses from column names
        n_hyps = sum(1 for name in fieldnames if name.startswith('hypothesis_'))

        for row in reader:
            # Skip samples without ground truth text
            if not row.get('ground_truth_text', '').strip():
                continue

            # Filter by split if specified
            if split_filter and row.get('split') != split_filter:
                continue

            # Extract hypotheses
            hypotheses = []
            for i in range(1, n_hyps + 1):
                hyp = row.get(f'hypothesis_{i}', '')
                score = row.get(f'score_{i}', '')
                if hyp:
                    hypotheses.append({
                        'phonemes': hyp,
                        'score': float(score) if score else None,
                    })

            if not hypotheses:
                continue

            # Calculate PER for best hypothesis
            per = calculate_per(hypotheses[0]['phonemes'], row.get('ground_truth_phonemes', ''))

            samples.append({
                'sample_id': row['sample_id'],
                'session': row['session'],
                'split': row.get('split', 'unknown'),
                'hypotheses': hypotheses,
                'ground_truth_text': row['ground_truth_text'],
                'ground_truth_phonemes': row.get('ground_truth_phonemes', ''),
                'per': per,
                'n_unique_hyps': len(set(h['phonemes'] for h in hypotheses)),
            })

    return samples


def format_nbest_input(hypotheses: list[dict], include_scores: bool = True) -> str:
    """Format N-best hypotheses as LLM input."""
    lines = []
    for i, hyp in enumerate(hypotheses, 1):
        if include_scores and hyp['score'] is not None:
            lines.append(f"Candidate {i} (score {hyp['score']:.2f}): {hyp['phonemes']}")
        else:
            lines.append(f"Candidate {i}: {hyp['phonemes']}")
    return '\n'.join(lines)


def stratified_sample(samples: list[dict], n_samples: int, seed: int = 42) -> list[dict]:
    """
    Stratified sampling by PER and hypothesis diversity.

    Prioritizes samples where multiple hypotheses differ (more interesting for training).
    """
    random.seed(seed)

    # Separate by diversity
    diverse = [s for s in samples if s['n_unique_hyps'] > 1]
    single = [s for s in samples if s['n_unique_hyps'] == 1]

    print(f"Samples with hypothesis diversity: {len(diverse):,}")
    print(f"Samples with single hypothesis:    {len(single):,}")

    # Stratify diverse samples by PER
    low = [s for s in diverse if s['per'] < 0.07]
    medium = [s for s in diverse if 0.07 <= s['per'] < 0.12]
    high = [s for s in diverse if s['per'] >= 0.12]

    print(f"\nDiverse samples by PER stratum:")
    print(f"  Low PER (0-7%):    {len(low):,}")
    print(f"  Medium PER (7-12%): {len(medium):,}")
    print(f"  High PER (12%+):   {len(high):,}")

    # Allocate samples: prioritize diverse, then fill with single
    # Target: 80% diverse, 20% single (or as available)
    n_diverse = min(int(n_samples * 0.8), len(diverse))
    n_single = min(n_samples - n_diverse, len(single))

    selected = []

    # Sample from diverse strata
    if n_diverse > 0:
        n_per_stratum = n_diverse // 3
        remainder = n_diverse % 3

        for i, (stratum, name) in enumerate([(low, 'low'), (medium, 'medium'), (high, 'high')]):
            n = n_per_stratum + (1 if i < remainder else 0)
            if len(stratum) < n:
                print(f"  Warning: {name} diverse stratum has only {len(stratum)} samples")
                selected.extend(stratum)
            else:
                selected.extend(random.sample(stratum, n))

    # Add single-hypothesis samples
    if n_single > 0:
        selected.extend(random.sample(single, n_single))

    random.shuffle(selected)
    return selected


def create_finetune_example(sample: dict, include_scores: bool = True) -> dict:
    """Create a single fine-tuning example in OpenAI format."""
    system_prompt = SYSTEM_PROMPT_NBEST_SCORED if include_scores else SYSTEM_PROMPT_NBEST
    user_content = format_nbest_input(sample['hypotheses'], include_scores=include_scores)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample['ground_truth_text']}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Create OpenAI fine-tuning dataset from N-best predictions')
    parser.add_argument('--input', type=str, default='rnn_baseline/phoneme_predictions_beam.csv',
                        help='Input CSV file with beam search predictions')
    parser.add_argument('--output', type=str, default='rnn_baseline/finetune_train_nbest.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--n-samples', type=int, default=150,
                        help='Number of samples to include')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-remaining', type=str, default='rnn_baseline/finetune_test_nbest.jsonl',
                        help='Output JSONL for remaining samples (test set)')
    parser.add_argument('--train-from-split', type=str, default='both', choices=['train', 'val', 'both'],
                        help='Which split to sample training data from')
    parser.add_argument('--test-from-split', type=str, default='val', choices=['train', 'val', 'both'],
                        help='Which split to use for test data')
    parser.add_argument('--no-scores', action='store_true',
                        help='Exclude acoustic scores from input (just show candidates)')
    args = parser.parse_args()

    include_scores = not args.no_scores

    # Load training candidates
    train_split_filter = None if args.train_from_split == 'both' else args.train_from_split
    test_split_filter = None if args.test_from_split == 'both' else args.test_from_split

    print(f"Loading beam search predictions from {args.input}...")
    print(f"  Training samples from: {args.train_from_split} split")
    print(f"  Test samples from: {args.test_from_split} split")
    print(f"  Include scores: {include_scores}")

    train_candidates = load_beam_predictions(args.input, split_filter=train_split_filter)
    print(f"\nLoaded {len(train_candidates):,} training candidates")

    test_candidates = load_beam_predictions(args.input, split_filter=test_split_filter)
    print(f"Loaded {len(test_candidates):,} test candidates")

    if len(train_candidates) == 0:
        print("ERROR: No training candidates found!")
        return 1

    # PER statistics
    pers = [s['per'] for s in train_candidates]
    print(f"\nPER statistics (training candidates):")
    print(f"  Min:    {min(pers)*100:.2f}%")
    print(f"  Max:    {max(pers)*100:.2f}%")
    print(f"  Mean:   {sum(pers)/len(pers)*100:.2f}%")
    print(f"  Median: {sorted(pers)[len(pers)//2]*100:.2f}%")

    # Stratified sampling
    print(f"\nSelecting {args.n_samples} stratified samples...")
    train_samples = stratified_sample(train_candidates, args.n_samples, args.seed)

    # Test set: exclude training samples
    train_ids = {s['sample_id'] for s in train_samples}
    test_samples = [s for s in test_candidates if s['sample_id'] not in train_ids]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Test:  {len(test_samples):,} samples")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write train JSONL
    print(f"\nWriting train set to {args.output}...")
    with open(args.output, 'w') as f:
        for sample in train_samples:
            example = create_finetune_example(sample, include_scores=include_scores)
            f.write(json.dumps(example) + '\n')

    # Write test JSONL
    print(f"Writing test set to {args.output_remaining}...")
    with open(args.output_remaining, 'w') as f:
        for sample in test_samples:
            example = create_finetune_example(sample, include_scores=include_scores)
            f.write(json.dumps(example) + '\n')

    # Show sample examples
    print(f"\n{'='*60}")
    print("Sample training examples:")
    print(f"{'='*60}")
    for i, sample in enumerate(train_samples[:2]):
        print(f"\nExample {i+1} (PER: {sample['per']*100:.1f}%, {sample['n_unique_hyps']} unique hyps):")
        print(f"  Input:")
        for j, hyp in enumerate(sample['hypotheses'][:3], 1):
            print(f"    Candidate {j}: {hyp['phonemes'][:50]}...")
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
