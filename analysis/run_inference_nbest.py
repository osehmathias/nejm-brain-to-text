#!/usr/bin/env python3
"""
Run inference on N-best hypotheses test set using fine-tuned OpenAI model.

The model receives multiple phoneme candidates and outputs the predicted text.

Usage:
    python run_inference_nbest.py --model ft:gpt-4.1-2025-04-14:org::phoneme-nbest:xxxxx
    python run_inference_nbest.py --model ft:gpt-4.1-2025-04-14:org::phoneme-nbest:xxxxx --input rnn_baseline/finetune_test_nbest.jsonl
"""

import argparse
import json
import csv
import time
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError, APITimeoutError


def load_test_data(jsonl_path: str) -> list[dict]:
    """Load test data from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            messages = data['messages']

            # Extract system, user, assistant content
            system_content = None
            user_content = None
            assistant_content = None

            for msg in messages:
                if msg['role'] == 'system':
                    system_content = msg['content']
                elif msg['role'] == 'user':
                    user_content = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_content = msg['content']

            # Parse candidates from user content
            candidates = []
            for line in user_content.split('\n'):
                if line.startswith('Candidate '):
                    candidates.append(line)

            samples.append({
                'idx': idx,
                'system': system_content,
                'user_content': user_content,
                'candidates': candidates,
                'n_candidates': len(candidates),
                'ground_truth': assistant_content,
            })

    return samples


def load_existing_results(csv_path: str) -> set[int]:
    """Load already-processed sample indices from existing results file."""
    if not Path(csv_path).exists():
        return set()

    processed = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(int(row['idx']))

    return processed


def run_inference(client: OpenAI, model: str, samples: list[dict], max_retries: int = 10) -> list[dict]:
    """Run inference on all samples with robust rate limit handling."""
    results = []

    for sample in tqdm(samples, desc="Running inference (N-best)"):
        messages = []
        if sample['system']:
            messages.append({"role": "system", "content": sample['system']})
        messages.append({"role": "user", "content": sample['user_content']})

        # Retry logic with exponential backoff + jitter
        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=256,
                    temperature=0,  # Deterministic output
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                prediction = response.choices[0].message.content.strip()

                results.append({
                    'idx': sample['idx'],
                    'n_candidates': sample['n_candidates'],
                    'prediction': prediction,
                    'ground_truth': sample['ground_truth'],
                    'time_ms': round(elapsed_ms, 2),
                    'tokens_in': response.usage.prompt_tokens,
                    'tokens_out': response.usage.completion_tokens,
                })
                break

            except RateLimitError as e:
                base_wait = min(60, 2 ** attempt)
                jitter = random.uniform(0, base_wait * 0.5)
                wait_time = base_wait + jitter
                tqdm.write(f"Rate limit hit. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

            except (APIError, APITimeoutError) as e:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                tqdm.write(f"API error: {e}. Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    tqdm.write(f"Error: {e}. Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"Failed after {max_retries} attempts: {e}")
                    results.append({
                        'idx': sample['idx'],
                        'n_candidates': sample['n_candidates'],
                        'prediction': f"ERROR: {e}",
                        'ground_truth': sample['ground_truth'],
                        'time_ms': 0,
                        'tokens_in': 0,
                        'tokens_out': 0,
                    })

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with N-best fine-tuned model')
    parser.add_argument('--model', type=str, required=True,
                        help='Fine-tuned model ID (e.g., ft:gpt-4.1-2025-04-14:org::phoneme-nbest:xxxxx)')
    parser.add_argument('--input', type=str, default='rnn_baseline/finetune_test_nbest.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output', type=str, default='rnn_baseline/inference_results_nbest.csv',
                        help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples (for testing)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print("N-BEST HYPOTHESIS INFERENCE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Load test data
    print(f"\nLoading test data...")
    all_samples = load_test_data(args.input)
    print(f"Loaded {len(all_samples):,} samples")

    # Show candidate statistics
    n_candidates = [s['n_candidates'] for s in all_samples]
    print(f"Candidates per sample: {min(n_candidates)}-{max(n_candidates)} (avg: {sum(n_candidates)/len(n_candidates):.1f})")

    # Check for existing results (resume support)
    print(f"\nChecking for existing results...")
    processed = load_existing_results(args.output)
    if processed:
        print(f"Found {len(processed):,} already processed samples")
        samples = [s for s in all_samples if s['idx'] not in processed]
        print(f"Remaining: {len(samples):,} samples to process")
    else:
        samples = all_samples
        print("No existing results found, starting fresh")

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    if len(samples) == 0:
        print("\nAll samples already processed!")
        return

    # Initialize OpenAI client
    client = OpenAI()

    # Run inference
    print(f"\nRunning inference on {len(samples):,} samples...")
    start_time = time.time()
    results = run_inference(client, args.model, samples)
    total_time = time.time() - start_time

    # Save results
    print(f"\nSaving results to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    file_exists = Path(args.output).exists() and len(processed) > 0
    fieldnames = ['idx', 'n_candidates', 'prediction', 'ground_truth', 'time_ms', 'tokens_in', 'tokens_out']

    with open(args.output, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    # Update total count
    total_processed = len(processed) + len(results)

    # Summary statistics
    total_tokens_in = sum(r['tokens_in'] for r in results)
    total_tokens_out = sum(r['tokens_out'] for r in results)
    avg_time_ms = sum(r['time_ms'] for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r['prediction'].startswith('ERROR'))

    print(f"\n{'='*60}")
    print("SUMMARY (this run)")
    print(f"{'='*60}")
    print(f"Processed now:     {len(results):,}")
    print(f"Previously done:   {len(processed):,}")
    print(f"Total complete:    {total_processed:,} / {len(all_samples):,}")
    print(f"Errors:            {errors}")
    print(f"Total time:        {total_time:.1f}s")
    print(f"Avg time/sample:   {avg_time_ms:.0f}ms")
    print(f"Total tokens in:   {total_tokens_in:,}")
    print(f"Total tokens out:  {total_tokens_out:,}")

    # Estimate cost (GPT-4.1 pricing)
    cost_in = total_tokens_in / 1_000_000 * 2.00
    cost_out = total_tokens_out / 1_000_000 * 8.00
    print(f"Est. cost:         ${cost_in + cost_out:.2f}")
    print(f"{'='*60}")

    # Show sample results
    print("\nSample results:")
    for i, r in enumerate(results[:3]):
        print(f"\n[{i+1}] ({r['n_candidates']} candidates)")
        print(f"  Prediction:   {r['prediction']}")
        print(f"  Ground truth: {r['ground_truth']}")

    print(f"\nTo calculate WER:")
    print(f"  python calculate_wer.py --input {args.output}")


if __name__ == '__main__':
    main()
