#!/usr/bin/env python3
"""
Run baseline inference (no fine-tuning) using Claude or OpenAI models.

Tests zero-shot and few-shot variants for phoneme-to-text conversion.

Usage:
    # Zero-shot with Claude
    python run_inference_baseline.py --provider anthropic --model claude-opus-4-5-20250514 --mode zero-shot

    # Few-shot with OpenAI
    python run_inference_baseline.py --provider openai --model gpt-4.1-2025-04-14 --mode few-shot --n-shots 5

    # Test with limit
    python run_inference_baseline.py --provider anthropic --model claude-opus-4-5-20250514 --mode zero-shot --limit 10
"""

import argparse
import json
import csv
import time
import random
from pathlib import Path
from tqdm import tqdm

# Lazy imports for providers
openai_client = None
anthropic_client = None


SYSTEM_PROMPT_ZERO_SHOT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

Output only the decoded sentence, nothing else."""


SYSTEM_PROMPT_FEW_SHOT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

Here are some examples:

{examples}

Now convert the following phoneme sequence. Output only the decoded sentence, nothing else."""


def get_openai_client():
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()
    return openai_client


def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()
    return anthropic_client


def load_val_data(csv_path: str) -> list[dict]:
    """Load val-only samples from predictions CSV."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only load val split (clean, uncontaminated)
            if row.get('split') != 'val':
                continue
            if not row['ground_truth_text'].strip():
                continue

            samples.append({
                'phonemes': row['prediction'],
                'ground_truth': row['ground_truth_text'],
                'sample_id': row['sample_id'],
            })
    return samples


def select_few_shot_examples(samples: list[dict], n_shots: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Select few-shot examples and return (examples, remaining_samples)."""
    random.seed(seed)

    # Select diverse examples (different lengths, etc.)
    sorted_samples = sorted(samples, key=lambda x: len(x['phonemes']))

    # Sample from different parts of the distribution
    indices = []
    step = len(sorted_samples) // n_shots
    for i in range(n_shots):
        idx = min(i * step + random.randint(0, step // 2), len(sorted_samples) - 1)
        indices.append(idx)

    examples = [sorted_samples[i] for i in indices]
    example_ids = {e['sample_id'] for e in examples}
    remaining = [s for s in samples if s['sample_id'] not in example_ids]

    return examples, remaining


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format examples for the prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"Input: {ex['phonemes']}")
        formatted.append(f"Output: {ex['ground_truth']}")
        formatted.append("")
    return "\n".join(formatted)


def call_openai(model: str, system_prompt: str, user_content: str, max_retries: int = 10) -> tuple[str, dict]:
    """Call OpenAI API with retry logic."""
    from openai import RateLimitError, APIError, APITimeoutError

    client = get_openai_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=256,
                temperature=0,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return response.choices[0].message.content.strip(), {
                'time_ms': round(elapsed_ms, 2),
                'tokens_in': response.usage.prompt_tokens,
                'tokens_out': response.usage.completion_tokens,
            }

        except RateLimitError:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            tqdm.write(f"Rate limit. Waiting {wait:.1f}s...")
            time.sleep(wait)
        except (APIError, APITimeoutError) as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            tqdm.write(f"API error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return "ERROR: Max retries exceeded", {'time_ms': 0, 'tokens_in': 0, 'tokens_out': 0}


def call_anthropic(model: str, system_prompt: str, user_content: str, max_retries: int = 10) -> tuple[str, dict]:
    """Call Anthropic API with retry logic."""
    import anthropic

    client = get_anthropic_client()

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return response.content[0].text.strip(), {
                'time_ms': round(elapsed_ms, 2),
                'tokens_in': response.usage.input_tokens,
                'tokens_out': response.usage.output_tokens,
            }

        except anthropic.RateLimitError:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            tqdm.write(f"Rate limit. Waiting {wait:.1f}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            tqdm.write(f"API error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return "ERROR: Max retries exceeded", {'time_ms': 0, 'tokens_in': 0, 'tokens_out': 0}


def run_inference(provider: str, model: str, system_prompt: str, samples: list[dict]) -> list[dict]:
    """Run inference on all samples."""
    results = []

    call_fn = call_openai if provider == 'openai' else call_anthropic

    for sample in tqdm(samples, desc=f"Running inference ({provider})"):
        prediction, stats = call_fn(model, system_prompt, sample['phonemes'])

        results.append({
            'sample_id': sample['sample_id'],
            'phonemes': sample['phonemes'],
            'prediction': prediction,
            'ground_truth': sample['ground_truth'],
            **stats,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Run baseline inference (no fine-tuning)')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'anthropic'],
                        help='API provider')
    parser.add_argument('--model', type=str, required=True,
                        help='Model ID (e.g., gpt-4.1-2025-04-14, claude-opus-4-5-20250514)')
    parser.add_argument('--mode', type=str, default='zero-shot', choices=['zero-shot', 'few-shot'],
                        help='Inference mode')
    parser.add_argument('--n-shots', type=int, default=5,
                        help='Number of few-shot examples (only used in few-shot mode)')
    parser.add_argument('--input', type=str, default='rnn_baseline/phoneme_predictions.csv',
                        help='Input CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (auto-generated if not specified)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Auto-generate output filename
    if args.output is None:
        model_short = args.model.split('/')[-1].replace(':', '-')[:30]
        args.output = f"rnn_baseline/baseline_{args.provider}_{model_short}_{args.mode}.csv"

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Load val data only
    print(f"\nLoading val-only data...")
    all_samples = load_val_data(args.input)
    print(f"Loaded {len(all_samples):,} val samples")

    # Handle few-shot examples
    few_shot_examples = []
    if args.mode == 'few-shot':
        few_shot_examples, samples = select_few_shot_examples(all_samples, args.n_shots, args.seed)
        print(f"Selected {len(few_shot_examples)} few-shot examples")
        print(f"Remaining test samples: {len(samples)}")

        # Build system prompt with examples
        examples_text = format_few_shot_examples(few_shot_examples)
        system_prompt = SYSTEM_PROMPT_FEW_SHOT.format(examples=examples_text)
    else:
        samples = all_samples
        system_prompt = SYSTEM_PROMPT_ZERO_SHOT

    # Apply limit
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples to process!")
        return 1

    # Run inference
    print(f"\nRunning inference on {len(samples):,} samples...")
    start_time = time.time()
    results = run_inference(args.provider, args.model, system_prompt, samples)
    total_time = time.time() - start_time

    # Save results
    print(f"\nSaving results to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', newline='') as f:
        fieldnames = ['sample_id', 'phonemes', 'prediction', 'ground_truth', 'time_ms', 'tokens_in', 'tokens_out']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total_tokens_in = sum(r['tokens_in'] for r in results)
    total_tokens_out = sum(r['tokens_out'] for r in results)
    avg_time_ms = sum(r['time_ms'] for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r['prediction'].startswith('ERROR'))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Provider:          {args.provider}")
    print(f"Model:             {args.model}")
    print(f"Mode:              {args.mode}")
    if args.mode == 'few-shot':
        print(f"Few-shot examples: {args.n_shots}")
    print(f"Samples processed: {len(results):,}")
    print(f"Errors:            {errors}")
    print(f"Total time:        {total_time:.1f}s")
    print(f"Avg time/sample:   {avg_time_ms:.0f}ms")
    print(f"Total tokens in:   {total_tokens_in:,}")
    print(f"Total tokens out:  {total_tokens_out:,}")

    # Cost estimation
    if args.provider == 'openai':
        cost_in = total_tokens_in / 1_000_000 * 2.00
        cost_out = total_tokens_out / 1_000_000 * 8.00
    else:  # anthropic
        cost_in = total_tokens_in / 1_000_000 * 15.00  # Opus pricing
        cost_out = total_tokens_out / 1_000_000 * 75.00

    print(f"Est. cost:         ${cost_in + cost_out:.2f}")
    print(f"{'='*60}")

    # Sample results
    print("\nSample results:")
    for i, r in enumerate(results[:3]):
        print(f"\n[{i+1}] {r['sample_id']}")
        print(f"  Phonemes:    {r['phonemes'][:60]}...")
        print(f"  Prediction:  {r['prediction']}")
        print(f"  Ground truth: {r['ground_truth']}")

    # Save few-shot examples for reference
    if args.mode == 'few-shot':
        examples_file = args.output.replace('.csv', '_examples.json')
        with open(examples_file, 'w') as f:
            json.dump([{'phonemes': e['phonemes'], 'ground_truth': e['ground_truth']} for e in few_shot_examples], f, indent=2)
        print(f"\nFew-shot examples saved to: {examples_file}")

    print(f"\nTo calculate WER:")
    print(f"  python calculate_wer.py --input {args.output}")


if __name__ == '__main__':
    main()
