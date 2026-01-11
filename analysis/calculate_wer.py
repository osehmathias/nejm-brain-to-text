#!/usr/bin/env python3
"""
Calculate Word Error Rate (WER) from inference results.

WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

Usage:
    python calculate_wer.py
    python calculate_wer.py --input rnn_baseline/inference_results.csv
    python calculate_wer.py --input rnn_baseline/inference_results.csv --detailed
"""

import argparse
import csv
import re
import math
from pathlib import Path
from collections import defaultdict
import editdistance


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def calculate_wer_single(prediction: str, reference: str) -> dict:
    """Calculate WER for a single sample."""
    pred_words = normalize_text(prediction).split()
    ref_words = normalize_text(reference).split()

    if len(ref_words) == 0:
        return {
            'wer': 0.0 if len(pred_words) == 0 else 1.0,
            'errors': len(pred_words),
            'ref_words': 0,
            'pred_words': len(pred_words),
        }

    errors = editdistance.eval(pred_words, ref_words)
    wer = errors / len(ref_words)

    return {
        'wer': wer,
        'errors': errors,
        'ref_words': len(ref_words),
        'pred_words': len(pred_words),
    }


def calculate_confidence_interval(wer: float, n_words: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate confidence interval for WER using normal approximation.

    SE = sqrt(WER * (1 - WER) / N)
    CI = WER ± z * SE
    """
    if n_words == 0:
        return (0.0, 0.0)

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    se = math.sqrt(wer * (1 - wer) / n_words)
    margin = z * se

    return (max(0, wer - margin), min(1, wer + margin))


def load_results(csv_path: str) -> list[dict]:
    """Load inference results from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip error rows
            if row['prediction'].startswith('ERROR'):
                continue
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate WER from inference results')
    parser.add_argument('--input', type=str, default='rnn_baseline/inference_results.csv',
                        help='Input CSV file from inference')
    parser.add_argument('--output', type=str, default='rnn_baseline/wer_results.csv',
                        help='Output CSV with per-sample WER')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-sample results')
    parser.add_argument('--errors-only', action='store_true',
                        help='Only show samples with errors')
    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results):,} samples\n")

    if len(results) == 0:
        print("ERROR: No valid results found!")
        return 1

    # Calculate WER for each sample
    total_errors = 0
    total_ref_words = 0
    sample_wers = []
    detailed_results = []

    for row in results:
        wer_result = calculate_wer_single(row['prediction'], row['ground_truth'])

        total_errors += wer_result['errors']
        total_ref_words += wer_result['ref_words']
        sample_wers.append(wer_result['wer'])

        detailed_results.append({
            'prediction': row['prediction'],
            'ground_truth': row['ground_truth'],
            'wer': wer_result['wer'],
            'errors': wer_result['errors'],
            'ref_words': wer_result['ref_words'],
        })

    # Aggregate WER
    aggregate_wer = total_errors / total_ref_words if total_ref_words > 0 else 0
    ci_low, ci_high = calculate_confidence_interval(aggregate_wer, total_ref_words)

    # Per-sample statistics
    mean_wer = sum(sample_wers) / len(sample_wers)
    sorted_wers = sorted(sample_wers)
    median_wer = sorted_wers[len(sorted_wers) // 2]

    # Error distribution
    perfect = sum(1 for w in sample_wers if w == 0)
    low_error = sum(1 for w in sample_wers if 0 < w <= 0.1)
    med_error = sum(1 for w in sample_wers if 0.1 < w <= 0.3)
    high_error = sum(1 for w in sample_wers if w > 0.3)

    # Print results
    print("=" * 70)
    print("WORD ERROR RATE (WER) RESULTS")
    print("=" * 70)

    print(f"\n{'AGGREGATE METRICS':^70}")
    print("-" * 70)
    print(f"  Total samples:        {len(results):,}")
    print(f"  Total reference words: {total_ref_words:,}")
    print(f"  Total errors:         {total_errors:,}")
    print(f"")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  AGGREGATE WER:  {aggregate_wer*100:>6.2f}%               │")
    print(f"  │  95% CI:         [{ci_low*100:.2f}%, {ci_high*100:.2f}%]       │")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n{'PER-SAMPLE STATISTICS':^70}")
    print("-" * 70)
    print(f"  Mean WER:             {mean_wer*100:.2f}%")
    print(f"  Median WER:           {median_wer*100:.2f}%")
    print(f"  Min WER:              {min(sample_wers)*100:.2f}%")
    print(f"  Max WER:              {max(sample_wers)*100:.2f}%")

    print(f"\n{'ERROR DISTRIBUTION':^70}")
    print("-" * 70)
    print(f"  Perfect (0%):         {perfect:,} ({perfect/len(results)*100:.1f}%)")
    print(f"  Low (0-10%]:          {low_error:,} ({low_error/len(results)*100:.1f}%)")
    print(f"  Medium (10-30%]:      {med_error:,} ({med_error/len(results)*100:.1f}%)")
    print(f"  High (>30%):          {high_error:,} ({high_error/len(results)*100:.1f}%)")

    print(f"\n{'PRECISION ANALYSIS':^70}")
    print("-" * 70)
    se = math.sqrt(aggregate_wer * (1 - aggregate_wer) / total_ref_words)
    print(f"  Standard Error:       {se*100:.4f}%")
    print(f"  95% CI Width:         ±{(ci_high - ci_low)/2*100:.3f}%")
    single_decimal = (ci_high - ci_low) / 2 <= 0.0005
    print(f"  Single-decimal precision: {'YES ✓' if single_decimal else 'NO ✗'}")

    print("\n" + "=" * 70)

    # Save detailed results
    print(f"\nSaving detailed results to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['prediction', 'ground_truth', 'wer', 'errors', 'ref_words']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)

    # Show detailed results if requested
    if args.detailed or args.errors_only:
        print(f"\n{'SAMPLE DETAILS':^70}")
        print("-" * 70)

        samples_to_show = detailed_results
        if args.errors_only:
            samples_to_show = [r for r in detailed_results if r['errors'] > 0]
            print(f"Showing {len(samples_to_show)} samples with errors:\n")
        else:
            samples_to_show = detailed_results[:20]
            print("Showing first 20 samples:\n")

        for i, r in enumerate(samples_to_show[:50]):  # Cap at 50
            print(f"[{i+1}] WER: {r['wer']*100:.1f}% ({r['errors']} errors / {r['ref_words']} words)")
            print(f"    Pred: {r['prediction'][:70]}{'...' if len(r['prediction']) > 70 else ''}")
            print(f"    True: {r['ground_truth'][:70]}{'...' if len(r['ground_truth']) > 70 else ''}")
            print()

    # Summary for paper
    print(f"\n{'FOR PAPER':^70}")
    print("-" * 70)
    print(f"  \"Word Error Rate: {aggregate_wer*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)\"")
    print(f"  \"N = {len(results):,} sentences ({total_ref_words:,} words)\"")


if __name__ == '__main__':
    main()
