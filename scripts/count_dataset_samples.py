#!/usr/bin/env python3
"""
Count samples in the brain-to-text dataset.

Usage:
    python scripts/count_dataset_samples.py
    python scripts/count_dataset_samples.py --data-dir /path/to/hdf5_data_final
    python scripts/count_dataset_samples.py --output dataset_stats.json
"""

import argparse
import h5py
import json
import os
from collections import defaultdict


def analyze_session(session_path):
    """Analyze a single session directory."""
    stats = {
        'train_trials': 0,
        'val_trials': 0,
        'train_total_timesteps': 0,
        'val_total_timesteps': 0,
        'train_total_phonemes': 0,
        'val_total_phonemes': 0,
    }

    for split in ['train', 'val']:
        hdf5_path = os.path.join(session_path, f'data_{split}.hdf5')
        if not os.path.exists(hdf5_path):
            continue

        with h5py.File(hdf5_path, 'r') as f:
            trial_keys = [k for k in f.keys() if k.startswith('trial')]
            stats[f'{split}_trials'] = len(trial_keys)

            for trial_key in trial_keys:
                trial = f[trial_key]

                # Count timesteps (neural data length)
                if 'neural_data' in trial:
                    stats[f'{split}_total_timesteps'] += trial['neural_data'].shape[0]
                elif 'tx1' in trial:
                    stats[f'{split}_total_timesteps'] += trial['tx1'].shape[0]

                # Count phonemes
                if 'phonemes' in trial:
                    stats[f'{split}_total_phonemes'] += len(trial['phonemes'][()])
                elif 'phone_seq' in trial:
                    stats[f'{split}_total_phonemes'] += len(trial['phone_seq'][()])
                elif 'transcription' in trial:
                    # Rough estimate: ~4 phonemes per word, ~5 chars per word
                    text = trial['transcription'][()].decode() if isinstance(trial['transcription'][()], bytes) else str(trial['transcription'][()])
                    stats[f'{split}_total_phonemes'] += len(text.split()) * 4

    return stats


def main():
    parser = argparse.ArgumentParser(description='Count samples in brain-to-text dataset')
    parser.add_argument('--data-dir', type=str, default='data/hdf5_data_final',
                        help='Path to hdf5_data_final directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Try: python scripts/count_dataset_samples.py --data-dir /path/to/hdf5_data_final")
        return 1

    sessions = sorted([d for d in os.listdir(args.data_dir)
                       if os.path.isdir(os.path.join(args.data_dir, d))])

    print(f"Found {len(sessions)} sessions in {args.data_dir}\n")
    print(f"{'Session':<20} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 50)

    all_stats = {}
    totals = defaultdict(int)

    for session in sessions:
        session_path = os.path.join(args.data_dir, session)
        stats = analyze_session(session_path)
        all_stats[session] = stats

        for key, value in stats.items():
            totals[key] += value

        total = stats['train_trials'] + stats['val_trials']
        print(f"{session:<20} {stats['train_trials']:>8} {stats['val_trials']:>8} {total:>8}")

    print("-" * 50)
    total_all = totals['train_trials'] + totals['val_trials']
    print(f"{'TOTAL':<20} {totals['train_trials']:>8} {totals['val_trials']:>8} {total_all:>8}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total sessions:        {len(sessions)}")
    print(f"Total train trials:    {totals['train_trials']:,}")
    print(f"Total val trials:      {totals['val_trials']:,}")
    print(f"Total trials:          {total_all:,}")
    print(f"Total train timesteps: {totals['train_total_timesteps']:,}")
    print(f"Total val timesteps:   {totals['val_total_timesteps']:,}")
    print(f"Total train phonemes:  {totals['train_total_phonemes']:,}")
    print(f"Total val phonemes:    {totals['val_total_phonemes']:,}")

    # Save to JSON if requested
    if args.output:
        output_data = {
            'sessions': all_stats,
            'totals': dict(totals),
            'num_sessions': len(sessions),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nStats saved to: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
