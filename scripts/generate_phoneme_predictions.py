#!/usr/bin/env python3
"""
Generate phoneme predictions for all samples using CTC decoding.

Output format: sample_id, session, prediction, time_ms, ground_truth_phonemes, ground_truth_text

Usage:
    python scripts/generate_phoneme_predictions.py --model_path trained_models/baseline_rnn
    python scripts/generate_phoneme_predictions.py --model_path trained_models/rnn_attention --split both
    python scripts/generate_phoneme_predictions.py --model_path trained_models/baseline_rnn --output predictions.csv
    python scripts/generate_phoneme_predictions.py --model_path trained_models/baseline_rnn --upload-s3
"""

import os
import sys
import torch
import numpy as np
import argparse
import time
import csv
from datetime import datetime
from tqdm import tqdm

import boto3
from botocore.config import Config

S3_BUCKET = "river-data-prod-us-east-1"
S3_REGION = "us-east-1"

# Add model_training to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_training'))

from omegaconf import OmegaConf
from data_augmentations import gauss_smooth

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]


def extract_transcription(input_arr):
    """Convert byte array to string."""
    end_idx = np.argwhere(input_arr == 0)
    if len(end_idx) > 0:
        end_idx = end_idx[0, 0]
    else:
        end_idx = len(input_arr)
    return ''.join(chr(c) for c in input_arr[:end_idx])


def load_model(model_path, device):
    """Load model and config from checkpoint."""
    args_path = os.path.join(model_path, 'checkpoint/args.yaml')
    checkpoint_path = os.path.join(model_path, 'checkpoint/best_checkpoint')

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Args file not found: {args_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_args = OmegaConf.load(args_path)

    # Determine model type based on config
    if 'attention' in model_args:
        from rnn_model_attention import GRUDecoderWithAttention
        model = GRUDecoderWithAttention(
            neural_dim=model_args['model']['n_input_features'],
            n_units=model_args['model']['n_units'],
            n_days=len(model_args['dataset']['sessions']),
            n_classes=model_args['dataset']['n_classes'],
            rnn_dropout=model_args['model']['rnn_dropout'],
            input_dropout=model_args['model']['input_network']['input_layer_dropout'],
            n_layers=model_args['model']['n_layers'],
            patch_size=model_args['model']['patch_size'],
            patch_stride=model_args['model']['patch_stride'],
            n_attention_heads=model_args['attention'].get('n_heads', 8),
            attention_dropout=model_args['attention'].get('dropout', 0.1),
            attention_layers=model_args['attention'].get('n_layers', 1),
        )
    else:
        from rnn_model import GRUDecoder
        model = GRUDecoder(
            neural_dim=model_args['model']['n_input_features'],
            n_units=model_args['model']['n_units'],
            n_days=len(model_args['dataset']['sessions']),
            n_classes=model_args['dataset']['n_classes'],
            rnn_dropout=model_args['model']['rnn_dropout'],
            input_dropout=model_args['model']['input_network']['input_layer_dropout'],
            n_layers=model_args['model']['n_layers'],
            patch_size=model_args['model']['patch_size'],
            patch_stride=model_args['model']['patch_stride'],
        )

    # Load weights
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Clean up state dict keys (remove module. and _orig_mod. prefixes)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    return model, model_args


def run_inference(neural_input, day_idx, model, model_args, device):
    """Run model inference and return logits."""
    with torch.autocast(device_type=device.type, enabled=model_args['use_amp'], dtype=torch.bfloat16):
        # Apply gaussian smoothing
        x = gauss_smooth(
            inputs=neural_input,
            device=device,
            smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding='valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x=x,
                day_idx=torch.tensor([day_idx], device=device),
                states=None,
                return_state=True,
            )

    return logits.float().cpu().numpy()


def decode_ctc(logits):
    """Greedy CTC decoding: argmax, remove blanks, collapse duplicates."""
    pred_seq = np.argmax(logits[0], axis=-1)

    # Remove blanks and collapse duplicates
    decoded = []
    prev = -1
    for p in pred_seq:
        if p != 0 and p != prev:  # 0 is blank
            decoded.append(int(p))
        prev = p

    return decoded


def upload_to_s3(local_path, s3_prefix="nejm-brain-to-text/phoneme_predictions"):
    """Upload file to S3 bucket."""
    config = Config(region_name=S3_REGION)
    s3_client = boto3.client("s3", config=config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(local_path)
    s3_key = f"{s3_prefix}/{timestamp}/{filename}"

    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} ({file_size_mb:.1f} MB) to s3://{S3_BUCKET}/{s3_key}")

    s3_client.upload_file(local_path, S3_BUCKET, s3_key)

    # Also upload to "latest"
    latest_key = f"{s3_prefix}/latest/{filename}"
    s3_client.copy_object(
        Bucket=S3_BUCKET,
        CopySource=f"{S3_BUCKET}/{s3_key}",
        Key=latest_key
    )

    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
    print(f"Updated latest: s3://{S3_BUCKET}/{latest_key}")

    return s3_key


def main():
    parser = argparse.ArgumentParser(description='Generate phoneme predictions for all samples')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model directory (e.g., trained_models/baseline_rnn)')
    parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final',
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'],
                        help='Which split to process')
    parser.add_argument('--output', type=str, default='phoneme_predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number (-1 for CPU)')
    parser.add_argument('--upload-s3', action='store_true',
                        help='Upload results to S3 bucket')
    parser.add_argument('--s3-prefix', type=str, default='nejm-brain-to-text/phoneme_predictions',
                        help='S3 prefix for upload')
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Load model
    print(f'Loading model from {args.model_path}...')
    model, model_args = load_model(args.model_path, device)
    sessions = model_args['dataset']['sessions']
    print(f'Model loaded. {len(sessions)} sessions configured.')

    # Determine splits to process
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    # Collect all trials
    import h5py

    trials = []
    for session_idx, session in enumerate(sessions):
        for split in splits:
            hdf5_path = os.path.join(args.data_dir, session, f'data_{split}.hdf5')
            if not os.path.exists(hdf5_path):
                continue

            with h5py.File(hdf5_path, 'r') as f:
                trial_keys = sorted([k for k in f.keys() if k.startswith('trial')])
                for trial_key in trial_keys:
                    trials.append({
                        'session': session,
                        'session_idx': session_idx,
                        'split': split,
                        'trial_key': trial_key,
                        'hdf5_path': hdf5_path,
                    })

    print(f'Found {len(trials)} total trials to process')

    # Process trials and write results
    results = []
    total_time_ms = 0

    with tqdm(total=len(trials), desc='Generating predictions') as pbar:
        for trial_info in trials:
            with h5py.File(trial_info['hdf5_path'], 'r') as f:
                trial = f[trial_info['trial_key']]

                # Load neural data
                neural_features = trial['input_features'][:]
                neural_input = torch.tensor(neural_features, device=device, dtype=torch.float32)
                neural_input = neural_input.unsqueeze(0)  # Add batch dim

                # Get ground truth
                seq_class_ids = trial['seq_class_ids'][:] if 'seq_class_ids' in trial else None
                seq_len = trial.attrs.get('seq_len', 0)
                transcription = trial['transcription'][:] if 'transcription' in trial else None
                block_num = trial.attrs.get('block_num', -1)
                trial_num = trial.attrs.get('trial_num', -1)

                # Ground truth phonemes
                if seq_class_ids is not None and seq_len > 0:
                    gt_phoneme_ids = seq_class_ids[:seq_len]
                    gt_phonemes = ' '.join([LOGIT_TO_PHONEME[p] for p in gt_phoneme_ids])
                else:
                    gt_phonemes = ''

                # Ground truth text
                if transcription is not None:
                    gt_text = extract_transcription(transcription)
                else:
                    gt_text = ''

                # Run inference with timing
                start_time = time.perf_counter()
                logits = run_inference(neural_input, trial_info['session_idx'], model, model_args, device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                total_time_ms += elapsed_ms

                # Decode
                pred_ids = decode_ctc(logits)
                pred_phonemes = ' '.join([LOGIT_TO_PHONEME[p] for p in pred_ids])

                # Create sample ID
                sample_id = f"{trial_info['session']}_b{block_num}_t{trial_num}"

                results.append({
                    'sample_id': sample_id,
                    'session': trial_info['session'],
                    'split': trial_info['split'],
                    'block': block_num,
                    'trial': trial_num,
                    'prediction': pred_phonemes,
                    'time_ms': round(elapsed_ms, 2),
                    'ground_truth_phonemes': gt_phonemes,
                    'ground_truth_text': gt_text,
                })

                pbar.update(1)

    # Write CSV
    print(f'\nWriting results to {args.output}...')
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['sample_id', 'session', 'split', 'block', 'trial',
                      'prediction', 'time_ms', 'ground_truth_phonemes', 'ground_truth_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'Total samples:      {len(results):,}')
    print(f'Total time:         {total_time_ms/1000:.2f}s')
    print(f'Avg time/sample:    {total_time_ms/len(results):.2f}ms')
    print(f'Output file:        {args.output}')
    print(f'{"="*60}')

    # Show sample output
    print('\nSample output (first 3 rows):')
    for r in results[:3]:
        print(f"\n  {r['sample_id']}")
        print(f"    Pred: {r['prediction'][:60]}...")
        print(f"    True: {r['ground_truth_phonemes'][:60]}...")
        print(f"    Text: {r['ground_truth_text'][:60]}...")
        print(f"    Time: {r['time_ms']}ms")

    # Upload to S3 if requested
    if args.upload_s3:
        print(f'\n{"="*60}')
        print('UPLOADING TO S3')
        print(f'{"="*60}')
        s3_key = upload_to_s3(args.output, args.s3_prefix)
        print(f'S3 location: s3://{S3_BUCKET}/{s3_key}')


if __name__ == '__main__':
    main()
