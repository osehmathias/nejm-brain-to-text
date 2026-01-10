"""
Phoneme-only evaluation script.
Outputs predicted phoneme sequences without language model.

Usage:
    python evaluate_phonemes.py --model_path ../data/t15_pretrained_rnn_baseline --eval_type val
"""

import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME

parser = argparse.ArgumentParser(description='Evaluate RNN model - phoneme output only (no language model)')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory')
parser.add_argument('--eval_type', type=str, default='val', choices=['val', 'test'],
                    help='Evaluation type: "val" or "test"')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata')
parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU number to use. Set to -1 for CPU.')
parser.add_argument('--output_csv', type=str, default=None,
                    help='Optional: save phoneme predictions to CSV')
args = parser.parse_args()

# Setup device
if torch.cuda.is_available() and args.gpu_number >= 0:
    if args.gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU {args.gpu_number} not available. Available: {torch.cuda.device_count()}')
    device = torch.device(f'cuda:{args.gpu_number}')
    print(f'Using GPU: {torch.cuda.get_device_name(args.gpu_number)}')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Load model config
model_args = OmegaConf.load(os.path.join(args.model_path, 'checkpoint/args.yaml'))

# Load CSV metadata
b2txt_csv_df = pd.read_csv(args.csv_path)

# Initialize model
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
checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint/best_checkpoint'), weights_only=False)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load data
test_data = {}
total_trials = 0
for session in model_args['dataset']['sessions']:
    files = os.listdir(os.path.join(args.data_dir, session))
    if f'data_{args.eval_type}.hdf5' in files:
        eval_file = os.path.join(args.data_dir, session, f'data_{args.eval_type}.hdf5')
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        total_trials += len(data["neural_features"])
        print(f'Loaded {len(data["neural_features"])} trials from {session}')

print(f'\nTotal {args.eval_type} trials: {total_trials}')

# Run inference
results = []
total_edit_distance = 0
total_phonemes = 0

with tqdm(total=total_trials, desc='Decoding phonemes') as pbar:
    for session, data in test_data.items():
        input_layer = model_args['dataset']['sessions'].index(session)

        for trial_idx in range(len(data['neural_features'])):
            # Get neural input
            neural_input = data['neural_features'][trial_idx]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # Run model
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)

            # Decode phonemes (greedy CTC decoding)
            pred_seq = np.argmax(logits[0], axis=-1)
            pred_seq = [int(p) for p in pred_seq if p != 0]  # remove blanks
            pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]  # remove duplicates
            pred_phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq]

            # Get metadata
            block_num = data['block_num'][trial_idx]
            trial_num = data['trial_num'][trial_idx]

            result = {
                'session': session,
                'block': block_num,
                'trial': trial_num,
                'pred_phonemes': ' '.join(pred_phonemes),
            }

            # If validation, compute PER
            if args.eval_type == 'val':
                true_seq = data['seq_class_ids'][trial_idx][:data['seq_len'][trial_idx]]
                true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_seq]

                ed = editdistance.eval(pred_seq, list(true_seq))
                total_edit_distance += ed
                total_phonemes += len(true_seq)

                result['true_phonemes'] = ' '.join(true_phonemes)
                result['sentence_label'] = data['sentence_label'][trial_idx]
                result['edit_distance'] = ed
                result['per'] = ed / len(true_seq) if len(true_seq) > 0 else 0

            results.append(result)
            pbar.update(1)

# Print summary
print('\n' + '='*60)
if args.eval_type == 'val':
    aggregate_per = total_edit_distance / total_phonemes
    print(f'Aggregate Phoneme Error Rate (PER): {100*aggregate_per:.2f}%')
    print(f'Total edit distance: {total_edit_distance}')
    print(f'Total phonemes: {total_phonemes}')
print('='*60)

# Save to CSV if requested
if args.output_csv:
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f'\nResults saved to {args.output_csv}')

# Print sample predictions
print('\nSample predictions:')
for i, r in enumerate(results[:5]):
    print(f"\n[{r['session']} Block {r['block']} Trial {r['trial']}]")
    if args.eval_type == 'val':
        print(f"  Sentence: {r['sentence_label']}")
        print(f"  True:     {r['true_phonemes']}")
    print(f"  Pred:     {r['pred_phonemes']}")
    if args.eval_type == 'val':
        print(f"  PER:      {100*r['per']:.1f}%")
