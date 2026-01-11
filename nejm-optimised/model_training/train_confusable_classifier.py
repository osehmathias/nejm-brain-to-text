"""
Train Confusable Word Classifier

This script trains the confusable word classifier using hidden states extracted
from the trained RNN model on the training data.

Usage:
    python train_confusable_classifier.py \
        --rnn_checkpoint path/to/rnn_checkpoint \
        --data_dir path/to/hdf5_data \
        --output_dir path/to/output

The script will:
1. Load the trained RNN model
2. Extract hidden states for confusable word regions
3. Train binary classifiers for each confusable pair
4. Save the trained classifier
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnn_model import GRUDecoder
from confusable_classifier import (
    ConfusableClassifier,
    CONFUSABLE_PAIRS,
    pool_hidden_states,
    save_classifier,
)
from data_augmentations import gauss_smooth

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


class ConfusableWordDataset(Dataset):
    """Dataset for training confusable word classifier."""

    def __init__(
        self,
        hidden_states: List[torch.Tensor],
        labels: List[int],
        pair_names: List[str],
    ):
        self.hidden_states = hidden_states
        self.labels = labels
        self.pair_names = pair_names

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'label': self.labels[idx],
            'pair_name': self.pair_names[idx],
        }


def load_rnn_model(checkpoint_path: str, args_path: str, device: str) -> Tuple[GRUDecoder, dict]:
    """Load the trained RNN model."""
    logging.info(f"Loading RNN model from {checkpoint_path}")

    # Load config
    config = OmegaConf.load(args_path)
    model_args = config.model

    # Create model
    model = GRUDecoder(
        neural_dim=model_args['neural_dim'],
        n_units=model_args['n_units'],
        n_days=model_args['n_days'],
        n_classes=model_args['n_classes'],
        rnn_dropout=0.0,  # No dropout during inference
        input_dropout=0.0,
        n_layers=model_args['n_layers'],
        patch_size=model_args.get('patch_size', 0),
        patch_stride=model_args.get('patch_stride', 0),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info(f"Loaded RNN model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, config


def extract_word_boundaries(
    transcription: str,
    n_frames: int,
    phoneme_seq: Optional[np.ndarray] = None,
) -> List[Tuple[str, int, int]]:
    """
    Extract approximate word boundaries from transcription.

    Returns list of (word, start_frame, end_frame) tuples.
    """
    words = transcription.lower().split()
    if len(words) == 0:
        return []

    # Simple linear estimation based on word count
    frames_per_word = n_frames / len(words)

    boundaries = []
    for i, word in enumerate(words):
        # Clean punctuation
        clean_word = re.sub(r'[^a-zA-Z\'\-]', '', word).lower()
        if len(clean_word) == 0:
            continue

        start_frame = int(i * frames_per_word)
        end_frame = int((i + 1) * frames_per_word)

        # Ensure valid range
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

        if end_frame > start_frame:
            boundaries.append((clean_word, start_frame, end_frame))

    return boundaries


def find_confusable_words_in_transcription(
    transcription: str,
    pairs: Dict[str, Tuple[str, str]],
) -> List[Tuple[str, str, int]]:
    """
    Find all confusable words in a transcription.

    Returns list of (word, pair_name, class_idx) tuples.
    """
    words = transcription.lower().split()
    confusable = []

    # Build word to pair lookup
    word_to_pairs = {}
    for pair_name, (w0, w1) in pairs.items():
        if w0 not in word_to_pairs:
            word_to_pairs[w0] = []
        word_to_pairs[w0].append((pair_name, 0))
        if w1 not in word_to_pairs:
            word_to_pairs[w1] = []
        word_to_pairs[w1].append((pair_name, 1))

    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-zA-Z\'\-]', '', word).lower()
        if clean_word in word_to_pairs:
            for pair_name, class_idx in word_to_pairs[clean_word]:
                confusable.append((clean_word, pair_name, class_idx, i))

    return confusable


def extract_training_data(
    model: GRUDecoder,
    data_dir: str,
    config: dict,
    device: str,
    max_samples_per_pair: int = 5000,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[int]]]:
    """
    Extract hidden states for confusable words from training data.

    Returns:
        hidden_states_by_pair: Dict mapping pair_name to list of hidden state tensors
        labels_by_pair: Dict mapping pair_name to list of labels (0 or 1)
    """
    logging.info("Extracting training data for confusable classifier...")

    hidden_states_by_pair = {pair_name: [] for pair_name in CONFUSABLE_PAIRS}
    labels_by_pair = {pair_name: [] for pair_name in CONFUSABLE_PAIRS}

    # Get dataset config
    dataset_config = config.dataset

    # Find all HDF5 files
    hdf5_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.hdf5')
    ])

    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {data_dir}")

    logging.info(f"Found {len(hdf5_files)} HDF5 files")

    # Get day mapping from config
    day_mapping = dataset_config.get('day_mapping', {})
    if not day_mapping:
        # Create default mapping based on file order
        day_mapping = {os.path.basename(f).replace('.hdf5', ''): i for i, f in enumerate(hdf5_files)}

    # Process each file
    total_samples = 0
    for hdf5_path in tqdm(hdf5_files, desc="Processing files"):
        session_name = os.path.basename(hdf5_path).replace('.hdf5', '')

        # Get day index
        if session_name in day_mapping:
            day_idx = day_mapping[session_name]
        else:
            continue  # Skip files not in mapping

        with h5py.File(hdf5_path, 'r') as f:
            for trial_key in f.keys():
                trial = f[trial_key]

                # Get neural features
                neural_features = trial['input_features'][:]
                n_time_steps = trial.attrs.get('n_time_steps', neural_features.shape[0])

                # Get transcription
                if 'transcription' not in trial:
                    continue
                transcription_bytes = trial['transcription'][:]
                transcription = ''.join(chr(c) for c in transcription_bytes if c != 0)

                if len(transcription.strip()) == 0:
                    continue

                # Find confusable words
                confusable_words = find_confusable_words_in_transcription(
                    transcription, CONFUSABLE_PAIRS
                )

                if len(confusable_words) == 0:
                    continue

                # Prepare input for model
                x = torch.tensor(neural_features[:n_time_steps], dtype=torch.float32).unsqueeze(0)

                # Apply Gaussian smoothing
                smooth_std = dataset_config.data_transforms.get('smooth_kernel_std', 2)
                smooth_size = dataset_config.data_transforms.get('smooth_kernel_size', 100)
                x = gauss_smooth(x, device, smooth_std, smooth_size, padding='valid')

                x = x.to(device)
                day_tensor = torch.tensor([day_idx], device=device)

                # Get hidden states
                with torch.no_grad():
                    logits, hidden_states = model(
                        x, day_tensor, return_hidden=True
                    )

                hidden_states = hidden_states.squeeze(0).cpu()  # (time, hidden_dim)
                n_frames = hidden_states.shape[0]

                # Get word boundaries
                word_boundaries = extract_word_boundaries(transcription, n_frames)

                # Extract hidden states for each confusable word
                for word, pair_name, class_idx, word_idx in confusable_words:
                    # Check if we have enough samples for this pair
                    if len(hidden_states_by_pair[pair_name]) >= max_samples_per_pair:
                        continue

                    # Find the word boundary
                    if word_idx < len(word_boundaries):
                        _, start_frame, end_frame = word_boundaries[word_idx]
                    else:
                        continue

                    # Extract and pool hidden states
                    word_hidden = pool_hidden_states(
                        hidden_states, start_frame, end_frame, method='mean'
                    )

                    hidden_states_by_pair[pair_name].append(word_hidden)
                    labels_by_pair[pair_name].append(class_idx)
                    total_samples += 1

    logging.info(f"Extracted {total_samples} total samples")
    for pair_name in CONFUSABLE_PAIRS:
        n_samples = len(hidden_states_by_pair[pair_name])
        if n_samples > 0:
            n_class_0 = sum(1 for l in labels_by_pair[pair_name] if l == 0)
            n_class_1 = n_samples - n_class_0
            logging.info(f"  {pair_name}: {n_samples} samples (class 0: {n_class_0}, class 1: {n_class_1})")

    return hidden_states_by_pair, labels_by_pair


def train_classifier(
    classifier: ConfusableClassifier,
    hidden_states_by_pair: Dict[str, List[torch.Tensor]],
    labels_by_pair: Dict[str, List[int]],
    device: str,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.1,
) -> ConfusableClassifier:
    """Train the confusable classifier."""
    logging.info("Training confusable classifier...")

    classifier = classifier.to(device)

    # Train each pair classifier separately
    for pair_name in CONFUSABLE_PAIRS:
        hidden_states = hidden_states_by_pair[pair_name]
        labels = labels_by_pair[pair_name]

        if len(hidden_states) < 10:
            logging.warning(f"Skipping {pair_name} - only {len(hidden_states)} samples")
            continue

        logging.info(f"\nTraining {pair_name} classifier ({len(hidden_states)} samples)...")

        # Convert to tensors
        X = torch.stack(hidden_states)
        y = torch.tensor(labels, dtype=torch.long)

        # Split into train/val
        n_val = max(1, int(len(X) * val_split))
        indices = torch.randperm(len(X))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer for this pair's classifier
        optimizer = torch.optim.AdamW(
            classifier.classifiers[pair_name].parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None

        for epoch in range(n_epochs):
            classifier.classifiers[pair_name].train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = classifier.classifiers[pair_name](batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.shape[0]
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_X.shape[0]

            scheduler.step()

            # Validation
            classifier.classifiers[pair_name].eval()
            with torch.no_grad():
                val_logits = classifier.classifiers[pair_name](X_val.to(device))
                val_acc = (val_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = classifier.classifiers[pair_name].state_dict()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_acc = train_correct / train_total
                logging.info(
                    f"  Epoch {epoch + 1}/{n_epochs}: "
                    f"train_loss={train_loss / train_total:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

        # Restore best model
        if best_state is not None:
            classifier.classifiers[pair_name].load_state_dict(best_state)
        logging.info(f"  Best validation accuracy: {best_val_acc:.4f}")

    classifier.eval()
    return classifier


def main():
    parser = argparse.ArgumentParser(description="Train confusable word classifier")
    parser.add_argument('--rnn_checkpoint', type=str, required=True,
                        help='Path to RNN model checkpoint')
    parser.add_argument('--rnn_args', type=str, default=None,
                        help='Path to RNN args.yaml (default: same dir as checkpoint)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to HDF5 data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trained classifier')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum samples per pair')

    args = parser.parse_args()

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Find args file
    if args.rnn_args is None:
        args.rnn_args = os.path.join(os.path.dirname(args.rnn_checkpoint), 'args.yaml')
    if not os.path.exists(args.rnn_args):
        raise ValueError(f"RNN args file not found: {args.rnn_args}")

    # Load RNN model
    model, config = load_rnn_model(args.rnn_checkpoint, args.rnn_args, device)

    # Extract training data
    hidden_states_by_pair, labels_by_pair = extract_training_data(
        model, args.data_dir, config, device, args.max_samples
    )

    # Create and train classifier
    classifier = ConfusableClassifier(hidden_dim=config.model.n_units)
    classifier = train_classifier(
        classifier,
        hidden_states_by_pair,
        labels_by_pair,
        device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save classifier
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'confusable_classifier.pt')
    save_classifier(classifier, output_path)

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
