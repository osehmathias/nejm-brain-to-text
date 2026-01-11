"""
Train the Conformer-based brain-to-text decoder.

Usage:
    python train_conformer.py
    python train_conformer.py --config custom_args.yaml
"""

import argparse
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Conformer brain-to-text decoder')
    parser.add_argument(
        '--config',
        type=str,
        default='conformer_args.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Initialize trainer and train
    trainer = BrainToTextDecoder_Trainer(config)
    metrics = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best validation PER: {trainer.best_val_PER:.4f}")


if __name__ == '__main__':
    main()
