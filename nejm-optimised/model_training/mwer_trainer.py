"""
Minimum Word Error Rate (MWER) Fine-tuning Trainer

This module implements MWER training for fine-tuning a pretrained CTC model
to directly optimize Word Error Rate.

The approach:
1. Generate N-best phoneme hypotheses using beam search
2. Decode hypotheses to words (using greedy phoneme-to-word mapping)
3. Compute WER for each hypothesis against reference
4. Use policy gradient (REINFORCE) to update model weights

Usage:
    python train_model.py --config mwer_args.yaml

References:
    - Prabhavalkar et al., "Minimum Word Error Rate Training for Attention-based
      Sequence-to-Sequence Models" (2018)
    - Shannon, "Optimizing Expected Word Error Rate via Sampling for Speech
      Recognition" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import editdistance
import logging
from dataclasses import dataclass

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


# Phoneme to index mapping (must match training)
PHONEME_LIST = [
    'BLANK',  # 0 - CTC blank
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # 40 - Word boundary/silence
]


@dataclass
class MWERConfig:
    """Configuration for MWER training."""
    beam_size: int = 8
    sample_size: int = 4
    baseline_type: str = 'mean'  # 'mean', 'min', 'none'
    wer_weight: float = 1.0
    ctc_weight: float = 0.1
    temperature: float = 1.0
    use_sampling: bool = True


def greedy_ctc_decode(logits: torch.Tensor) -> List[int]:
    """
    Perform greedy CTC decoding on logits.

    Args:
        logits: Tensor of shape (time, n_classes)

    Returns:
        List of phoneme indices (blanks and duplicates removed)
    """
    # Get argmax predictions
    predictions = logits.argmax(dim=-1).cpu().numpy()

    # Remove consecutive duplicates
    decoded = []
    prev = -1
    for p in predictions:
        if p != prev:
            decoded.append(p)
            prev = p

    # Remove blanks (index 0)
    decoded = [p for p in decoded if p != 0]

    return decoded


def beam_search_decode(
    logits: torch.Tensor,
    beam_size: int = 8,
    blank_idx: int = 0,
) -> List[Tuple[List[int], float]]:
    """
    Perform beam search CTC decoding.

    Args:
        logits: Tensor of shape (time, n_classes)
        beam_size: Number of beams to maintain
        blank_idx: Index of the blank token

    Returns:
        List of (phoneme_sequence, log_prob) tuples for top beams
    """
    T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)

    # Initialize beams: (prefix, log_prob, last_token)
    beams = [([], 0.0, -1)]

    for t in range(T):
        new_beams = {}

        for prefix, score, last_token in beams:
            for v in range(V):
                token_score = log_probs[t, v].item()
                new_score = score + token_score

                if v == blank_idx:
                    # Blank doesn't extend prefix
                    key = tuple(prefix)
                    if key not in new_beams or new_beams[key][1] < new_score:
                        new_beams[key] = (prefix, new_score, v)
                elif v == last_token:
                    # Same token - don't extend (CTC collapse)
                    key = tuple(prefix)
                    if key not in new_beams or new_beams[key][1] < new_score:
                        new_beams[key] = (prefix, new_score, v)
                else:
                    # New token - extend prefix
                    new_prefix = prefix + [v]
                    key = tuple(new_prefix)
                    if key not in new_beams or new_beams[key][1] < new_score:
                        new_beams[key] = (new_prefix, new_score, v)

        # Keep top beams
        beams = sorted(new_beams.values(), key=lambda x: x[1], reverse=True)[:beam_size]

    return [(b[0], b[1]) for b in beams]


def sample_from_logits(
    logits: torch.Tensor,
    n_samples: int = 4,
    temperature: float = 1.0,
) -> List[Tuple[List[int], float]]:
    """
    Sample multiple hypotheses from logits using temperature sampling.

    Args:
        logits: Tensor of shape (time, n_classes)
        n_samples: Number of samples to generate
        temperature: Sampling temperature

    Returns:
        List of (phoneme_sequence, log_prob) tuples
    """
    T, V = logits.shape
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = torch.exp(log_probs)

    samples = []
    for _ in range(n_samples):
        # Sample token at each timestep
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Compute log probability
        log_prob = sum(log_probs[t, sampled[t]].item() for t in range(T))

        # CTC decode (remove blanks and duplicates)
        decoded = []
        prev = -1
        for t in range(T):
            p = sampled[t].item()
            if p != prev and p != 0:
                decoded.append(p)
            prev = p

        samples.append((decoded, log_prob))

    return samples


def phonemes_to_text(phoneme_indices: List[int]) -> str:
    """
    Convert phoneme indices to text representation.

    This is a simplified version - in practice you'd use a phoneme-to-word
    dictionary or language model.
    """
    phonemes = [PHONEME_LIST[i] for i in phoneme_indices if 0 < i < len(PHONEME_LIST)]
    return ' '.join(phonemes)


def compute_wer(hypothesis: List[int], reference: List[int]) -> float:
    """
    Compute Word Error Rate between hypothesis and reference phoneme sequences.

    For simplicity, we treat phoneme sequences as "words" for WER computation.
    In a full system, you'd decode to words first.
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    distance = editdistance.eval(hypothesis, reference)
    return distance / len(reference)


def compute_per(hypothesis: List[int], reference: List[int]) -> float:
    """Compute Phoneme Error Rate."""
    return compute_wer(hypothesis, reference)


class MWERLoss(nn.Module):
    """
    Minimum Word Error Rate Loss.

    Uses policy gradient (REINFORCE) to optimize expected WER.
    """

    def __init__(self, config: MWERConfig):
        super().__init__()
        self.config = config
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MWER loss.

        Args:
            logits: Model outputs of shape (batch, time, n_classes)
            targets: Target phoneme sequences of shape (batch, max_target_len)
            input_lengths: Length of each input in batch
            target_lengths: Length of each target in batch

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        batch_size = logits.shape[0]
        device = logits.device

        total_mwer_loss = 0.0
        total_ctc_loss = 0.0
        total_wer = 0.0
        total_baseline = 0.0

        for b in range(batch_size):
            # Get this sample's data
            T = input_lengths[b].item()
            sample_logits = logits[b, :T]  # (T, n_classes)
            target_len = target_lengths[b].item()
            reference = targets[b, :target_len].cpu().tolist()

            # Generate hypotheses
            if self.config.use_sampling:
                hypotheses = sample_from_logits(
                    sample_logits,
                    n_samples=self.config.sample_size,
                    temperature=self.config.temperature,
                )
            else:
                hypotheses = beam_search_decode(
                    sample_logits,
                    beam_size=self.config.beam_size,
                )[:self.config.sample_size]

            if len(hypotheses) == 0:
                continue

            # Compute WER for each hypothesis
            wers = []
            log_probs = []
            for hyp, log_prob in hypotheses:
                wer = compute_per(hyp, reference)
                wers.append(wer)
                log_probs.append(log_prob)

            wers = np.array(wers)
            log_probs = np.array(log_probs)

            # Compute baseline
            if self.config.baseline_type == 'mean':
                baseline = np.mean(wers)
            elif self.config.baseline_type == 'min':
                baseline = np.min(wers)
            else:
                baseline = 0.0

            # Compute MWER loss (REINFORCE style)
            # Loss = sum_i [ log_prob_i * (wer_i - baseline) ]
            advantages = wers - baseline

            # Normalize log probs (convert to proper probabilities)
            log_probs_tensor = torch.tensor(log_probs, device=device, dtype=torch.float32)
            probs = F.softmax(log_probs_tensor, dim=0)

            # Expected WER
            expected_wer = (probs * torch.tensor(wers, device=device)).sum()

            # Policy gradient loss
            mwer_loss = (probs * torch.tensor(advantages, device=device)).sum()

            total_mwer_loss += mwer_loss
            total_wer += expected_wer.item()
            total_baseline += baseline

        # Average over batch
        mwer_loss = total_mwer_loss / batch_size

        # Optionally add CTC loss for stability
        if self.config.ctc_weight > 0:
            ctc_log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_loss(
                ctc_log_probs, targets, input_lengths, target_lengths
            ).mean()
            total_ctc_loss = ctc_loss.item()
        else:
            ctc_loss = 0.0

        # Combined loss
        loss = self.config.wer_weight * mwer_loss + self.config.ctc_weight * ctc_loss

        metrics = {
            'mwer_loss': mwer_loss.item() if isinstance(mwer_loss, torch.Tensor) else mwer_loss,
            'ctc_loss': total_ctc_loss,
            'expected_wer': total_wer / batch_size,
            'baseline': total_baseline / batch_size,
        }

        return loss, metrics


class MWERTrainer:
    """
    MWER Fine-tuning Trainer.

    Extends the base CTC trainer with MWER loss computation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MWERConfig,
        device: str = 'cuda',
    ):
        self.model = model
        self.config = config
        self.device = device
        self.mwer_loss = MWERLoss(config)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MWER loss for a batch."""
        return self.mwer_loss(logits, targets, input_lengths, target_lengths)


def create_mwer_config(args) -> MWERConfig:
    """Create MWERConfig from args dict or OmegaConf."""
    mwer_dict = args.get('mwer', {})
    return MWERConfig(
        beam_size=mwer_dict.get('beam_size', 8),
        sample_size=mwer_dict.get('sample_size', 4),
        baseline_type=mwer_dict.get('baseline_type', 'mean'),
        wer_weight=mwer_dict.get('wer_weight', 1.0),
        ctc_weight=mwer_dict.get('ctc_weight', 0.1),
        temperature=mwer_dict.get('temperature', 1.0),
        use_sampling=mwer_dict.get('use_sampling', True),
    )


if __name__ == "__main__":
    # Test the MWER components
    print("Testing MWER components...")

    # Create dummy logits
    T, V = 50, 41
    logits = torch.randn(T, V)

    # Test greedy decode
    decoded = greedy_ctc_decode(logits)
    print(f"Greedy decoded {len(decoded)} phonemes")

    # Test beam search
    beams = beam_search_decode(logits, beam_size=4)
    print(f"Beam search returned {len(beams)} hypotheses")
    for i, (hyp, score) in enumerate(beams):
        print(f"  Beam {i}: {len(hyp)} phonemes, score={score:.4f}")

    # Test sampling
    samples = sample_from_logits(logits, n_samples=4)
    print(f"Sampling returned {len(samples)} hypotheses")

    # Test MWER loss
    config = MWERConfig()
    mwer_loss = MWERLoss(config)

    batch_logits = torch.randn(2, T, V)
    targets = torch.randint(1, V, (2, 20))
    input_lengths = torch.tensor([T, T])
    target_lengths = torch.tensor([20, 15])

    loss, metrics = mwer_loss(batch_logits, targets, input_lengths, target_lengths)
    print(f"\nMWER Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("\nTest passed!")
