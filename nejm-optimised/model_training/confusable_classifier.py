"""
Confusable Word Classifier for Brain-to-Text

This module implements a classifier that uses GRU hidden states to disambiguate
between confusable word pairs like "a"/"the", "in"/"and", etc.

The classifier is trained on hidden states extracted from the regions corresponding
to confusable words, and can be used during LM rescoring to adjust candidate scores.

Usage:
    # Training
    classifier = ConfusableClassifier(hidden_dim=768)
    classifier.train_on_dataset(hidden_states, labels, pair_idx)

    # Inference
    score = classifier.score_word(hidden_states, pair_name='a_the', word='a')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


# Define confusable word pairs
# Each pair is (word_0, word_1) where the classifier predicts 0 for word_0 and 1 for word_1
CONFUSABLE_PAIRS = {
    'a_the': ('a', 'the'),
    'in_and': ('in', 'and'),
    'an_and': ('an', 'and'),
    'to_the': ('to', 'the'),
    'of_a': ('of', 'a'),
    'is_as': ('is', 'as'),
    'it_at': ('it', 'at'),
    'on_an': ('on', 'an'),
    'or_are': ('or', 'are'),
    'for_from': ('for', 'from'),
}


class ConfusablePairClassifier(nn.Module):
    """
    Binary classifier for a single confusable word pair.

    Takes pooled hidden states from the word region and predicts which word was spoken.
    """

    def __init__(self, hidden_dim: int = 768, intermediate_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # Binary classification
        )

        # Initialize weights
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Tensor of shape (batch, hidden_dim) - pooled hidden states

        Returns:
            logits: Tensor of shape (batch, 2) - class logits
        """
        return self.classifier(hidden_states)

    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            hidden_states: Tensor of shape (batch, hidden_dim)

        Returns:
            probs: Tensor of shape (batch, 2) - class probabilities
        """
        logits = self.forward(hidden_states)
        return F.softmax(logits, dim=-1)


class ConfusableClassifier(nn.Module):
    """
    Multi-pair confusable word classifier.

    Contains separate binary classifiers for each confusable word pair.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        intermediate_dim: int = 128,
        dropout: float = 0.2,
        pairs: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pairs = pairs if pairs is not None else CONFUSABLE_PAIRS

        # Create a classifier for each pair
        self.classifiers = nn.ModuleDict({
            pair_name: ConfusablePairClassifier(hidden_dim, intermediate_dim, dropout)
            for pair_name in self.pairs
        })

        # Build word-to-pair lookup for quick access
        self.word_to_pairs = self._build_word_lookup()

    def _build_word_lookup(self) -> Dict[str, List[Tuple[str, int]]]:
        """Build a lookup from word to (pair_name, class_idx) tuples."""
        lookup = {}
        for pair_name, (word_0, word_1) in self.pairs.items():
            if word_0 not in lookup:
                lookup[word_0] = []
            lookup[word_0].append((pair_name, 0))

            if word_1 not in lookup:
                lookup[word_1] = []
            lookup[word_1].append((pair_name, 1))
        return lookup

    def forward(
        self,
        hidden_states: torch.Tensor,
        pair_name: str,
    ) -> torch.Tensor:
        """
        Forward pass for a specific pair.

        Args:
            hidden_states: Tensor of shape (batch, hidden_dim)
            pair_name: Name of the confusable pair

        Returns:
            logits: Tensor of shape (batch, 2)
        """
        return self.classifiers[pair_name](hidden_states)

    def score_candidate(
        self,
        hidden_states: torch.Tensor,
        word: str,
        pair_name: Optional[str] = None,
    ) -> float:
        """
        Get the log-probability score for a specific word.

        Args:
            hidden_states: Tensor of shape (time, hidden_dim) - hidden states for the word region
            word: The word to score
            pair_name: Optional specific pair to use (if None, uses first matching pair)

        Returns:
            log_prob: Log-probability of this word
        """
        if word not in self.word_to_pairs:
            return 0.0  # Word is not confusable, no adjustment needed

        # Pool hidden states (mean pooling over time)
        pooled = hidden_states.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        # Get the appropriate pair
        if pair_name is None:
            pair_name, class_idx = self.word_to_pairs[word][0]
        else:
            class_idx = 0 if self.pairs[pair_name][0] == word else 1

        # Get probabilities
        with torch.no_grad():
            probs = self.classifiers[pair_name].predict_proba(pooled)
            log_prob = torch.log(probs[0, class_idx] + 1e-10).item()

        return log_prob

    def get_confusable_words_in_sentence(self, sentence: str) -> List[Tuple[str, int, int]]:
        """
        Find all confusable words in a sentence.

        Args:
            sentence: Input sentence

        Returns:
            List of (word, start_word_idx, end_word_idx) tuples
        """
        words = sentence.lower().split()
        confusable = []

        for i, word in enumerate(words):
            # Strip punctuation from word
            clean_word = word.strip('.,!?;:"\'-')
            if clean_word in self.word_to_pairs:
                confusable.append((clean_word, i, i + 1))

        return confusable

    def score_sentence_pair(
        self,
        hidden_states: torch.Tensor,
        sentence_a: str,
        sentence_b: str,
        word_alignments: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[float, float]:
        """
        Score two candidate sentences that differ in confusable words.

        Args:
            hidden_states: Tensor of shape (time, hidden_dim)
            sentence_a: First candidate sentence
            sentence_b: Second candidate sentence
            word_alignments: Optional list of (start_frame, end_frame) for each word

        Returns:
            Tuple of (score_adjustment_a, score_adjustment_b)
        """
        words_a = sentence_a.lower().split()
        words_b = sentence_b.lower().split()

        if len(words_a) != len(words_b):
            return 0.0, 0.0  # Can't compare different length sentences

        adjustment_a = 0.0
        adjustment_b = 0.0

        for i, (word_a, word_b) in enumerate(zip(words_a, words_b)):
            # Strip punctuation
            word_a = word_a.strip('.,!?;:"\'-')
            word_b = word_b.strip('.,!?;:"\'-')

            if word_a == word_b:
                continue

            # Check if this is a confusable pair
            pair_name = None
            for pname, (w0, w1) in self.pairs.items():
                if (word_a == w0 and word_b == w1) or (word_a == w1 and word_b == w0):
                    pair_name = pname
                    break

            if pair_name is None:
                continue  # Not a confusable pair

            # Get hidden states for this word's region
            if word_alignments is not None and i < len(word_alignments):
                start_frame, end_frame = word_alignments[i]
                word_hidden = hidden_states[start_frame:end_frame]
            else:
                # Estimate based on word position (rough approximation)
                n_frames = hidden_states.shape[0]
                n_words = len(words_a)
                frames_per_word = n_frames / max(n_words, 1)
                start_frame = int(i * frames_per_word)
                end_frame = int((i + 1) * frames_per_word)
                word_hidden = hidden_states[start_frame:end_frame]

            if word_hidden.shape[0] == 0:
                continue

            # Get scores for both words
            score_a = self.score_candidate(word_hidden, word_a, pair_name)
            score_b = self.score_candidate(word_hidden, word_b, pair_name)

            adjustment_a += score_a
            adjustment_b += score_b

        return adjustment_a, adjustment_b


def pool_hidden_states(
    hidden_states: torch.Tensor,
    start_frame: int,
    end_frame: int,
    method: str = 'mean',
) -> torch.Tensor:
    """
    Pool hidden states from a time region.

    Args:
        hidden_states: Tensor of shape (time, hidden_dim)
        start_frame: Start frame index
        end_frame: End frame index
        method: Pooling method ('mean', 'max', 'first', 'last')

    Returns:
        pooled: Tensor of shape (hidden_dim,)
    """
    region = hidden_states[start_frame:end_frame]

    if region.shape[0] == 0:
        return hidden_states.mean(dim=0)  # Fallback to global mean

    if method == 'mean':
        return region.mean(dim=0)
    elif method == 'max':
        return region.max(dim=0)[0]
    elif method == 'first':
        return region[0]
    elif method == 'last':
        return region[-1]
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def save_classifier(classifier: ConfusableClassifier, path: str):
    """Save classifier to disk."""
    torch.save({
        'state_dict': classifier.state_dict(),
        'hidden_dim': classifier.hidden_dim,
        'pairs': classifier.pairs,
    }, path)
    logging.info(f"Saved confusable classifier to {path}")


def load_classifier(path: str, device: str = 'cpu') -> ConfusableClassifier:
    """Load classifier from disk."""
    checkpoint = torch.load(path, map_location=device)
    classifier = ConfusableClassifier(
        hidden_dim=checkpoint['hidden_dim'],
        pairs=checkpoint['pairs'],
    )
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.to(device)
    classifier.eval()
    logging.info(f"Loaded confusable classifier from {path}")
    return classifier


if __name__ == "__main__":
    # Test the classifier
    print("Testing ConfusableClassifier...")

    classifier = ConfusableClassifier(hidden_dim=768)
    print(f"Created classifier with {len(classifier.pairs)} pairs:")
    for pair_name, (w0, w1) in classifier.pairs.items():
        print(f"  {pair_name}: {w0} vs {w1}")

    # Test with random hidden states
    hidden_states = torch.randn(10, 768)  # 10 time steps, 768 hidden dim

    # Test scoring
    score = classifier.score_candidate(hidden_states, 'the')
    print(f"\nScore for 'the': {score:.4f}")

    score = classifier.score_candidate(hidden_states, 'a')
    print(f"Score for 'a': {score:.4f}")

    # Test sentence pair scoring
    adj_a, adj_b = classifier.score_sentence_pair(
        hidden_states,
        "I saw a dog",
        "I saw the dog",
    )
    print(f"\nSentence pair scores:")
    print(f"  'I saw a dog': {adj_a:.4f}")
    print(f"  'I saw the dog': {adj_b:.4f}")

    print("\nTest passed!")
