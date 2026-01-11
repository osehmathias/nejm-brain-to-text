# Brain-to-Text Decoder Testing Procedure

## Overview

This document describes the testing methodology for evaluating the brain-to-text phoneme decoder, including statistical requirements for Word Error Rate (WER) measurement and the evaluation pipeline.

## Evaluation Pipeline

### Stage 1: Phoneme Prediction Generation

The RNN decoder (baseline GRU or GRU+Attention) processes neural signals and outputs phoneme probabilities at each timestep. CTC (Connectionist Temporal Classification) decoding is applied:

1. **Greedy decoding**: `argmax` over phoneme classes at each timestep
2. **Blank removal**: Remove CTC blank tokens (class 0)
3. **Collapse duplicates**: Merge consecutive identical phonemes

Output: Predicted phoneme sequence for each trial.

### Stage 2: LLM-Based Phoneme-to-Text Conversion

A fine-tuned LLM converts predicted phoneme sequences to text:

- **Input**: Phoneme sequence (e.g., `DH AH | K AE T | S AE T`)
- **Output**: Text sentence (e.g., `the cat sat`)

This replaces the traditional 5-gram + OPT rescoring approach from the original paper.

### Stage 3: WER Calculation

Word Error Rate is computed as:

$$
\text{WER} = \frac{S + D + I}{N}
$$

Where:
- $S$ = substitutions
- $D$ = deletions
- $I$ = insertions
- $N$ = total words in reference

---

## Statistical Requirements for Single-Decimal Precision

### Objective

Report WER with single-decimal precision (e.g., "0.8%", "1.2%") with 95% confidence.

### Derivation

WER follows a binomial distribution where each word is a Bernoulli trial with error probability $p$.

**Standard error of WER:**

$$
\text{SE} = \sqrt{\frac{p(1-p)}{N}}
$$

**95% Confidence Interval:**

$$
\text{CI}_{95} = \pm 1.96 \times \text{SE}
$$

**Requirement for single-decimal precision:**

For reliable rounding to one decimal place, the 95% CI must be $\leq \pm 0.05\%$:

$$
1.96 \times \sqrt{\frac{p(1-p)}{N}} \leq 0.0005
$$

**Solving for N:**

$$
N \geq \frac{(1.96)^2 \times p(1-p)}{(0.0005)^2}
$$

$$
N \geq 1.537 \times 10^7 \times p(1-p)
$$

### Required Sample Sizes

| Expected WER | Required Words | Required Sentences (≈10 words/sent) | 95% CI |
|--------------|----------------|-------------------------------------|--------|
| 0.5% | 7.65 × 10⁴ | 7,650 | ±0.05% |
| 0.8% | 1.22 × 10⁵ | 12,200 | ±0.05% |
| 1.0% | 1.52 × 10⁵ | 15,200 | ±0.05% |
| 1.5% | 2.27 × 10⁵ | 22,700 | ±0.05% |

---

## Dataset Statistics

| Metric | Count |
|--------|-------|
| Total sessions | 45 |
| Total train trials | 8,072 |
| Total val trials | 1,426 |
| **Total trials** | **9,498** |
| Estimated words (≈10/trial) | ~9.5 × 10⁴ |

### Achievable Precision with Current Dataset

At 9.5 × 10⁴ words:

$$
\text{SE} = \sqrt{\frac{0.01 \times 0.99}{95000}} = 3.23 \times 10^{-4}
$$

$$
\text{CI}_{95} = \pm 1.96 \times 3.23 \times 10^{-4} = \pm 0.063\%
$$

**Conclusion**: At expected WER ≤ 0.5%, the dataset provides sufficient precision for single-decimal reporting. At WER = 1.0%, precision is ±0.063% (slightly above the ±0.05% ideal threshold but acceptable).

---

## K-Fold Cross-Validation Strategy

To maximize test data while training the LLM on real phoneme outputs:

1. **Split**: 5-fold cross-validation
2. **For each fold**:
   - Train LLM on 80% of phoneme predictions
   - Test on remaining 20%
3. **Aggregate**: Combine all test predictions for final WER calculation

This ensures:
- All 9,498 samples contribute to final WER
- LLM trained on realistic phoneme error patterns
- No data leakage between train and test

---

## Output Format

Phoneme predictions are generated in CSV format:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique identifier: `{session}_b{block}_t{trial}` |
| `session` | Source recording session |
| `split` | Original split: `train` or `val` |
| `block` | Block number within session |
| `trial` | Trial number within block |
| `prediction` | Predicted phoneme sequence |
| `time_ms` | Inference time in milliseconds |
| `ground_truth_phonemes` | True phoneme sequence |
| `ground_truth_text` | True sentence text |

---

## Execution

### Generate Predictions

```bash
python scripts/generate_phoneme_predictions.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --split both \
    --output phoneme_predictions.csv \
    --upload-s3
```

### S3 Output Location

```
s3://river-data-prod-us-east-1/nejm-brain-to-text/phoneme_predictions/latest/phoneme_predictions.csv
```

---

## References

1. Willett, F.R., et al. (2023). A high-performance speech neuroprosthesis. *New England Journal of Medicine*.
2. Graves, A., et al. (2006). Connectionist temporal classification. *ICML*.
