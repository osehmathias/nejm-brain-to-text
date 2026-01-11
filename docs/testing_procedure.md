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

**Script**: `scripts/generate_phoneme_predictions.py`

```bash
python scripts/generate_phoneme_predictions.py \
    --model_path trained_models/baseline_rnn \
    --split both \
    --output phoneme_predictions.csv \
    --upload-s3
```

### Stage 2: LLM-Based Phoneme-to-Text Conversion

A fine-tuned LLM converts predicted phoneme sequences to text, replacing the traditional 5-gram + OPT rescoring approach from the original paper.

- **Input**: Phoneme sequence (e.g., `DH AH | K AE T | S AE T`)
- **Output**: Text sentence (e.g., `the cat sat`)

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

## LLM Fine-Tuning Configuration

### Model Selection

| Parameter | Value |
|-----------|-------|
| Base Model | `gpt-4.1-2025-04-14` |
| Fine-tuned Model | `ft:gpt-4.1-2025-04-14:*:phoneme-predict:*` |
| Training Method | Supervised Fine-Tuning |

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | OpenAI default for small datasets |
| Batch Size | 1 | Auto-selected for 150 samples |
| Learning Rate Multiplier | 2 | Auto-selected |
| Seed | 150 | Reproducibility |

### Training Data

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 150 | Fine-tuning (stratified by PER) |
| Test | 9,348 | WER evaluation (held out) |

### Stratified Sampling Strategy

Training samples are stratified by Phoneme Error Rate (PER) to ensure coverage of different error patterns:

| Stratum | PER Range | Samples | Purpose |
|---------|-----------|---------|---------|
| Low | 0-7% | 50 | Clean phoneme handling |
| Medium | 7-12% | 50 | Common error patterns |
| High | 12%+ | 50 | Robustness to noise |

**Script**: `analysis/create_finetune_dataset.py`

```bash
python create_finetune_dataset.py \
    --input rnn_baseline/phoneme_predictions.csv \
    --output rnn_baseline/finetune_train.jsonl \
    --n-samples 150
```

### System Prompt

```
You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format. Word boundaries are marked with " | ".

Output only the decoded sentence, nothing else.
```

### Training Data Format (JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are a phoneme-to-text decoder..."},
    {"role": "user", "content": "AY M  |  N AA T  |  K AW N T IH NG  |  AA N  |  IH T  | "},
    {"role": "assistant", "content": "I'm not counting on it."}
  ]
}
```

### Sample Size Justification

From first principles analysis:

1. **Task simplicity**: Phoneme-to-text is deterministic; GPT-4.1 already knows phoneme-grapheme mappings
2. **ARPABET familiarity**: Standard notation likely in base model training
3. **Error pattern coverage**: 150 stratified samples cover low/medium/high PER
4. **Diminishing returns**: Beyond ~100 examples, format is learned; additional examples teach edge cases
5. **Test set preservation**: 150 training samples leaves 9,348 for statistically rigorous WER evaluation

---

## Inference Pipeline

### Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 0 (deterministic) |
| Max Tokens | 256 |
| Retry Logic | 10 attempts with exponential backoff |
| Rate Limit Handling | Automatic with jitter |

### Execution

**Script**: `analysis/run_inference.py`

```bash
python run_inference.py \
    --model "ft:gpt-4.1-2025-04-14:org::phoneme-predict:xxxxx" \
    --input rnn_baseline/finetune_test.jsonl \
    --output rnn_baseline/inference_results.csv
```

### Resume Support

The inference script supports automatic resume after interruption:
- Detects existing results in output file
- Skips already-processed samples
- Appends new results

### Cost Estimation

| Component | Tokens | Rate | Est. Cost |
|-----------|--------|------|-----------|
| Input (9,348 × ~80) | ~750k | $2/1M | ~$1.50 |
| Output (9,348 × ~15) | ~140k | $8/1M | ~$1.12 |
| **Total** | | | **~$2.62** |

---

## WER Evaluation

### Text Normalization

Before WER calculation, text is normalized:
1. Convert to lowercase
2. Remove punctuation (except apostrophes in contractions)
3. Normalize whitespace

### Execution

**Script**: `analysis/calculate_wer.py`

```bash
python calculate_wer.py \
    --input rnn_baseline/inference_results.csv \
    --output rnn_baseline/wer_results.csv
```

### Output Metrics

| Metric | Description |
|--------|-------------|
| Aggregate WER | Total errors / Total reference words |
| 95% CI | Confidence interval using normal approximation |
| Per-sample WER | Mean, median, min, max |
| Error distribution | Perfect, low, medium, high error buckets |

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

### Phoneme Error Rate (RNN Baseline)

| Statistic | Overall | Train Split | Val Split |
|-----------|---------|-------------|-----------|
| Mean PER | 1.49% | 0.01% | ~10% |
| Median PER | 0.00% | 0.00% | - |
| Min PER | 0.00% | 0.00% | - |
| Max PER | 62.07% | 10.00% | 62.07% |

---

## Data Contamination Analysis

### The Problem

The RNN decoder is trained on the **train split** neural data. When generating phoneme predictions:

- **Train split**: RNN has seen this data during training → artificially low PER (0.01%)
- **Val split**: RNN has NOT seen this data → realistic PER (~10%)

This creates a contamination risk: WER measured on train-split samples would be artificially low.

### Evidence

| Split | Samples | Mean PER | Contamination |
|-------|---------|----------|---------------|
| Train | 8,072 | 0.01% | **Yes** (memorized) |
| Val | 1,426 | ~10% | **No** (clean) |
| Overall | 9,498 | 1.49% | Mixed |

### Solution: Split-Aware Evaluation

To ensure scientifically valid results:

1. **LLM Training**: Sample from **both** splits (needed for error pattern diversity)
2. **WER Evaluation**: Use **val split only** (clean, uncontaminated)

```bash
python create_finetune_dataset.py \
    --train-from-split both \
    --test-from-split val \
    --output rnn_baseline/finetune_train_v2.jsonl \
    --output-remaining rnn_baseline/finetune_test_v2.jsonl
```

### Final Dataset Split

| Set | Samples | Source | Purpose |
|-----|---------|--------|---------|
| LLM Train | 150 | Both splits | Error pattern diversity |
| LLM Test | 1,322 | Val only | **Clean WER evaluation** |

### Achievable Precision with Clean Test Set

With 1,322 val-only samples (~13,220 words):

$$
\text{SE} = \sqrt{\frac{0.005 \times 0.995}{13220}} = 5.5 \times 10^{-4}
$$

$$
\text{CI}_{95} = \pm 1.96 \times 5.5 \times 10^{-4} = \pm 0.11\%
$$

| Measured WER | 95% CI | Entire CI < 1%? |
|--------------|--------|-----------------|
| 0.5% | [0.39%, 0.61%] | Yes |
| 0.7% | [0.59%, 0.81%] | Yes |
| 0.8% | [0.69%, 0.91%] | Yes |
| 0.9% | [0.79%, 1.01%] | No (straddles) |

**Conclusion**: With the clean val-only test set, if measured WER ≤ 0.85%, we can confidently claim "< 1%" with 95% confidence. Precision is ±0.11% (sufficient for demonstrating sub-1% WER).

---

## File Structure

```
analysis/
├── rnn_baseline/
│   ├── phoneme_predictions.csv      # Stage 1 output (all 9,498 samples)
│   ├── finetune_train_v2.jsonl      # 150 training samples (from both splits)
│   ├── finetune_test_v2.jsonl       # 1,322 test samples (val only, clean)
│   ├── inference_results_v2.csv     # Stage 2 output
│   └── wer_results_v2.csv           # Stage 3 output
├── create_finetune_dataset.py
├── run_inference.py
└── calculate_wer.py
```

---

## S3 Locations

| Asset | S3 Path |
|-------|---------|
| Phoneme predictions | `s3://river-data-prod-us-east-1/nejm-brain-to-text/phoneme_predictions/rnn_baseline/latest/` |
| Model weights | `s3://river-weights/nejm-brain-to-text/` |

---

## Complete Execution Pipeline

```bash
# 1. Generate phoneme predictions (on EC2 with GPU)
cd model_training
python ../scripts/generate_phoneme_predictions.py \
    --model_path trained_models/baseline_rnn \
    --split both \
    --output ../phoneme_predictions.csv \
    --upload-s3

# 2. Download predictions locally
aws s3 cp s3://river-data-prod-us-east-1/nejm-brain-to-text/phoneme_predictions/rnn_baseline/latest/phoneme_predictions.csv analysis/rnn_baseline/

# 3. Create fine-tuning dataset (contamination-aware split)
cd analysis
python create_finetune_dataset.py \
    --train-from-split both \
    --test-from-split val \
    --output rnn_baseline/finetune_train_v2.jsonl \
    --output-remaining rnn_baseline/finetune_test_v2.jsonl

# 4. Upload to OpenAI and fine-tune
openai api files.create -f rnn_baseline/finetune_train_v2.jsonl -p fine-tune
# Create fine-tuning job via OpenAI dashboard or API

# 5. Run inference on clean test set (val only)
python run_inference.py \
    --model "ft:gpt-4.1-2025-04-14:..." \
    --input rnn_baseline/finetune_test_v2.jsonl \
    --output rnn_baseline/inference_results_v2.csv

# 6. Calculate WER
python calculate_wer.py \
    --input rnn_baseline/inference_results_v2.csv \
    --output rnn_baseline/wer_results_v2.csv \
    --detailed
```

---

## References

1. Willett, F.R., et al. (2023). A high-performance speech neuroprosthesis. *New England Journal of Medicine*.
2. Graves, A., et al. (2006). Connectionist temporal classification. *ICML*.
3. OpenAI. (2025). Fine-tuning documentation. https://platform.openai.com/docs/guides/fine-tuning
