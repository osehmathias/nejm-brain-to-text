# Brain-to-Text v2: Architecture Improvement Plan

## Executive Summary

This document outlines the plan to improve the brain-to-text speech neuroprosthesis system from the NEJM 2024 paper. The goal is to reduce Phoneme Error Rate (PER) and Word Error Rate (WER) while maintaining real-time streaming capability.

**Current Baseline:**
- PER: ~10.1%
- WER: ~2.5%

**Target Performance:**
- PER: ~7-8% (20-30% relative improvement)
- WER: ~1.5-2% (20-40% relative improvement)

---

## 1. Current Architecture (Baseline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE (GRU + OPT)                     │
└─────────────────────────────────────────────────────────────────────────────┘

Neural Input (256 electrodes × 2 features = 512 dim @ 50Hz)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Day-Specific Input Layer                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Linear(512 → 512) + Dropout(0.2)  [One per recording day]          │   │
│  │  Purpose: Handle electrode drift across days                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Patch Embedding                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Unfold: kernel=14, stride=4                                         │   │
│  │  Output: (batch, seq_len, 512 × 14 = 7168)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      5-Layer Bidirectional GRU                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GRU(7168 → 768) × 5 layers                                          │   │
│  │  Dropout: 0.4                                                        │   │
│  │  Parameters: ~25M                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CTC Output Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Linear(768 → 41 phonemes + blank)                                   │   │
│  │  Loss: CTC Loss                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Beam Search Decoding                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  5-gram phoneme LM (KenLM)                                           │   │
│  │  Beam width: 500-1500                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM Rescoring                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OPT-6.7B (16-bit, ~13GB VRAM)                                       │   │
│  │  Rescores N-best list from beam search                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        Final Text Output
```

### Baseline Model Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Neural Input | Electrodes | 256 |
| Neural Input | Features per electrode | 2 (threshold crossings) |
| Neural Input | Sample rate | 50 Hz |
| Patch Embedding | Kernel size | 14 |
| Patch Embedding | Stride | 4 |
| GRU | Hidden size | 768 |
| GRU | Layers | 5 |
| GRU | Dropout | 0.4 |
| GRU | Bidirectional | Yes |
| Output | Classes | 41 (39 phonemes + SIL + blank) |
| Training | Batch size | 64 |
| Training | Learning rate | 0.005 (cosine decay) |
| Training | Steps | 120,000 |

---

## 2. Proposed Architecture (Conformer + Llama)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PROPOSED ARCHITECTURE (Conformer + Llama)                  │
└─────────────────────────────────────────────────────────────────────────────┘

Neural Input (256 electrodes × 2 features = 512 dim @ 50Hz)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Day-Specific Input Layer                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Linear(512 → 512) + Dropout(0.2)  [One per recording day]          │   │
│  │  Purpose: Handle electrode drift across days (UNCHANGED)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Patch Embedding + Projection                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Unfold: kernel=14, stride=4                                         │   │
│  │  Linear(7168 → 768) + LayerNorm                                      │   │
│  │  Positional Encoding (sinusoidal)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     6-Layer CAUSAL Conformer                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each layer:                                                     │   │
│  │    ├── Feed-Forward Module (×0.5)                                    │   │
│  │    │     └── Linear(768→3072) → SiLU → Dropout → Linear(3072→768)   │   │
│  │    ├── Causal Multi-Head Self-Attention                              │   │
│  │    │     └── 8 heads, d_k=96, CAUSAL MASK                           │   │
│  │    ├── Causal Convolution Module                                     │   │
│  │    │     └── Depthwise Conv1d(kernel=31) + CAUSAL PADDING           │   │
│  │    └── Feed-Forward Module (×0.5)                                    │   │
│  │                                                                      │   │
│  │  Parameters: ~45M                                                    │   │
│  │  Streaming: YES (causal attention + causal convolution)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CTC Output Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Linear(768 → 41 phonemes + blank)                                   │   │
│  │  Loss: CTC Loss (UNCHANGED)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Beam Search Decoding                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  5-gram phoneme LM (KenLM) (UNCHANGED)                               │   │
│  │  Beam width: 500-1500                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM Rescoring (UPGRADED)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Llama 3.1 8B (4-bit quantized, ~6GB VRAM)                          │   │
│  │  + QLoRA fine-tuned adapter (~100MB)                                │   │
│  │  Rescores N-best list from beam search                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        Final Text Output
```

### Proposed Model Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Conformer | Model dimension (d_model) | 768 |
| Conformer | Attention heads | 8 |
| Conformer | Feed-forward dim | 3072 |
| Conformer | Layers | 6 |
| Conformer | Conv kernel size | 31 |
| Conformer | Dropout | 0.1 |
| Conformer | Causal | Yes |
| Training | Learning rate | 0.001 (lower for transformer) |
| Training | Warmup steps | 2000 (more for transformer) |
| Training | Gradient clip | 1.0 (tighter for transformer) |
| Training | Weight decay | 0.01 |
| Llama | Base model | meta-llama/Llama-3.1-8B |
| Llama | Quantization | 4-bit (NF4) |
| Llama | LoRA rank | 64 |
| Llama | LoRA alpha | 128 |

---

## 3. Key Architectural Changes

### 3.1 GRU → Causal Conformer

**Why Conformer?**
- Combines self-attention (global context) with convolution (local patterns)
- Proven in ASR: 15-25% relative WER reduction over LSTMs (Google, 2020)
- Captures long-range dependencies better than RNNs

**Why Causal?**
- Required for real-time streaming output
- Uses causal attention mask (can only attend to past positions)
- Uses causal convolution padding (left-pad only)
- Matches the deployment requirement of the neuroprosthesis

### 3.2 OPT-6.7B → Llama 3.1 8B (QLoRA)

**Why Llama 3.1?**
- Better performance than OPT on language tasks
- More efficient architecture (Grouped-Query Attention)
- Active community and support

**Why QLoRA?**
- 4-bit quantization: ~6GB VRAM (vs ~16GB for full Llama)
- LoRA adapters: Only train ~0.5% of parameters
- Fits on g5.xlarge (24GB A10G GPU) with room for training
- Can fine-tune on domain-specific error correction

---

## 4. Expected Improvements

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| PER | 10.1% | 7-8% | 20-30% relative |
| WER | 2.5% | 1.5-2% | 20-40% relative |

### Improvement Sources

1. **Conformer Encoder (+15-25% PER reduction)**
   - Better modeling of neural signal patterns
   - Self-attention captures long-range phoneme dependencies
   - Convolution captures local articulatory features

2. **Llama 3.1 Rescoring (+10-15% WER reduction)**
   - Better language understanding than OPT
   - QLoRA fine-tuning on transcription correction
   - Improved handling of rare words and context

---

## 5. Implementation Files

```
nejm-brain-to-text-repo/
├── model_training/
│   ├── conformer_model.py      # NEW: Causal Conformer architecture
│   ├── conformer_args.yaml     # NEW: Conformer training config
│   ├── train_conformer.py      # NEW: Conformer training entry point
│   ├── rnn_model.py            # Existing GRU model
│   ├── rnn_trainer.py          # MODIFIED: Supports both GRU and Conformer
│   └── rnn_args.yaml           # MODIFIED: Added wandb config
│
├── language_model/
│   ├── llama_rescorer.py       # NEW: Llama 3.1 integration
│   ├── train_llama_qlora.py    # NEW: QLoRA fine-tuning
│   └── language-model-standalone.py  # MODIFIED: Supports Llama
│
├── setup_g5xlarge.sh           # NEW: Full setup for g5.xlarge
└── docs/
    └── ARCHITECTURE_PLAN.md    # This document
```

---

## 6. Training Instructions

### 6.1 Environment Setup (AWS g5.xlarge)

```bash
# Clone and enter repository
git clone <repo-url>
cd nejm-brain-to-text-repo

# Run setup script
chmod +x setup_g5xlarge.sh
./setup_g5xlarge.sh

# Activate environment
conda activate b2txt25
```

### 6.2 Download Data

Data is available in S3:
```bash
# Main training data (11.5 GB)
aws s3 cp s3://river-data-prod-us-east-1/brain-to-text-2025/brain-to-text-25.zip ./
unzip brain-to-text-25.zip -d data/

# Or use the Dryad download script
python download_data.py
```

### 6.3 Train Conformer (Stage 1)

```bash
cd model_training

# Train Conformer encoder
python train_conformer.py --config conformer_args.yaml

# Expected: ~10-12 hours on g5.xlarge
# Monitor: https://wandb.ai/<username>/brain-to-text
```

### 6.4 Fine-tune Llama (Stage 2)

```bash
cd language_model

# Create training data from Conformer validation outputs
python train_llama_qlora.py \
    --train_data training_pairs.jsonl \
    --output_dir ./llama_adapter \
    --num_epochs 3

# Expected: ~2-3 hours on g5.xlarge
```

### 6.5 Evaluate

```bash
cd model_training

# Evaluate Conformer + Llama
python evaluate_model.py \
    --checkpoint_path trained_models/conformer_v1/best_model.pt \
    --model_type conformer \
    --use_llm \
    --llm_type llama \
    --llama_adapter ../language_model/llama_adapter
```

---

## 7. Experiment Tracking

All experiments are tracked with Weights & Biases (wandb):

- **Project:** `brain-to-text`
- **Metrics logged:**
  - Training: loss, gradient norm, learning rate
  - Validation: PER, loss, per-day PER
  - Model: architecture type, hyperparameters

### Wandb Dashboard

```
https://wandb.ai/<username>/brain-to-text
```

---

## 8. Hardware Requirements

### Training (g5.xlarge recommended)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA A10G (24GB) or better |
| VRAM | 16GB minimum (Conformer alone: ~8GB, +Llama QLoRA: ~14GB) |
| RAM | 32GB recommended |
| Storage | 50GB for data + models |

### Inference

| Component | Requirement |
|-----------|-------------|
| GPU | 8GB minimum (4-bit Llama) |
| Latency | <100ms per utterance target |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Conformer overfitting | Lower dropout (0.1), more weight decay (0.01), careful LR schedule |
| VRAM overflow | 4-bit quantization, gradient checkpointing, reduced batch size |
| Causal degrades PER | Accepted trade-off for streaming; may increase PER by 1-2% vs bidirectional |
| Domain shift in Llama | Fine-tune with QLoRA on actual transcription errors |

---

## 10. Timeline

| Phase | Tasks |
|-------|-------|
| Phase 1 | Setup environment, download data, verify baseline |
| Phase 2 | Train Conformer, tune hyperparameters |
| Phase 3 | Fine-tune Llama with QLoRA |
| Phase 4 | Integrate and evaluate full pipeline |
| Phase 5 | Ablations and final benchmarking |

---

## Appendix A: File Descriptions

### conformer_model.py

Implements the Causal Conformer architecture:
- `CausalMultiHeadAttention`: Self-attention with causal mask
- `CausalConvolutionModule`: Depthwise separable convolution with causal padding
- `ConformerBlock`: Full Conformer block (FFN-Attention-Conv-FFN)
- `ConformerDecoder`: Full model with day-specific layers and CTC head

### llama_rescorer.py

Llama 3.1 integration for rescoring:
- `build_llama()`: Load 4-bit quantized model with optional QLoRA adapter
- `rescore_with_llama()`: Score hypothesis sentences
- `llama_lm_decode()`: Full N-best rescoring with interpolation

### train_llama_qlora.py

QLoRA fine-tuning pipeline:
- Creates training pairs from acoustic model outputs
- Fine-tunes Llama to correct transcription errors
- Saves LoRA adapters for inference

---

## Appendix B: References

1. Willett et al. (2023). "A high-performance speech neuroprosthesis." *Nature*
2. Gulati et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition." *Interspeech*
3. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*
4. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS*
