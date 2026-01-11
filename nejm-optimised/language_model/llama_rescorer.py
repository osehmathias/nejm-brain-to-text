"""
Llama-based Language Model Rescorer for Brain-to-Text

Replaces OPT 6.7B with Llama 3.1 8B for improved WER.
Supports both inference-only and QLoRA fine-tuning modes.

Usage:
    # Inference only (4-bit quantized)
    model, tokenizer = build_llama()
    scores = rescore_with_llama(model, tokenizer, device, hypotheses, length_penalty)

    # With QLoRA adapter
    model, tokenizer = build_llama(adapter_path="path/to/adapter")
"""

import torch
import numpy as np
import logging
from typing import List, Optional, Tuple

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def build_llama(
    model_name: str = 'meta-llama/Llama-3.1-8B',
    cache_dir: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_4bit: bool = True,
    adapter_path: Optional[str] = None,
) -> Tuple:
    """
    Load Llama 3.1 8B model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache model weights
        device: Device to load model on
        use_4bit: Whether to use 4-bit quantization (recommended for 24GB VRAM)
        adapter_path: Path to QLoRA adapter weights (optional)

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logging.info(f"Loading Llama model: {model_name}")

    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logging.info("Using 4-bit quantization")
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto" if device == 'cuda' else None,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )

    # Load QLoRA adapter if provided
    if adapter_path is not None:
        from peft import PeftModel
        logging.info(f"Loading QLoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference

    # Set to evaluation mode
    model.eval()

    # Ensure padding token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loaded Llama model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


@torch.inference_mode()
def rescore_with_llama(
    model,
    tokenizer,
    device: str,
    hypotheses: List[str],
    length_penalty: float = 0.0,
    batch_size: int = 8,
) -> List[float]:
    """
    Rescore hypotheses using Llama model.

    Computes log-probability of each hypothesis under the language model.

    Args:
        model: Llama model
        tokenizer: Llama tokenizer
        device: Device for computation
        hypotheses: List of candidate sentences to score
        length_penalty: Penalty per token (subtracted from score)
        batch_size: Batch size for inference

    Returns:
        List of log-probability scores for each hypothesis
    """
    model.eval()

    all_scores = []

    # Process in batches to avoid OOM
    for i in range(0, len(hypotheses), batch_size):
        batch = hypotheses[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # Compute log-probabilities
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        log_probs = log_probs.cpu().numpy()

        input_ids = inputs['input_ids'].cpu().numpy()
        attention_mask = inputs['attention_mask'].cpu().numpy()

        for j in range(len(batch)):
            n_tokens = int(attention_mask[j].sum())
            # Sum log-probs of each token given the previous context
            score = sum(
                log_probs[j, t - 1, input_ids[j, t]]
                for t in range(1, n_tokens)
            )
            score = score - n_tokens * length_penalty
            all_scores.append(score)

    return all_scores


def llama_lm_decode(
    model,
    tokenizer,
    device: str,
    nbest: List,
    acoustic_scale: float,
    length_penalty: float,
    alpha: float,
    return_confidence: bool = False,
    current_context_str: Optional[str] = None,
):
    """
    Decode using Llama LM rescoring.

    Drop-in replacement for gpt2_lm_decode.

    Args:
        model: Llama model
        tokenizer: Llama tokenizer
        device: Device for computation
        nbest: N-best list from ngram decoder [(sentence, acoustic_score, lm_score), ...]
        acoustic_scale: Weight for acoustic score
        length_penalty: Penalty per token
        alpha: Interpolation weight for LLM scores (1 = only LLM, 0 = only ngram)
        return_confidence: Whether to return confidence score
        current_context_str: Previous context to prepend to hypotheses

    Returns:
        If return_confidence=False: (best_hypothesis, nbest_with_scores)
        If return_confidence=True: (best_hypothesis, nbest_with_scores, confidence)
    """
    hypotheses = []
    acoustic_scores = []
    old_lm_scores = []

    for out in nbest:
        hyp = out[0].strip()
        if len(hyp) == 0:
            continue

        # Add context to the front of each sentence
        if current_context_str is not None and len(current_context_str.split()) > 0:
            hyp = current_context_str + ' ' + hyp

        # Clean up punctuation spacing
        hyp = hyp.replace('>', '')
        hyp = hyp.replace('  ', ' ')
        hyp = hyp.replace(' ,', ',')
        hyp = hyp.replace(' .', '.')
        hyp = hyp.replace(' ?', '?')

        hypotheses.append(hyp)
        acoustic_scores.append(out[1])
        old_lm_scores.append(out[2])

    if len(hypotheses) == 0:
        logging.error('In llama_lm_decode, len(hypotheses) == 0')
        return ("", []) if not return_confidence else ("", [], 0.)

    # Convert to numpy arrays
    acoustic_scores = np.array(acoustic_scores)
    old_lm_scores = np.array(old_lm_scores)

    # Get new LM scores from Llama
    try:
        new_lm_scores = np.array(rescore_with_llama(
            model, tokenizer, device, hypotheses, length_penalty
        ))
    except Exception as e:
        logging.error(f'Error during Llama rescore: {e}')
        # Fallback to batched rescoring
        try:
            new_lm_scores = []
            batch_size = max(1, len(hypotheses) // 5)
            for i in range(0, len(hypotheses), batch_size):
                scores = rescore_with_llama(
                    model, tokenizer, device,
                    hypotheses[i:i + batch_size],
                    length_penalty
                )
                new_lm_scores.extend(scores)
            new_lm_scores = np.array(new_lm_scores)
        except Exception as e:
            logging.error(f'Error during batched Llama rescore: {e}')
            new_lm_scores = np.zeros(len(hypotheses))

    # Remove context from start of each sentence
    if current_context_str is not None and len(current_context_str.split()) > 0:
        hypotheses = [h[(len(current_context_str) + 1):] for h in hypotheses]

    # Calculate total scores
    total_scores = (acoustic_scale * acoustic_scores) + \
                   ((1 - alpha) * old_lm_scores) + \
                   (alpha * new_lm_scores)

    # Get the best hypothesis
    max_idx = np.argmax(total_scores)
    best_hyp = hypotheses[max_idx]

    # Create nbest output
    nbest_out = []
    min_len = np.min((len(nbest), len(new_lm_scores), len(total_scores)))
    for i in range(min_len):
        nbest_out.append(';'.join(map(str, [
            nbest[i][0], nbest[i][1], nbest[i][2],
            new_lm_scores[i], total_scores[i]
        ])))

    # Return
    if not return_confidence:
        return best_hyp, nbest_out
    else:
        total_scores = total_scores - np.max(total_scores)
        probs = np.exp(total_scores)
        return best_hyp, nbest_out, probs[max_idx] / np.sum(probs)


def llama_lm_decode_with_confusable(
    model,
    tokenizer,
    device: str,
    nbest: List,
    acoustic_scale: float,
    length_penalty: float,
    alpha: float,
    confusable_classifier,
    hidden_states: Optional[torch.Tensor] = None,
    confusable_weight: float = 0.5,
    return_confidence: bool = False,
    current_context_str: Optional[str] = None,
):
    """
    Decode using Llama LM rescoring with confusable word classifier adjustment.

    This extends llama_lm_decode by adding score adjustments from the confusable
    word classifier for candidates that differ in confusable words.

    Args:
        model: Llama model
        tokenizer: Llama tokenizer
        device: Device for computation
        nbest: N-best list from ngram decoder [(sentence, acoustic_score, lm_score), ...]
        acoustic_scale: Weight for acoustic score
        length_penalty: Penalty per token
        alpha: Interpolation weight for LLM scores
        confusable_classifier: Trained ConfusableClassifier instance
        hidden_states: GRU hidden states tensor of shape (time, hidden_dim)
        confusable_weight: Weight for confusable classifier scores
        return_confidence: Whether to return confidence score
        current_context_str: Previous context to prepend to hypotheses

    Returns:
        If return_confidence=False: (best_hypothesis, nbest_with_scores)
        If return_confidence=True: (best_hypothesis, nbest_with_scores, confidence)
    """
    # First get base scores from Llama
    result = llama_lm_decode(
        model, tokenizer, device, nbest,
        acoustic_scale, length_penalty, alpha,
        return_confidence=True,
        current_context_str=current_context_str,
    )

    best_hyp, nbest_out, confidence = result

    if confusable_classifier is None or hidden_states is None:
        if return_confidence:
            return best_hyp, nbest_out, confidence
        return best_hyp, nbest_out

    # Parse nbest_out to get sentences and scores
    sentences = []
    scores = []
    for entry in nbest_out:
        parts = entry.split(';')
        if len(parts) >= 5:
            sentences.append(parts[0])
            scores.append(float(parts[4]))  # total_score

    if len(sentences) < 2:
        if return_confidence:
            return best_hyp, nbest_out, confidence
        return best_hyp, nbest_out

    # Apply confusable classifier adjustments
    adjusted_scores = scores.copy()

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            adj_i, adj_j = confusable_classifier.score_sentence_pair(
                hidden_states, sentences[i], sentences[j]
            )
            adjusted_scores[i] += confusable_weight * adj_i
            adjusted_scores[j] += confusable_weight * adj_j

    # Find new best hypothesis
    best_idx = np.argmax(adjusted_scores)
    best_hyp = sentences[best_idx]

    # Update nbest_out with adjusted scores
    new_nbest_out = []
    for i, entry in enumerate(nbest_out):
        parts = entry.split(';')
        if len(parts) >= 5:
            parts[4] = str(adjusted_scores[i])
            new_nbest_out.append(';'.join(parts))
        else:
            new_nbest_out.append(entry)

    if return_confidence:
        adjusted_scores = np.array(adjusted_scores)
        adjusted_scores = adjusted_scores - np.max(adjusted_scores)
        probs = np.exp(adjusted_scores)
        confidence = probs[best_idx] / np.sum(probs)
        return best_hyp, new_nbest_out, confidence

    return best_hyp, new_nbest_out


# For backwards compatibility
def build_opt_or_llama(
    model_type: str = 'llama',
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: str = 'cuda',
    use_4bit: bool = True,
    adapter_path: Optional[str] = None,
) -> Tuple:
    """
    Factory function to build either OPT or Llama model.

    Args:
        model_type: 'opt' or 'llama'
        model_name: HuggingFace model name (defaults based on type)
        cache_dir: Cache directory for model weights
        device: Device to load model on
        use_4bit: Use 4-bit quantization (Llama only)
        adapter_path: Path to QLoRA adapter (Llama only)

    Returns:
        Tuple of (model, tokenizer, rescore_function)
    """
    if model_type == 'llama':
        if model_name is None:
            model_name = 'meta-llama/Llama-3.1-8B'
        model, tokenizer = build_llama(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            use_4bit=use_4bit,
            adapter_path=adapter_path,
        )
        rescore_fn = rescore_with_llama
    else:
        # Fall back to OPT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if model_name is None:
            model_name = 'facebook/opt-6.7b'

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        )
        if device != 'cpu':
            model = model.to(device)
        model.eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        # Use the same rescore function (works for both)
        rescore_fn = rescore_with_llama

    return model, tokenizer, rescore_fn


if __name__ == "__main__":
    # Test the module
    print("Testing Llama rescorer...")

    # Test with a small model for quick verification
    test_hypotheses = [
        "hello how are you today",
        "hello how are you doing",
        "hello how were you today",
    ]

    try:
        # Try to load a small model for testing
        model, tokenizer = build_llama(
            model_name='facebook/opt-125m',  # Small model for testing
            use_4bit=False,
        )

        device = next(model.parameters()).device
        scores = rescore_with_llama(model, tokenizer, str(device), test_hypotheses, 0.0)

        print(f"Test hypotheses: {test_hypotheses}")
        print(f"Scores: {scores}")
        print("Test passed!")

    except Exception as e:
        print(f"Test failed (expected if dependencies missing): {e}")
