#!/usr/bin/env python3
"""
Evaluate Qwen model on persona-based dialogue summarization.

This script loads a Qwen model and evaluates it on dialogue persona generation,
computing metrics like BLEU, ROUGE, CHRF, and LaBSE similarity.
Results are saved to a CSV file with model name and data count.
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm
from sacrebleu.metrics import CHRF
import numpy as np

# Model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = "./data"

# Generation configuration
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
DO_SAMPLE = True
NUM_BEAMS = 1


def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)


def load_dialogues(data_dir: str, split: str = "test", samples: int = -1) -> List[Dict]:
    """Load dialogues dataset from JSON file."""
    filename = f"dialogues_{split}.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if samples >= 0:
        data = data[:samples]
        print(f"Loaded {len(data)} examples from {filename} (limited to {samples})")
    else:
        print(f"Loaded {len(data)} examples from {filename}")
    return data


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Try to load as Seq2Seq model (for T5, BART, etc.)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model.model_type = "seq2seq"
    except ValueError:
        # Fall back to CausalLM model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model.model_type = "causal"

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def build_dialogue_text(dialogue: List[Dict]) -> str:
    """Build dialogue text from messages.

    Args:
        dialogue: List of message dicts with 'speaker' and 'text'

    Returns:
        Formatted dialogue text
    """
    messages = []
    for msg in dialogue:
        speaker_label = "Пользователь 1" if msg['speaker'] == 1 else "Пользователь 2"
        messages.append(f"{speaker_label}: {msg['text']}")
    return "\n".join(messages)


def build_dialogue_text_batch(dialogues: List[Dict]) -> str:
    """Build dialogue texts from multiple dialogues for batching."""
    texts = []
    for dialogue in dialogues:
        texts.append(build_dialogue_text(dialogue))
    return texts


def generate_persona(
    model,
    tokenizer,
    dialogue: List[Dict],
    target_speaker: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Generate a persona description from dialogue using few-shot prompting."""

    dialogue_text = build_dialogue_text(dialogue)

    # Few-shot examples in Russian
    examples = """Ты - ассистент для описания личности человека на основе диалога. Опиши личность указанного участника в виде списка фактов.

Пример 1:
Диалог:
Пользователь 1: Привет! Работаю учителем.
Пользователь 2: Привет! А какие предметы?
Пользователь 1: Математику и физику.
Пользователь 2: Круто! У меня собака.
Пользователь 1: А у меня две дочки.

Опиши личность Пользователя 1:
- Работает учителем (математика и физика)
- Есть две дочери

Пример 2:
Диалог:
Пользователь 1: Привет! Люблю путешествовать.
Пользователь 2: Куда ездил?
Пользователь 1: В Турцию, в Египет.
Пользователь 2: Я люблю готовить, я повар.
Пользователь 1: У меня есть собака.
Пользователь 2: У меня подруга подарила котенка.

Опиши личность Пользователя 2:
- Любит готовить
- Работает поваром
- Есть подруга

Диалог:
{dialogue_text}

Опиши личность Пользователя {target_speaker}:"""

    prompt = examples.format(dialogue_text=dialogue_text, target_speaker=target_speaker)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Different generation params for seq2seq vs causal models
        if getattr(model, "model_type", None) == "seq2seq":
            # T5/BART style
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )
            # Decode the generated output
            predicted_persona = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Causal model style
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_persona = full_output.split(f"Опиши личность Пользователя {target_speaker}:")[-1].strip()

    # Clean up: stop at newlines that suggest the model is continuing
    predicted_persona = predicted_persona.split("\n\n")[0].strip()
    predicted_persona = predicted_persona.split("Диалог:")[0].strip()

    return predicted_persona


def compute_bleu_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute BLEU score between references and hypotheses.

    Uses standard BLEU-4 with equal weights (0.25, 0.25, 0.25, 0.25).
    """
    smoothing = SmoothingFunction()

    # Tokenize references and hypotheses
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    hyp_tokens = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]

    # Calculate BLEU-4 with equal weights
    try:
        score = corpus_bleu(
            [[ref] for ref in ref_tokens],
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing.method1,
        )
        bleu_score = score * 100
    except Exception:
        bleu_score = 0.0

    return {"BLEU": bleu_score}


def compute_chrf_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute CHRF score between references and hypotheses."""
    chrf = CHRF()

    # Compute CHRF score - pass lists directly
    result = chrf.corpus_score(hypotheses, [references])

    return {
        "CHRF": result.score,
        "CHRF+": result.score,  # CHRF+ is same as CHRF for single references
    }


def build_dialogue_text(dialogue: List[Dict]) -> str:
    """Build dialogue text from messages."""
    messages = []
    for msg in dialogue:
        speaker_label = "Пользователь 1" if msg['speaker'] == 1 else "Пользователь 2"
        messages.append(f"{speaker_label}: {msg['text']}")
    return "\n".join(messages)


def generate_personas_batched(model, tokenizer, dialogues: List[Dict], target_speaker: int,
                                   is_seq2seq: bool,
                                   max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE, batch_size: int = 1):
    """Generate persona descriptions from dialogues in batches."""

    # Few-shot examples in Russian
    examples = """Ты - ассистент для описания личности человека на основе диалога. Опиши личность указанного участника в виде списка фактов.

Пример 1:
Диалог:
Пользователь 1: Привет! Работаю учителем.
Пользователь 2: Привет! А какие предметы?
Пользователь 1: Математику и физику.
Пользователь 2: Круто! У меня собака.
Пользователь 1: А у меня две дочки.

Опиши личность Пользователя 1:
- Работает учителем (математика и физика)
- Есть две дочери

Пример 2:
Диалог:
Пользователь 1: Привет! Люблю путешествовать.
Пользователь 2: Куда ездил?
Пользователь 1: В Турцию, в Египет.
Пользователь 2: Я люблю готовить, я повар.
Пользователь 1: У меня есть собака.
Пользователь 2: У меня подруга подарила котенка.

Опиши личность Пользователя 2:
- Любит готовить
- Работает поваром
- Есть подруга

"""

    if is_seq2seq:
        # Seq2Seq: dialogue -> persona
        prompts = [f"{examples}Диалог:\n{dialogue_text}\n\nОпиши личность Пользователя {target_speaker}:" for dialogue_text in build_dialogue_text_batch(dialogues)]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        # Causal LM
        prompts = [f"{examples}Диалог:\n{dialogue_text}\n\nОпиши личность Пользователя {target_speaker}:" for dialogue_text in build_dialogue_text_batch(dialogues)]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Extract continuation after the prompt
        results = []
        for output in full_outputs:
            if f"Опиши личность Пользователя {target_speaker}:" in output:
                result = output.split(f"Опиши личность Пользователя {target_speaker}:")[-1].strip()
            else:
                result = output
            # Clean up
            result = result.split("\n\n")[0].strip()
            result = result.split("Диалог:")[0].strip()
            results.append(result)

    return results

    return results


def compute_labse_similarity(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute LaBSE similarity between references and hypotheses.

    LaBSE (Language-agnostic BERT Sentence Embedding) uses multilingual sentence
    embeddings to compute cosine similarity.
    """
    print("Loading LaBSE model...")
    labse_model_name = "sentence-transformers/LaBSE"
    labse_tokenizer = AutoTokenizer.from_pretrained(labse_model_name)
    labse_model = AutoModel.from_pretrained(labse_model_name)

    # Move to same device as the main model
    labse_device = torch.device("cpu")  # Use CPU to avoid MPS memory issues

    labse_model = labse_model.to(labse_device)
    labse_model.eval()

    similarities = []

    print("Computing LaBSE similarities...")
    with torch.no_grad():
        for ref, hyp in zip(references, hypotheses):
            # Tokenize
            ref_inputs = labse_tokenizer(ref, return_tensors="pt", padding=True, truncation=True, max_length=128)
            hyp_inputs = labse_tokenizer(hyp, return_tensors="pt", padding=True, truncation=True, max_length=128)

            # Move to device
            ref_inputs = {k: v.to(labse_device) for k, v in ref_inputs.items()}
            hyp_inputs = {k: v.to(labse_device) for k, v in hyp_inputs.items()}

            # Get embeddings
            ref_outputs = labse_model(**ref_inputs)
            hyp_outputs = labse_model(**hyp_inputs)

            # Use mean pooling
            ref_embedding = ref_outputs.last_hidden_state.mean(dim=1)
            hyp_embedding = hyp_outputs.last_hidden_state.mean(dim=1)

            # Compute cosine similarity
            similarity = F.cosine_similarity(ref_embedding, hyp_embedding)
            similarities.append(similarity.item())

    # Clean up
    del labse_model
    del labse_tokenizer

    return {
        "LaBSE-Similarity": sum(similarities) / len(similarities) * 100,
    }


def compute_rouge_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores between references and hypotheses."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores) * 100,
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores) * 100,
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores) * 100,
    }


def evaluate_model(
    model_path: str,
    num_samples: int = None,
    output_file: str = None,
    batch_size: int = 1,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
    target_speaker: int = 1,
    split: str = "test",
):
    """
    Evaluate model on dialogue persona generation.

    Args:
        model_path: Path to the model (can be local or HuggingFace hub)
        num_samples: Number of samples to evaluate (None for all)
        output_file: Path to save predictions (None for no saving)
        batch_size: Batch size for evaluation (for future batch generation support)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (vs greedy decoding)
        target_speaker: Which speaker to generate persona for (1 or 2)
        split: Dataset split to use (train, val, or test)
    """
    print("=" * 80)
    print("Starting evaluation...")
    print("=" * 80)

    # Download NLTK data
    download_nltk_data()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load dialogues
    dialogues = load_dialogues(DATA_DIR, split, num_samples)

    # Limit samples if specified
    if num_samples is not None:
        print(f"Evaluating on {num_samples} samples")
    else:
        print(f"Evaluating on all {len(dialogues)} samples")

    print(f"Target speaker: {target_speaker}")

    # Generate predictions with batching
    print("\nGenerating predictions...")
    predictions = []
    references = []

    # Batch size for generation
    batch_size = 16
    num_samples = len(dialogues)
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Generating {num_samples} predictions in {num_batches} batches (size={batch_size})...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = dialogues[start_idx:end_idx]

        # Prepare batch inputs
        batch_dialogues = [item.get("dialogue", []) for item in batch_data]

        # Generate in batch
        batch_preds = generate_personas_batched(
            model, tokenizer, batch_dialogues, target_speaker, is_seq2seq, max_new_tokens, temperature, batch_size
        )

        # Store results
        for item, pred in zip(batch_data, batch_preds):
            if target_speaker == 1:
                reference_persona = item.get("persona_1", "")
            else:
                reference_persona = item.get("persona_2", "")

            if reference_persona:
                predictions.append({
                    "id": item.get("id", ""),
                    "target_speaker": target_speaker,
                    "reference": reference_persona,
                    "prediction": pred,
                })
                references.append(reference_persona)

        print(f"Processed batch {batch_idx + 1}/{num_batches} ({end_idx}/{num_samples} samples)")

    # Compute metrics
    print("\nComputing evaluation metrics...")
    hypotheses = [p["prediction"] for p in predictions]

    # BLEU scores
    bleu_scores = compute_bleu_score(references, hypotheses)

    # ROUGE scores
    rouge_scores = compute_rouge_score(references, hypotheses)

    # CHRF scores
    chrf_scores = compute_chrf_score(references, hypotheses)

    # LaBSE similarity
    labse_scores = compute_labse_similarity(references, hypotheses)

    # Get model name (use path basename for local models)
    model_name = os.path.basename(model_path) if os.path.exists(model_path) else model_path
    data_count = len(predictions)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Target Speaker: {target_speaker}")
    print(f"Total samples evaluated: {data_count}")

    print("\nBLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nCHRF Scores:")
    for metric, score in chrf_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nLaBSE Similarity:")
    for metric, score in labse_scores.items():
        print(f"  {metric}: {score:.2f}")
    print("=" * 80)

    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 80)
    for i, pred in enumerate(predictions[:10]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference Persona {pred['target_speaker']}: {pred['reference']}")
        print(f"Prediction: {pred['prediction']}")

    # Save predictions if output file specified
    if output_file:
        print(f"\nSaving predictions to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print("Predictions saved!")

    # Save evaluation results to CSV (append mode)
    csv_file = "persona_evaluation_results.csv"
    csv_exists = os.path.exists(csv_file)

    # Flatten all metrics into a single dict
    all_metrics = {
        "model_name": model_name,
        "target_speaker": target_speaker,
        "split": split,
        "data_count": data_count,
        **bleu_scores,
        **rouge_scores,
        **chrf_scores,
        **labse_scores,
    }

    print(f"\nSaving evaluation results to {csv_file}...")
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(all_metrics)
    print(f"Evaluation results saved to {csv_file}!")

    # Return metrics
    return {
        "bleu": bleu_scores,
        "rouge": rouge_scores,
        "chrf": chrf_scores,
        "labse": labse_scores,
        "total_samples": len(predictions),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on persona generation from dialogues"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Path to model (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save predictions JSON (default: no saving)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )
    parser.add_argument(
        "--target-speaker",
        type=int,
        default=1,
        choices=[1, 2],
        help="Target speaker to generate persona for (1 or 2, default: 1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use (default: test)",
    )

    args = parser.parse_args()

    # Run evaluation with command-line arguments
    evaluate_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
        target_speaker=args.target_speaker,
        split=args.split,
    )


if __name__ == "__main__":
    main()
