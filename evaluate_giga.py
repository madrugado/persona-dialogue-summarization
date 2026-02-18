#!/usr/bin/env python3
"""
Evaluate GigaChat API on persona-based dialogue summarization.

This script:
1. Queries GigaChat API to generate persona descriptions from dialogues
2. Supports generating personas for either participant 1 or 2
3. Evaluates generated personas against ground truth using BLEU, ROUGE, CHRF, LaBSE
"""

import argparse
import json
import os
from typing import List, Dict
import numpy as np

from sacrebleu.metrics import BLEU, CHRF
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from gigachat import GigaChat, Chat, Messages, MessagesRole


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


def query_giga_api(prompt: str, api_key: str, model: str = "GigaChat-Max", base_url: str = "https://api.giga.chat") -> str:
    """Query GigaChat API to generate persona description."""
    giga = GigaChat(
        model=model,
        credentials=api_key,
        scope="GIGACHAT_API_PERS",
    )

    response = giga.chat(
        Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content="Ты - виртуальный ассистент. Опиши личность человека на основе его диалога."
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=prompt
                ),
            ]
        )
    )

    if response and response.choices and len(response.choices) > 0:
        return response.choices[0].message.content.strip()
    else:
        return ""


def query_giga_and_save(
    dialogues: List[Dict],
    api_key: str,
    output_file: str,
    model: str = "GigaChat-Max"
) -> None:
    """Query GigaChat API for all dialogues and save responses."""
    responses = []

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

    for i, item in enumerate(dialogues):
        item_id = item.get('id', i)
        dialogue_text = build_dialogue_text(item.get('dialogue', []))

        if not dialogue_text:
            responses.append({"id": item_id, "response": ""})
            continue

        # Create prompt for GigaChat
        prompt = examples + f"""На основе следующего диалога опиши личность Пользователя 2.

{dialogue_text}

Опиши личность Пользователя 2 в виде списка фактов о нём:"""

        try:
            print(f"Querying GigaChat API for item {i+1}/{len(dialogues)}...")
            giga_response = query_giga_api(prompt, api_key, model)
            responses.append({"id": item_id, "response": giga_response})
            print(f"  Response: {giga_response[:100]}...")
        except Exception as e:
            print(f"  Error for item {i+1}: {e}")
            responses.append({"id": item_id, "response": f"ERROR: {str(e)}"})

    # Save responses
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(responses)} responses to {output_file}")
    return responses


def evaluate_giga_responses(dialogues: List[Dict], giga_file: str) -> List[Dict]:
    """Load saved GigaChat API responses and prepare for evaluation."""
    with open(giga_file, "r", encoding="utf-8") as f:
        giga_responses = json.load(f)

    print(f"Loaded {len(giga_responses)} GigaChat responses")

    # Match Giga responses with dialogues by ID
    giga_dict = {item.get('id'): item.get('response', '') for item in giga_responses}

    predictions = []
    references = []

    for item in dialogues:
        item_id = item.get('id', '')

        # Always use persona_2 as target
        reference_persona = item.get('persona_2', '')

        if not reference_persona:
            continue

        giga_response = giga_dict.get(item_id, '')
        predictions.append(giga_response)
        references.append(reference_persona)

    return predictions, references


def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute BLEU, ROUGE, CHRF, and LaBSE similarity scores."""
    print("Computing BLEU...")
    bleu_metric = BLEU()
    bleu_result = bleu_metric.corpus_score(hypotheses, [references])

    print("Computing ROUGE...")
    rouge_scorer_instance = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for ref, hyp in zip(references, hypotheses):
        scores = rouge_scorer_instance.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    print("Computing CHRF...")
    chrf_metric = CHRF()
    chrf_result = chrf_metric.corpus_score(hypotheses, [references])

    # LaBSE similarity
    print("Computing LaBSE similarity...")
    labse_model = SentenceTransformer('sentence-transformers/LaBSE')
    ref_embeddings = labse_model.encode(references, convert_to_tensor=True)
    hyp_embeddings = labse_model.encode(hypotheses, convert_to_tensor=True)
    similarities = util.cos_sim(hyp_embeddings, ref_embeddings)
    similarity_scores = torch.diagonal(similarities).cpu().numpy()

    return {
        'BLEU': bleu_result.score,
        'ROUGE-1': np.mean(rouge1_scores) * 100,
        'ROUGE-2': np.mean(rouge2_scores) * 100,
        'ROUGE-L': np.mean(rougeL_scores) * 100,
        'CHRF': chrf_result.score,
        'LaBSE-Similarity': np.mean(similarity_scores) * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query GigaChat API for persona generation and evaluate responses"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory containing dialogues_*.json",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="GigaChat API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GigaChat-Max",
        help="GigaChat model name (default: GigaChat-Max)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of samples to process (default: 2, use -1 for full dataset)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="giga_personas_responses.json",
        help="Output file for API responses",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="giga_personas_evaluation_results.json",
        help="Output file for evaluation results",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing responses file, don't query API",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="giga_personas_responses.json",
        help="Input file with API responses to evaluate (used with --eval-only)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--target-speaker",
        type=int,
        default=2,
        choices=[1, 2],
        help="Target speaker to generate persona for (1 or 2, default: 2)",
    )

    args = parser.parse_args()

    # Load dialogues
    dialogues = load_dialogues(args.data_dir, args.split, args.samples)

    # Determine input file for responses
    giga_file = os.path.join(args.data_dir, args.input_file if args.eval_only else args.output_file)

    if args.eval_only:
        print(f"Evaluation only mode: using existing responses from {giga_file}")
        if not os.path.exists(giga_file):
            print(f"Error: Response file not found: {giga_file}")
            return
    else:
        # Query GigaChat API and save responses
        responses = query_giga_and_save(dialogues, args.api_key, giga_file, args.model)

        if len(responses) == 0:
            print("No responses to evaluate!")
            return

    # Evaluate saved responses
    predictions, references = evaluate_giga_responses(dialogues, giga_file)

    if len(predictions) == 0:
        print("No predictions to evaluate!")
        return

    # Compute metrics
    print("\nComputing evaluation metrics...")
    print("=" * 80)
    metrics = compute_metrics(references, predictions)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Target Speaker: 2")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 80)

    # Save results - convert numpy types to Python types
    results = {
        'model': f'GigaChat-{args.model}',
        'target_speaker': 2,
        'split': args.split,
        'total_samples': len(references),
        **{k: float(v) for k, v in metrics.items()},
    }

    eval_output_path = os.path.join(args.data_dir, args.eval_file)
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {eval_output_path}")

    # Show sample predictions
    print("\nSample predictions (first 5):")
    print("-" * 80)
    for i, item in enumerate(dialogues[:5]):
        reference = item.get('persona_2', '')
        predicted = predictions[i] if i < len(predictions) else "N/A"
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference Persona 2: {reference}")
        print(f"Predicted: {predicted}")


if __name__ == "__main__":
    main()
