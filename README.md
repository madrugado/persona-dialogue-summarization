# Persona Generation from Dialogues

Framework for fine-tuning and evaluating models on persona generation from dialogues.

## Features

- **Model-agnostic**: Automatically detects and supports both seq2seq (RuT5, T5, BART) and causal LM (Qwen, Llama, Mistral) architectures
- **Two participants**: Generate persona descriptions for either speaker 1 or speaker 2
- **Multiple metrics**: BLEU, ROUGE, CHRF, LaBSE similarity
- **Data preprocessing**: Clean HTML tags from dialogues and split into train/val/test (80:10:10)
- **Easy to use**: Simple CLI interface with all common training parameters
- **Colab-ready**: Full notebook for Google Colab evaluation

## Data Preparation

The `prepare_data.py` script processes raw dialogues with HTML tags:

```bash
python prepare_data.py --input ./data/dialogues.tsv --output-dir ./data
```

This creates:
- `data/dialogues_train.json` - 80% for training
- `data/dialogues_val.json` - 10% for validation
- `data/dialogues_test.json` - 10% for testing

### Data Format

Each dialogue contains:
- `id` - Unique identifier
- `persona_1` - Persona description for speaker 1 (ground truth)
- `persona_2` - Persona description for speaker 2 (ground truth)
- `dialogue` - List of messages with `speaker` (1 or 2) and `text`

## Evaluation

### GigaChat API Evaluation

Evaluate using GigaChat API:

```bash
# Evaluate speaker 1 on test set
python evaluate_giga.py \
    --api-key YOUR_API_KEY \
    --target-speaker 1 \
    --split test

# Evaluate speaker 2
python evaluate_giga.py \
    --api-key YOUR_API_KEY \
    --target-speaker 2 \
    --split test

# Evaluate on validation set with 10 samples
python evaluate_giga.py \
    --api-key YOUR_API_KEY \
    --num-samples 10 \
    --split val
```

### Local Model Evaluation

Evaluate using local or HuggingFace models:

```bash
# Evaluate Qwen on speaker 1
python evaluate_qwen.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --target-speaker 1 \
    --split test

# Evaluate with 10 samples for quick test
python evaluate_qwen.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --num-samples 10 \
    --target-speaker 2

# Evaluate without sampling
python evaluate_qwen.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --no-sample \
    --target-speaker 1
```

## Google Colab

For easy evaluation without local GPU, use the provided notebook:

1. Open `evaluate_models.ipynb` in Google Colab
2. Run cells sequentially
3. Configure model and parameters
4. Run evaluation

The notebook includes:
- Dependency installation
- GPU detection and memory monitoring
- Model loading with automatic type detection
- Batched persona generation
- Evaluation with multiple metrics
- Model comparison

## Evaluation Metrics

The evaluation scripts compute:

- **BLEU** - n-gram precision score
- **ROUGE-1, ROUGE-2, ROUGE-L** - Recall-oriented metrics
- **CHRF** - Character-level F-score
- **LaBSE Similarity** - Semantic similarity using LaBSE embeddings

Results are automatically saved to `persona_evaluation_results.csv`.

## Examples

### Evaluate GigaChat

```bash
python evaluate_giga.py \
    --api-key $GIGACHAT_API_KEY \
    --target-speaker 1 \
    --split test
```

### Evaluate Qwen Model

```bash
python evaluate_qwen.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --target-speaker 1 \
    --split test
```

### Quick Test on 5 Samples

```bash
python evaluate_qwen.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --num-samples 5 \
    --target-speaker 1
```

## Supported Model Types

### Seq2Seq Models (Encoder-Decoder)
- `cointegrated/rut5-base` - Russian T5 (recommended)
- `cointegrated/rut5-small` - Smaller Russian T5
- `google/flan-t5-base` - English T5
- `facebook/bart-base` - BART
- Any T5/BART variant from HuggingFace Hub

### Causal LMs (Decoder-Only)
- `Qwen/Qwen2.5-0.5B-Instruct` - Qwen
- `Qwen/Qwen2.5-1.5B-Instruct` - Qwen
- `meta-llama/Llama-3.2-1B-Instruct` - Llama
- Any decoder-only model from HuggingFace Hub

## Installation

```bash
# Clone repository
git clone git@github.com:madrugado/persona-dialogue-summarization.git
cd persona-dialogue-summarization

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- Rouge-score
- SacreBLEU
- sentence-transformers (for LaBSE)
- GigaChat SDK (for API evaluation)

## License

This project uses dialogues dataset and pre-trained models. Please check the respective licenses for each component.

## Remote Repository

```bash
git remote add origin git@github.com:madrugado/persona-dialogue-summarization.git
```

## License

This project uses dialogues dataset and pre-trained models. Please check the respective licenses for each component.
