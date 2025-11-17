# ManchuAI-OCR

Optical Character Recognition for Manchu script using multiple model architectures.

## Experimental Environment

- **CPU**: Intel Core i9-13900KS (32 cores)
- **GPU**: NVIDIA RTX 6000 Ada Generation (49GB VRAM)
- **RAM**: 188GB

## Installation

```bash
uv sync
```

## Quick Start

The `scripts/` folder contains the main entry points:

### 1. Train Models

```bash
python scripts/train.py
```

Trains all models by default.

```bash
python scripts/train.py --target_model llama-32-11b
```

Train a specific model.

```bash
python scripts/train.py --target_model llama-32-11b qwen-25-3b
```

Train multiple models (space-separated).

### 2. Evaluate Models

```bash
python scripts/evaluate.py
```

Evaluates all models by default.

```bash
python scripts/evaluate.py --target_model llama-32-11b
```

Evaluate a specific model.

```bash
python scripts/evaluate.py --target_model llama-32-11b qwen-25-3b
```

Evaluate multiple models (space-separated).

### 3. Visualize Results

```bash
python scripts/visualize.py
```

Generates 11 publication-ready visualizations and saves them to `results/paper/figures/`:

1. Accuracy bar plot - Model comparison
2. Character Error Rate (CER) comparison
3. F1 score comparison
4. Inference time analysis
5. Performance comparison chart (VLM models)
6. Checkpoint evaluation trends
7. Word length performance analysis
8. VLM vs CRNN comparison
9. Character error heatmap
10. Model training curves
11. Training stability (gradient norms)

```bash
# Custom output directory
python scripts/visualize.py --output ./my_figures

# Custom metrics directory
python scripts/visualize.py --metrics ./custom_metrics
```

Run individual visualizations:

```bash
python scripts/figures/checkpoint_trends.py
python scripts/figures/comparison_vlm_vs_crnn.py
```

Output: PNG (300 DPI) and PDF formats with Times New Roman fonts.

## Configuration

All configuration files are located in `configs/` directory and use YAML format with hierarchical structure:
- `default`: Base settings for all models
- Model-specific sections: Override defaults (e.g., `qwen-25-3b`, `llama-32-11b`, `crnn-base-3m`)

### 1. Training (`configs/training.yaml`)

```bash
# Basic usage - uses default config
python scripts/train.py
```

Key parameters:

```yaml
default:
  training:
    num_train_epochs: 5
    learning_rate: 2.0e-4
    per_device_train_batch_size: 4
    warmup_steps: 100
    save_steps: 1000                    # Checkpoint frequency
    eval_steps: 1000                    # Evaluation frequency
    metric_for_best_model: "manchu_cer" # Best model selection metric

  peft:                                 # LoRA settings (VLM only)
    r: 32
    lora_alpha: 64
    lora_dropout: 0.05
```

Override for specific models:

```yaml
llama-32-11b:
  training:
    learning_rate: 1.0e-4  # Lower learning rate for larger model
    num_train_epochs: 5

crnn-base-3m:
  training:
    num_train_epochs: 100
    batch_size: 16
    learning_rate: 1e-3
```

### 2. Evaluation (`configs/evaluation.yaml`)

```bash
# Basic usage - uses default config
python scripts/evaluate.py
```

Key parameters:

```yaml
default:
  validation:
    num_samples: 1000
    step_num: best  # Options: "best", "latest", or step number (e.g., 21000)
  test:
    num_samples: 753
    step_num: best
```

Checkpoint selection:
- `best` - Best checkpoint by `metric_for_best_model`
- `latest` - Most recent checkpoint
- `<number>` - Specific step (e.g., `21000` → checkpoint-21000)

Override for specific checkpoints:

```yaml
qwen-25-3b:
  validation:
    num_samples: 1000
    step_num: 106000  # Evaluate specific checkpoint
  test:
    num_samples: 218
    step_num: 106000
```

### 3. Data (`configs/data.yaml`)

Dataset configuration and preprocessing settings:

```yaml
default:
  dataset_name: mic7ch/manchu_sub2_new_12Nov
  train_split: train
  val_split: validation
  test_split: test
  image_key: im
  text_key: [roman, manchu]
  cache: false  # Set to true to cache downloaded data
```

### 4. Models (`configs/models.yaml`)

Model definitions and base model mappings:

```yaml
models:
  - name: qwen-25-3b
    base_model: unsloth/Qwen2.5-VL-3B-Instruct
    model_class: VLM
  - name: llama-32-11b
    base_model: unsloth/Llama-3.2-11B-Vision-Instruct
    model_class: VLM
  - name: crnn-base-3m
    base_model: crnn
    model_class: CRNN
```

## Models

### Vision Language Models

- **qwen-25-3b/7b**: Qwen2.5-VL-3B/7B
- **llama-32-11b**: Llama-3.2-11B

### CRNN Models

- **crnn-base-3m**: Convolutional Recurrent Neural Network

## Results

Evaluation results are saved in `results/` directory.
