# ManchuAI-OCR

Optical Character Recognition for Manchu script using Vision Language Models (VLMs) and CRNN architectures.

## Experimental Environment

- **CPU**: Intel Core i9-13900KS (32 cores)
- **GPU**: NVIDIA RTX 6000 Ada Generation (49GB VRAM)
- **RAM**: 188GB

## Prerequisites

- **NVIDIA GPU** with CUDA support (tested with RTX 6000 Ada)
- **CUDA Toolkit 12.x** (tested with CUDA 12.4)
- **cuDNN** compatible with your CUDA version
- **Python 3.10+**

Verify your CUDA installation:

```bash
nvcc --version
nvidia-smi
```

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Quick Start

### 1. Train Models

```bash
# Train all models
python scripts/train.py

# Train a specific model
python scripts/train.py --target_model llama-32-11b

# Train multiple models
python scripts/train.py --target_model llama-32-11b qwen-25-3b
```

### 2. Evaluate Models

```bash
# Evaluate best checkpoint for all models
python scripts/evaluate_best.py

# Evaluate specific model
python scripts/evaluate_best.py --target_model llama-32-11b

# Evaluate all checkpoints (for trend analysis)
python scripts/evaluate_checkpoints.py
```

### 3. Generate Paper Assets (Optional)

```bash
# Generate all figures and tables
python scripts/paper/generate.py
```

Outputs publication-ready figures (PNG 300 DPI + PDF) and LaTeX tables to `paper/figures/` and `paper/tables/`.

### 4. Download Pre-trained Models (Optional)

Skip training by downloading our pre-trained models:

```bash
# Download all models
python scripts/download_models.py

# Download specific model(s)
python scripts/download_models.py --model llama-32-11b
python scripts/download_models.py --model qwen-25-3b qwen-25-7b

# Force re-download
python scripts/download_models.py --model llama-32-11b --force
```

Available models:
| Model | Hugging Face | Parameters |
|-------|--------------|------------|
| llama-32-11b | [mic7ch/manchu-ocr-llama-32-11b](https://huggingface.co/mic7ch/manchu-ocr-llama-32-11b) | 11B |
| qwen-25-7b | [mic7ch/manchu-ocr-qwen-25-7b](https://huggingface.co/mic7ch/manchu-ocr-qwen-25-7b) | 7B |
| qwen-25-3b | [mic7ch/manchu-ocr-qwen-25-3b](https://huggingface.co/mic7ch/manchu-ocr-qwen-25-3b) | 3B |
| crnn-base-3m | [mic7ch/manchu-ocr-crnn-base-3m](https://huggingface.co/mic7ch/manchu-ocr-crnn-base-3m) | 3M |

## Project Structure

```
ManchuAI-OCR/
├── configs/                          # Configuration files
│   ├── base.yaml                     # Dataset and model definitions
│   ├── training.yaml                 # Training hyperparameters
│   └── evaluation.yaml               # Evaluation settings
├── src/                              # Source code
│   ├── CRNN/                         # CRNN model implementation
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── inference.py
│   ├── evaluation/                   # Evaluation framework
│   │   ├── vlm_evaluator.py
│   │   ├── crnn_evaluator.py
│   │   ├── metrics.py                # CER, WA, F1, etc.
│   │   └── utils.py
│   ├── training/                     # Training wrappers
│   │   ├── vlm_trainer.py
│   │   └── crnn_trainer.py
│   └── utils/                        # Shared utilities
│       ├── config.py                 # ConfigLoader
│       ├── dataset.py                # Dataset loading
│       └── files.py                  # File I/O helpers
├── scripts/                          # Entry points
│   ├── train.py                      # Model training
│   ├── evaluate_best.py              # Best checkpoint evaluation
│   ├── evaluate_checkpoints.py       # All checkpoints evaluation
│   ├── internal/                     # Shell scripts
│   └── paper/                        # Paper generation
│       ├── generate.py               # Main generator
│       ├── utils.py                  # Shared utilities
│       ├── figures/                  # Figure generators
│       └── tables/                   # Table generators
├── models/                           # Trained model checkpoints
│   └── VLM/{model}/
├── results/                          # Evaluation outputs
│   ├── metrics/{model}/              # Aggregated metrics
│   │   ├── best_checkpoint/
│   │   │   ├── validation.json
│   │   │   └── test.json
│   │   └── checkpoint-{step}_{split}.json
│   └── predictions/{model}/          # Per-sample predictions
│       ├── best_checkpoint/
│       └── checkpoint-{step}_{split}.json
├── paper/                            # Generated paper assets
│   ├── figures/                      # PNG + PDF figures
│   └── tables/                       # LaTeX tables
└── data/                             # Local data cache
```

## Configuration

### Base Configuration (`configs/base.yaml`)

Dataset and model definitions:

```yaml
data:
  dataset_name: mic7ch/manchu
  train_split: train
  val_split: validation
  test_split: test

models:
  - name: qwen-25-3b
    base_model: unsloth/Qwen2.5-VL-3B-Instruct
    model_class: VLM
  - name: qwen-25-7b
    base_model: unsloth/Qwen2.5-VL-7B-Instruct
    model_class: VLM
  - name: llama-32-11b
    base_model: unsloth/Llama-3.2-11B-Vision-Instruct
    model_class: VLM
  - name: crnn-base-3m
    base_model: crnn
    model_class: CRNN
```

### Training (`configs/training.yaml`)

Default settings with model-specific overrides:

```yaml
default:
  training:
    num_train_epochs: 6
    learning_rate: 2.0e-4
    per_device_train_batch_size: 4
    save_steps: 1000
    eval_steps: 1000
    metric_for_best_model: "manchu_cer"
  peft: # LoRA (VLM only)
    r: 32
    lora_alpha: 64
    lora_dropout: 0.05

llama-32-11b:
  training:
    learning_rate: 1.0e-4
    num_train_epochs: 5

crnn-base-3m:
  training:
    num_train_epochs: 100
    batch_size: 16
    learning_rate: 1e-3
```

### Evaluation (`configs/evaluation.yaml`)

```yaml
default:
  validation:
    num_samples: 1000
    step_num: best # "best", "latest", or step number
  test:
    num_samples: 753
    step_num: best

checkpoints:
  num_samples: 1000
  models: [llama-32-11b, qwen-25-3b, qwen-25-7b, crnn-base-3m]
```

## Models

### Vision Language Models (VLM)

- **LLaMA-3.2-VL-11B**: `unsloth/Llama-3.2-11B-Vision-Instruct`
- **Qwen-2.5-VL-7B**: `unsloth/Qwen2.5-VL-7B-Instruct`
- **Qwen-2.5-VL-3B**: `unsloth/Qwen2.5-VL-3B-Instruct`

### CRNN

- **CRNN-3M**: Custom CNN + 4-layer BiLSTM architecture
