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

Trains VLM and CRNN models on Manchu OCR datasets.

### 2. Evaluate Models

```bash
python scripts/evaluate.py
```

Evaluates trained models on validation and test datasets.

### 3. Generate Figures

```bash
python scripts/generate_figures.py
```

Creates performance comparison charts and analysis figures.

## Models

### Vision Language Models

- **qwen-25-3b/7b**: Qwen2.5-VL-3B/7B
- **llama-32-11b**: Llama-3.2-11B

### CRNN Models

- **crnn-base-3m**: Convolutional Recurrent Neural Network

### Closed Domain Models

- **openai-41**: OpenAI GPT-4.1-2025-04-14

## Results

Results are saved in `results/` directory.
