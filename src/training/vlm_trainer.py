import glob
import os
import numpy as np
import torch
import re
import random
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from src.evaluation.metrics import calculate_cer

from src.utils.files import create_dir


def train_vlm_model(
    model_config,
    training_config,
    training_output,
    converted_train_data,
    converted_val_data,
):

    base_model = model_config.get("base_model")
    model_name = model_config.get("name")

    loading_config = training_config.get("loading", {})
    peft_config = training_config.get("peft", {})
    sft_config = training_config.get("training", {}).copy()

    model, tokenizer = load_model_and_tokenizer(base_model, loading_config, peft_config)

    training_output_subdir = training_output / "checkpoints"
    create_dir(training_output_subdir)

    print(f"Training {model_name}...")
    print(f"Training output: {training_output_subdir}")

    FastVisionModel.for_training(model)

    # TODO: make this configurable.
    num_eval_samples = 400
    random.seed(42)
    random.shuffle(converted_val_data)
    eval_dataset = converted_val_data[:num_eval_samples]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max"),
        train_dataset=converted_train_data,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_manchu_cer(eval_pred, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits,
        args=SFTConfig(
            **sft_config,
            output_dir=training_output_subdir,
            run_name=f"{model_name}_training",
        ),
    )

    checkpoints = glob.glob(os.path.join(training_output_subdir, "checkpoint-*"))
    recent_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    if recent_checkpoint:
        print(f"Resuming from checkpoint {recent_checkpoint}")
    else:
        print("Starting training from scratch")

    try:
        trainer.train(resume_from_checkpoint=recent_checkpoint)
        print(f"VLM training finished for {model_name}.")
        best_model_path = training_output / "best_model"
        trainer.save_model(str(best_model_path))
        print(f"Best model saved to: {best_model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        return


def load_model_and_tokenizer(base_model_name, loading_config=None, peft_config=None):
    if loading_config is None:
        loading_config = {}
    if peft_config is None:
        peft_config = {}

    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_name, **loading_config
    )

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastVisionModel.get_peft_model(model, **peft_config)

    return model, tokenizer


def compute_manchu_cer(eval_pred, tokenizer):
    MANCHU_RE = re.compile(r"manchu\s*:\s*(.*)", flags=re.I)
    extract_manchu = lambda text: next(
        (
            m.group(1).strip()
            for line in text.splitlines()
            if (m := MANCHU_RE.match(line.strip()))
        ),
        "",
    )

    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

    print("\n[Eval Debug] Sample model output:")
    if decoded_labels and decoded_preds:
        print(f"  GT : {decoded_labels[0]}")
        print(f"  PR : {decoded_preds[0]}\n")

    cer_scores = [
        calculate_cer(extract_manchu(l), extract_manchu(p))
        for p, l in zip(decoded_preds, decoded_labels)
    ]
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    return {"manchu_cer": mean_cer}


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = torch.argmax(logits, dim=-1)
    return logits
