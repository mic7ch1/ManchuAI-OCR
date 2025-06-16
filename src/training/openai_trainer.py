import openai
import os
import time
from pathlib import Path
from openai.types.fine_tuning import SupervisedMethod, SupervisedHyperparameters
from src.utils.dataset import create_openai_jsonl_file


def upload_file_with_retry(client, file_path, purpose, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                response = client.files.create(file=f, purpose=purpose)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2**attempt)


def create_fine_tuning_job_with_retry(
    client,
    training_file_id,
    validation_file_id,
    model,
    suffix,
    epochs=None,
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            method_config = {
                "type": "supervised",
                "supervised": SupervisedMethod(
                    hyperparameters=SupervisedHyperparameters(
                        n_epochs=epochs if epochs is not None else "auto"
                    )
                ),
            }

            response = client.fine_tuning.jobs.create(
                training_file=training_file_id,
                validation_file=validation_file_id,
                model=model,
                suffix=suffix,
                method=method_config,
            )
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2**attempt)


def train_openai_model(
    model_config,
    training_config,
    dataset_config,
    train_dataset,
    val_dataset,
    output_dir_model_base,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    out_dir = output_dir_model_base
    out_dir.mkdir(parents=True, exist_ok=True)

    model_to_fine_tune = training_config.get("model_to_fine_tune")
    num_train_samples = training_config.get("num_train_samples")
    num_valid_samples = training_config.get("num_valid_samples")
    epochs = training_config.get("num_train_epochs")
    fine_tuning_suffix = training_config.get("fine_tuning_suffix")
    instruction = dataset_config.get("instruction")

    print(f"Creating training data with {num_train_samples} samples...")
    train_jsonl_file = create_openai_jsonl_file(
        train_dataset, num_train_samples, out_dir / "train_data.jsonl", instruction
    )

    if not train_jsonl_file:
        raise ValueError("Failed to create training file")

    print(f"Creating validation data with {num_valid_samples} samples...")
    valid_jsonl_file = create_openai_jsonl_file(
        val_dataset, num_valid_samples, out_dir / "validation_data.jsonl", instruction
    )

    print("Uploading training file...")
    training_file_response = upload_file_with_retry(
        client, train_jsonl_file, "fine-tune"
    )
    training_file_id = training_file_response.id

    validation_file_id = None
    if valid_jsonl_file and Path(valid_jsonl_file).exists():
        print("Uploading validation file...")
        validation_file_response = upload_file_with_retry(
            client, valid_jsonl_file, "fine-tune"
        )
        validation_file_id = validation_file_response.id

    print("Creating fine-tuning job...")
    fine_tuning_job = create_fine_tuning_job_with_retry(
        client,
        training_file_id,
        validation_file_id,
        model_to_fine_tune,
        fine_tuning_suffix,
        epochs,
    )

    print(f"Fine-tuning job created: {fine_tuning_job.id}")
    print(f"Status: {fine_tuning_job.status}")
