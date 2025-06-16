import shutil
from src.CRNN.trainer import CRNNTrainer, load_model_and_tokenizer
from src.utils.files import create_dir


def train_crnn_model(
    model_config,
    training_config,
    dataset_config,
    train_dataset,
    val_dataset,
    training_output,
):
    model_name = model_config["name"]
    train_config = training_config["training"]

    training_output_subdir = training_output / "checkpoints"
    create_dir(training_output_subdir)

    print(f"Training {model_name}...")
    print(f"Training output: {training_output_subdir}")
    model, tokenizer = load_model_and_tokenizer(
        model_config["base_model"],
        train_config=train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer = CRNNTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=train_config,
        output_dir=training_output_subdir,
    )

    checkpoints = list(training_output_subdir.glob("checkpoint-epoch-*.pth"))
    recent_checkpoint = (
        max(checkpoints, key=lambda x: x.stat().st_ctime) if checkpoints else None
    )

    if recent_checkpoint:
        print(f"Found checkpoint: {recent_checkpoint}")
        print("Note: Resume functionality not yet implemented for CRNN")
    else:
        print("Starting training from scratch")

    try:
        trainer.train(resume_from_checkpoint=recent_checkpoint)
        print(f"CRNN training finished for {model_name}.")

        best_model_path = training_output / "best_model"
        best_model_path.mkdir(exist_ok=True)

        best_checkpoint = training_output_subdir / "best_model.pth"
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, best_model_path / "best_model.pth")
            print(f"Best model saved to: {best_model_path}")
        else:
            print(f"Warning: Best checkpoint not found at {best_checkpoint}")

    except Exception as e:
        print(f"Error during training: {e}")
        return
