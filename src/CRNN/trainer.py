import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pathlib import Path
import json
from tqdm import tqdm

from src.CRNN.dataset import ManchuDataset
from src.CRNN.model import CRNN
from src.CRNN.utils import greedy_decode, build_character_dict, collate_crnn_batch
from src.CRNN.optimizer import create_optimizer
from src.CRNN.scheduler import create_scheduler
from src.CRNN.dataset import create_data_transforms


def load_model_and_tokenizer(
    base_model_name,
    loading_config=None,
    peft_config=None,
    train_config=None,
    train_dataset=None,
    val_dataset=None,
):
    if train_dataset is None or val_dataset is None:
        raise ValueError(
            "CRNN requires train_dataset and val_dataset to build character dictionary"
        )

    if train_config is None:
        train_config = {}

    char2idx, idx2char = build_character_dict(
        [train_dataset, val_dataset], text_key="manchu"
    )

    model = CRNN(
        len(char2idx),
        train_config["hidden_size"],
        train_config["dropout"],
    )

    return model, {"char2idx": char2idx, "idx2char": idx2char}


class CRNNTrainer:
    def _init__(
        self, model, tokenizer, train_dataset, eval_dataset, config, output_dir
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.char2idx = tokenizer["char2idx"]
        self.idx2char = tokenizer["idx2char"]
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda")

        self.model.to(self.device)

        transforms_list = create_data_transforms(config)

        self.train_dataset = ManchuDataset(
            train_dataset,
            self.char2idx,
            self.idx2char,
            text_key="manchu",
            transform=transforms.Compose(transforms_list),
            max_length=config["max_text_length"],
        )
        self.eval_dataset = ManchuDataset(
            eval_dataset,
            self.char2idx,
            self.idx2char,
            text_key="manchu",
            transform=transforms.Compose(transforms_list),
            max_length=config["max_text_length"],
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            config["batch_size"],
            True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=collate_crnn_batch,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            config["batch_size"],
            False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=collate_crnn_batch,
        )

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = create_optimizer(self.model.parameters(), config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.scaler = (
            torch.amp.GradScaler("cuda") if config["mixed_precision"] else None
        )

        self.best_loss = 1e9
        self.step = 0
        self.start_epoch = 0

        self.load_history()

    def load_history(self):
        history_path = self.output_dir / "history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                    if "best" in history:
                        self.best_loss = history["best"]
                        print(f"Loaded existing best loss: {self.best_loss:.4f}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load history from {history_path}: {e}")

    def save_history(self):
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump({"best": self.best_loss}, f, indent=2)

    def train_epoch(self, is_training=True):
        self.model.train() if is_training else self.model.eval()
        loader = self.train_loader if is_training else self.eval_loader
        loss_sum = 0

        if len(loader) == 0:
            print(
                f"Warning: {'Training' if is_training else 'Validation'} loader is empty!"
            )
            return float("inf")

        for imgs, lbls, lens, raws in tqdm(
            loader, desc="Train" if is_training else "Val"
        ):
            imgs, lbls, lens = (
                imgs.to(self.device),
                lbls.to(self.device),
                lens.to(self.device),
            )

            if (
                is_training
                and self.config["mixed_precision"]
                and self.scaler is not None
            ):
                with torch.amp.autocast("cuda"):
                    logits = self.model(imgs)
                    inp_len = torch.full(
                        (imgs.size(0),),
                        logits.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )
                logits_fp32 = logits.float()
                loss = self.criterion(logits_fp32, lbls, inp_len, lens)
            elif not is_training:
                with torch.no_grad():
                    logits = self.model(imgs)
                    inp_len = torch.full(
                        (imgs.size(0),),
                        logits.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )
                    loss = self.criterion(logits, lbls, inp_len, lens)
            else:
                logits = self.model(imgs)
                inp_len = torch.full(
                    (imgs.size(0),),
                    logits.size(0),
                    dtype=torch.long,
                    device=self.device,
                )
                loss = self.criterion(logits, lbls, inp_len, lens)

            if is_training:
                self.optimizer.zero_grad(set_to_none=True)

                if self.config["mixed_precision"] and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.config["gradient_clipping"]:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clipping"]["max_norm"],
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config["gradient_clipping"]:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clipping"]["max_norm"],
                        )
                    self.optimizer.step()

                self.step += 1

                if (
                    self.config.get("save_every_n_steps")
                    and self.step % self.config["save_every_n_steps"] == 0
                ):
                    step_checkpoint_path = (
                        self.output_dir / f"ckpt_step_{self.step}.pth"
                    )
                    self.save_checkpoint_with_step(
                        self.step, loss.item(), step_checkpoint_path
                    )
                    print(f"✓ step checkpoint saved at step {self.step}")

            loss_sum += loss.item()

            if (
                is_training
                and self.config.get("display_every_n_steps")
                and self.step % self.config["display_every_n_steps"] == 0
            ):
                print(f"\n[step {self.step}] GT  : {raws[0]}")
                print(
                    f"[step {self.step}] Pred: {greedy_decode(logits, self.idx2char)[0]}"
                )

        return loss_sum / len(loader)

    def save_model(self, checkpoint_path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "char2idx": self.char2idx,
                "idx2char": self.idx2char,
                "hidden_size": self.config["hidden_size"],
                "dropout": self.config["dropout"],
            },
            checkpoint_path,
        )

    def save_checkpoint(self, epoch, val_loss, checkpoint_path):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "val": val_loss,
                "char2idx": self.char2idx,
                "idx2char": self.idx2char,
                "hidden_size": self.config["hidden_size"],
                "dropout": self.config["dropout"],
            },
            checkpoint_path,
        )

    def save_checkpoint_with_step(self, step, loss, checkpoint_path):
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "loss": loss,
                "char2idx": self.char2idx,
                "idx2char": self.idx2char,
                "hidden_size": self.config["hidden_size"],
                "dropout": self.config["dropout"],
            },
            checkpoint_path,
        )

    def train(self, resume_from_checkpoint=None):
        num_epochs = self.config["num_train_epochs"]

        if resume_from_checkpoint is not None:
            self.resume_from_checkpoint(resume_from_checkpoint)

        for epoch in range(self.start_epoch, num_epochs):
            print(
                f"\nEpoch {epoch+1}/{num_epochs} lr={self.scheduler.get_last_lr()[0]:.1e}"
            )
            print(
                f"Train loader size: {len(self.train_loader)}, Eval loader size: {len(self.eval_loader)}"
            )

            train_loss = self.train_epoch(is_training=True)

            val_loss = self.train_epoch(is_training=False)

            self.scheduler.step()
            print(f"train {train_loss:.4f} | val {val_loss:.4f}")
            print(f"Current best loss: {self.best_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = self.output_dir / "best_model.pth"
                self.save_checkpoint(epoch, val_loss, checkpoint_path)
                self.save_history()
                print(f"✓ best model updated and saved to {checkpoint_path}")
            else:
                print(
                    f"✗ validation loss {val_loss:.4f} >= best loss {self.best_loss:.4f}, not saving"
                )

            if (
                self.config.get("save_every_n_epochs")
                and (epoch + 1) % self.config["save_every_n_epochs"] == 0
            ):
                self.save_checkpoint(
                    epoch, val_loss, self.output_dir / f"checkpoint-epoch-{epoch+1}.pth"
                )

        self.save_model(self.output_dir / "final_model.pth")

        self.save_history()

        print(f"CRNN training completed. Best validation loss: {self.best_loss:.4f}")

    def resume_from_checkpoint(self, checkpoint_path):
        if not checkpoint_path.exists():
            print(
                f"Warning: Checkpoint {checkpoint_path} does not exist, starting from scratch"
            )
            return

        print(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model"])

            if "optim" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optim"])

            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
                print(f"Resuming from epoch {self.start_epoch}")

            if "step" in checkpoint:
                self.step = checkpoint["step"]
                print(f"Resuming from step {self.step}")

            if "val" in checkpoint:
                checkpoint_val_loss = checkpoint["val"]
                print(f"Checkpoint validation loss: {checkpoint_val_loss:.4f}")

            print("✓ Successfully resumed from checkpoint")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
