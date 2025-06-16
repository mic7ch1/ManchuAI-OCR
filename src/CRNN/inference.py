import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import json

from src.CRNN.model import CRNN
from src.CRNN.utils import greedy_decode


class CRNNInference:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.model = None
        self.char2idx = None
        self.idx2char = None
        self.transform = None

    def load_model(self):
        """Load CRNN model and character dictionaries from checkpoint"""
        if self.model_path.is_dir():
            # Look for different possible checkpoint file names
            possible_names = ["best_model.pth", "model.pth", "final_model.pth"]
            checkpoint_path = None

            for name in possible_names:
                candidate_path = self.model_path / name
                if candidate_path.exists():
                    checkpoint_path = candidate_path
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No valid checkpoint found in {self.model_path}. Looked for: {possible_names}"
                )
        else:
            # If it's a file path directly
            checkpoint_path = self.model_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        print(f"Loading CRNN model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract character dictionaries
        self.char2idx = checkpoint["char2idx"]
        self.idx2char = checkpoint["idx2char"]

        # Initialize model with correct number of classes
        num_classes = len(self.char2idx)

        # Require model parameters to be saved in checkpoint
        if "hidden_size" not in checkpoint:
            raise KeyError(
                "Model checkpoint missing 'hidden_size' parameter. Please retrain the model with updated trainer."
            )
        if "dropout" not in checkpoint:
            raise KeyError(
                "Model checkpoint missing 'dropout' parameter. Please retrain the model with updated trainer."
            )

        hidden_size = checkpoint["hidden_size"]
        dropout = checkpoint["dropout"]
        self.model = CRNN(num_classes, hidden_size, dropout)

        # Load model state
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

        # Setup preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 480)),  # Standard CRNN input size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        print(f"Model loaded successfully with {num_classes} classes")

    def preprocess_image(self, image):
        """Preprocess PIL image for CRNN inference"""
        if self.transform is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def inference(self, image):
        """Run inference on a single image"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)  # [W, B, C]

        # Decode predictions using greedy decoding
        predictions = greedy_decode(logits, self.idx2char)

        return predictions[0] if predictions else ""

    @classmethod
    def from_pretrained(cls, model_path, device="cuda"):
        """Factory method to create and load model in one step"""
        inference = cls(model_path, device)
        inference.load_model()
        return inference


def load_crnn_model(model_path, device="cuda"):
    """Convenience function to load CRNN model for inference"""
    return CRNNInference.from_pretrained(model_path, device)
