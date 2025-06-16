from torch.utils.data import Dataset as PyTorchDataset
import torch
from torchvision import transforms


class ManchuDataset(PyTorchDataset):
    def _init__(
        self,
        hf_dataset,
        char2idx,
        idx2char,
        text_key="manchu",
        transform=None,
        max_length=64,
    ):
        self.hf_dataset = hf_dataset
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.text_key = text_key
        self.transform = transform
        self.max_length = max_length
        self.num_classes = len(char2idx)

    def _len__(self):
        return len(self.hf_dataset)

    def _getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample["im"]

        if self.transform:
            img = self.transform(img)

        text = sample[self.text_key]
        seq = [self.char2idx.get(ch, 1) for ch in text][: self.max_length]

        return img, torch.tensor(seq), len(seq), text


def create_data_transforms(config):
    return [
        transforms.Resize(
            (
                config["input_height"],
                config["input_width"],
            )
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(
            lambda x: x + torch.randn_like(x) * 0.01 if torch.rand(1) < 0.1 else x
        ),
    ]
