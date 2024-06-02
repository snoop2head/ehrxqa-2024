import os
from PIL import Image
import json
from glob import glob

import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float() / 0xFF
        img = (img - self.mean) / self.std
        return img

class CXR_Dataset(Dataset):
    def __init__(self, data: dict, image_folder: str, transforms: T.Compose, tokenizer: AutoTokenizer):
        self.data = data
        self.image_folder = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_path = os.path.join(self.image_folder, data["image_id"] + ".jpg")
        img = Image.open(image_path)
        img = self.transforms(img)

        text = data["question"]
        text = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        input_ids = text["input_ids"].squeeze()

        return img, input_ids
