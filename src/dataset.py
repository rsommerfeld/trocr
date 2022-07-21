from __future__ import division, print_function

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from .configs import paths
from .configs import constants
from .util import debug_print


def load_csv_labels(csv_path: Path = paths.label_file) -> dict[str, str]:
    assert csv_path.exists(), f"Label csv at {csv_path} does not exist."

    labels: dict[str, str] = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            label = row[1]
            image_name = row[0]
            labels[image_name] = label

    return labels


def load_filepaths_and_labels(data_dir: Path = paths.train_dir) -> tuple[list, list]:
    sample_paths: list[str] = []
    labels: list[str] = []

    label_dict = load_csv_labels()

    for file_name in os.listdir(data_dir):
        path = data_dir / file_name

        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            assert file_name in label_dict, f"No label for image '{file_name}'"
            label = label_dict[file_name]

            sample_paths.append(path)
            labels.append(label)

    debug_print(f"Loaded {len(sample_paths)} samples from {data_dir}")
    assert len(sample_paths) == len(labels)
    return sample_paths, labels


class HCRDataset(Dataset):
    def __init__(self, data_dir: Path, processor: TrOCRProcessor):
        self.image_name_list, self.label_list = load_filepaths_and_labels(data_dir)
        self.processor = processor

        self._max_label_len = max([constants.word_len_padding] + [len(label) for label in self.label_list])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]

        label = self.label_list[idx]
        label_tensor = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}

    def get_label(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.label_list[idx]

    def get_path(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.image_name_list[idx]


class MemoryDataset(Dataset):
    def __init__(self, images: list[Image.Image], processor: TrOCRProcessor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]

        # create fake label
        label_tensor: torch.Tensor = self.processor.tokenizer(
            "",
            return_tensors="pt",
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}
