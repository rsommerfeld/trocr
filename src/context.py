from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .dataset import HCRDataset


@dataclass
class Context:
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor

    train_dataset: HCRDataset
    train_dataloader: DataLoader

    val_dataset: HCRDataset
    val_dataloader: DataLoader
