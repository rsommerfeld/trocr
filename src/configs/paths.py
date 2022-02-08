import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
trocr_dir = dir_path.parent.parent  # this file is in src/configs

trocr_repo = "microsoft/trocr-base-handwritten"
model_path = trocr_dir / "model"

train_dir = trocr_dir / "train"
val_dir = trocr_dir / "val"
label_dir = trocr_dir / "gt"
label_file = label_dir / "labels.csv"


# automatically create all directories
for dir in [train_dir, val_dir, label_dir]:
    dir.mkdir(parents=True, exist_ok=True)
