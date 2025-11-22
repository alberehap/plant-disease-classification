
import os
import random
import csv
from PIL import Image
from pathlib import Path

random.seed(42)

CLEAN_DIR = Path("data/cleaned")
PROC_DIR = Path("data/processed")
TARGET_SIZE = (224, 224)
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}

os.makedirs("logs", exist_ok=True)

# Create directories
for split in SPLIT:
    for cls in CLEAN_DIR.iterdir():
        if cls.is_dir():
            (PROC_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

mapping_log = "logs/file_split_mapping.csv"

with open(mapping_log, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["src", "dst", "split", "class"])

    for cls in sorted(CLEAN_DIR.iterdir()):
        if not cls.is_dir():
            continue

        images = [p for p in cls.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(images)
        n = len(images)

        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])

        assignments = (
            ["train"] * n_train +
            ["val"] * n_val +
            ["test"] * (n - n_train - n_val)
        )

        for img_file, split in zip(images, assignments):
            dst = PROC_DIR / split / cls.name / img_file.name
            with Image.open(img_file) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                im.save(dst, format="JPEG", quality=95)

            writer.writerow([str(img_file), str(dst), split, cls.name])

print("Preprocessing complete. Mapping saved:", mapping_log)
