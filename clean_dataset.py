\
import os
from PIL import Image, UnidentifiedImageError
import hashlib
import shutil
import csv

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"
LOGS_DIR = "logs"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = os.path.join(LOGS_DIR, "cleaning_log.csv")

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def md5_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

hashes = {}
copied = 0
removed = 0

with open(log_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["action", "source", "destination", "reason"])

    for cls in os.listdir(RAW_DIR):
        src_cls = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(src_cls):
            continue

        dst_cls = os.path.join(CLEAN_DIR, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for fname in os.listdir(src_cls):
            src = os.path.join(src_cls, fname)
            if not os.path.isfile(src):
                continue

            if not is_valid_image(src):
                writer.writerow(["removed", src, "", "corrupted"])
                removed += 1
                continue

            file_hash = md5_hash(src)
            if file_hash in hashes:
                writer.writerow(["removed", src, hashes[file_hash], "duplicate"])
                removed += 1
                continue

            hashes[file_hash] = src

            dst = os.path.join(dst_cls, fname)
            i = 1
            base, ext = os.path.splitext(dst)
            while os.path.exists(dst):
                dst = f"{base}_{i}{ext}"
                i += 1

            shutil.copy2(src, dst)
            writer.writerow(["copied", src, dst, "ok"])
            copied += 1

print(f"Cleaning complete. Copied: {copied}, Removed: {removed}")
