"""Setup helper for Colab/Kaggle environments.

Run this at the top of your notebook:
    !pip install -q -r requirements.txt
    from scripts.setup_colab import setup_colab
    setup_colab()
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_colab(mount_drive=True, data_subdir="fashionpedia"):
    """Set up the environment for Colab/Kaggle.

    - Mounts Google Drive (if on Colab)
    - Creates necessary directories
    - Verifies GPU availability
    - Returns paths for data, indexes, and outputs
    """
    import torch

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Models will be very slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Mount Google Drive if on Colab
    drive_path = None
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            drive_path = Path("/content/drive/MyDrive/fashionista")
            drive_path.mkdir(parents=True, exist_ok=True)
            print(f"Google Drive mounted: {drive_path}")
        except ImportError:
            print("Not on Colab — skipping Drive mount.")

    # Create local directories
    dirs = ["data/fashionpedia", "indexes", "embeddings", "captions", "evaluation/results"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    paths = {
        "device": device,
        "data_dir": "data/fashionpedia",
        "index_dir": "indexes",
        "drive_path": str(drive_path) if drive_path else None,
    }

    print(f"\nSetup complete. Paths: {paths}")
    return paths


def download_fashionpedia_colab(data_dir="data/fashionpedia", full=False):
    """Download Fashionpedia dataset on Colab."""
    import subprocess

    os.makedirs(data_dir, exist_ok=True)
    s3_base = "https://s3.amazonaws.com/ifashionist-dataset"

    if full:
        print("Downloading full Fashionpedia dataset (~30GB)...")
        files = [
            ("images/train2020.zip", "train2020.zip"),
            ("images/val_test2020.zip", "val_test2020.zip"),
            ("annotations/instances_attributes_train2020.json", "instances_attributes_train2020.json"),
            ("annotations/instances_attributes_val2020.json", "instances_attributes_val2020.json"),
        ]
    else:
        print("Downloading Fashionpedia val/test split (~8K images)...")
        files = [
            ("images/val_test2020.zip", "val_test2020.zip"),
            ("annotations/instances_attributes_val2020.json", "instances_attributes_val2020.json"),
        ]

    for remote, local in files:
        local_path = os.path.join(data_dir, local)
        if not os.path.exists(local_path):
            print(f"  Downloading {remote}...")
            subprocess.run(["wget", "-c", f"{s3_base}/{remote}", "-O", local_path])
        else:
            print(f"  Already exists: {local}")

    # Extract zips
    for local in ["train2020.zip", "val_test2020.zip"]:
        zip_path = os.path.join(data_dir, local)
        if os.path.exists(zip_path):
            print(f"  Extracting {local}...")
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", data_dir])
            os.remove(zip_path)

    print(f"Dataset ready at {data_dir}/")
