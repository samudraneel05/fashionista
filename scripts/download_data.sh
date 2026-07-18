#!/bin/bash
# Download Fashionpedia dataset from S3
# Usage: bash scripts/download_data.sh [--full]
#   --full: Download full dataset (48K images). Without flag, downloads val/test split only (~8K images).

set -e

DATA_DIR="data/fashionpedia"
mkdir -p "$DATA_DIR"

FULL=false
if [ "$1" = "--full" ]; then
    FULL=true
fi

S3_BASE="https://s3.amazonaws.com/ifashionist-dataset"

if [ "$FULL" = true ]; then
    echo "Downloading full Fashionpedia dataset (train + val/test)..."
    echo "This is a large download (~30GB). Please be patient."

    echo "Downloading training images..."
    wget -c "$S3_BASE/images/train2020.zip" -O "$DATA_DIR/train2020.zip"

    echo "Downloading validation/test images..."
    wget -c "$S3_BASE/images/val_test2020.zip" -O "$DATA_DIR/val_test2020.zip"

    echo "Extracting training images..."
    unzip -q -o "$DATA_DIR/train2020.zip" -d "$DATA_DIR/"

    echo "Extracting validation/test images..."
    unzip -q -o "$DATA_DIR/val_test2020.zip" -d "$DATA_DIR/"

    echo "Downloading training annotations..."
    wget -c "$S3_BASE/annotations/instances_attributes_train2020.json" -O "$DATA_DIR/instances_attributes_train2020.json"

    echo "Downloading validation annotations..."
    wget -c "$S3_BASE/annotations/instances_attributes_val2020.json" -O "$DATA_DIR/instances_attributes_val2020.json"
else
    echo "Downloading Fashionpedia val/test split only (~8K images)..."
    echo "Use --full flag for complete dataset."

    echo "Downloading validation/test images..."
    wget -c "$S3_BASE/images/val_test2020.zip" -O "$DATA_DIR/val_test2020.zip"

    echo "Extracting validation/test images..."
    unzip -q -o "$DATA_DIR/val_test2020.zip" -d "$DATA_DIR/"

    echo "Downloading validation annotations..."
    wget -c "$S3_BASE/annotations/instances_attributes_val2020.json" -O "$DATA_DIR/instances_attributes_val2020.json"
fi

echo "Cleaning up zip files..."
rm -f "$DATA_DIR"/*.zip

echo "Done! Dataset saved to $DATA_DIR/"
echo ""
echo "Directory structure:"
find "$DATA_DIR" -maxdepth 2 -type d | head -20
