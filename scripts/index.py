"""CLI script for indexing images.

Usage:
    python scripts/index.py --version v0a --data_dir data/fashionpedia --output_dir indexes
    python scripts/index.py --version v0b --data_dir data/fashionpedia --output_dir indexes
    python scripts/index.py --version v1 --data_dir data/fashionpedia --output_dir indexes
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_utils import get_image_paths


def main():
    parser = argparse.ArgumentParser(description="Index images for fashion retrieval")
    parser.add_argument("--version", type=str, required=True,
                        choices=["v0a", "v0b", "v1", "v2", "v3", "v4"],
                        help="Indexer version")
    parser.add_argument("--data_dir", type=str, default="data/fashionpedia",
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="indexes",
                        help="Output directory for indexes")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for model inference")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to index (for dev/testing)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Get image paths
    image_paths = get_image_paths(args.data_dir)
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images to index")

    if not image_paths:
        print("No images found! Please download the dataset first:")
        print("  bash scripts/download_data.sh")
        sys.exit(1)

    # Set output directory
    version_output_dir = str(Path(args.output_dir) / args.version)

    # Get appropriate indexer
    if args.version == "v0a":
        from indexer.clip_indexer import CLIPIndexer
        indexer = CLIPIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)
    elif args.version == "v0b":
        from indexer.fashion_clip_indexer import FashionCLIPIndexer
        indexer = FashionCLIPIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)
    elif args.version == "v1":
        from indexer.multi_vector_indexer import MultiVectorIndexer
        indexer = MultiVectorIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)
    elif args.version == "v2":
        from indexer.region_indexer import RegionIndexer
        indexer = RegionIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)
    elif args.version == "v3":
        from indexer.vlm_indexer import VLMIndexer
        indexer = VLMIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)
    elif args.version == "v4":
        from indexer.hybrid_indexer import HybridIndexer
        indexer = HybridIndexer(data_dir=args.data_dir, output_dir=version_output_dir, device=args.device)

    # Store image paths for metadata
    indexer._image_paths = image_paths

    # Run indexing
    indexer.index(image_paths, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
