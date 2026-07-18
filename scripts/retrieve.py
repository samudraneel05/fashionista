"""CLI script for retrieval.

Usage:
    python scripts/retrieve.py --query "A person in a bright yellow raincoat" --version v0a --top_k 10
    python scripts/retrieve.py --query "Red tie and white shirt" --version v4 --top_k 5
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Retrieve images for a fashion query")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--version", type=str, default="v0a",
                        choices=["v0a", "v0b", "v1", "v2", "v3", "v4"],
                        help="Retriever version")
    parser.add_argument("--index_dir", type=str, default="indexes", help="Index directory")
    parser.add_argument("--data_dir", type=str, default="data/fashionpedia", help="Image data directory")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    version_index_dir = str(Path(args.index_dir) / args.version)

    if args.version in ("v0a", "v0b"):
        from retriever.clip_retriever import CLIPRetriever
        retriever = CLIPRetriever(
            version_name=args.version,
            index_dir=version_index_dir,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.version == "v1":
        from retriever.multi_vector_retriever import MultiVectorRetriever
        retriever = MultiVectorRetriever(
            index_dir=version_index_dir,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.version == "v2":
        from retriever.region_retriever import RegionRetriever
        retriever = RegionRetriever(
            index_dir=version_index_dir,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.version == "v3":
        from retriever.vlm_retriever import VLMRetriever
        retriever = VLMRetriever(
            index_dir=version_index_dir,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.version == "v4":
        from retriever.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            index_dir=version_index_dir,
            data_dir=args.data_dir,
            device=args.device,
        )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: '{args.query}'")
    print(f"Version: {args.version}")
    print(f"Top-{args.top_k} results:\n")
    print(f"{'Rank':<6} {'Image ID':<30} {'Score':<10} {'Path'}")
    print(f"{'-'*6} {'-'*30} {'-'*10} {'-'*50}")
    for i, result in enumerate(results):
        print(f"{i+1:<6} {result['image_id']:<30} {result['score']:<10.4f} {result.get('image_path', 'N/A')}")

        # Print channel scores if available (V1, V4)
        if "channel_scores" in result:
            for channel, score in result["channel_scores"].items():
                print(f"      └─ {channel}: {score:.4f}")


if __name__ == "__main__":
    main()
