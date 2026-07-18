"""CLI wrapper for evaluation.

Usage:
    python scripts/evaluate.py --versions v0a v0b --index_dir indexes
    python scripts/evaluate.py --all --index_dir indexes --annotations data/fashionpedia/instances_attributes_val2020.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import main

if __name__ == "__main__":
    main()
