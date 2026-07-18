"""Text normalizer for query and caption text processing.

Uses rule-based normalization for consistent text representation
between indexing and retrieval. Can be upgraded to LLM-based normalization.
"""

import re


def normalize_text(text: str) -> str:
    """Normalize text for consistent embedding.

    - Lowercase
    - Remove extra whitespace
    - Standardize garment terms (t-shirt → tshirt, etc.)
    - Remove punctuation except hyphens
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # Standardize garment terms
    replacements = {
        "t-shirt": "tshirt",
        "t shirt": "tshirt",
        "button-down": "buttondown",
        "button down": "buttondown",
        "long-sleeve": "longsleeve",
        "long sleeve": "longsleeve",
        "short-sleeve": "shortsleeve",
        "short sleeve": "shortsleeve",
        "v-neck": "vneck",
        "v neck": "vneck",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation except hyphens and apostrophes
    text = re.sub(r"[^\w\s\-']", "", text)

    return text.strip()


def normalize_query(query: str) -> str:
    """Normalize a search query."""
    return normalize_text(query)


def normalize_caption(caption: str) -> str:
    """Normalize a VLM-generated caption."""
    return normalize_text(caption)
