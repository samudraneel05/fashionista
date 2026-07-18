"""Query parser for extracting attributes from natural language queries.

Supports rule-based extraction of:
- Color-item pairs (e.g., "red shirt", "blue pants")
- Environment keywords (e.g., "office", "park", "home")
- Style intent (e.g., "casual", "formal", "weekend")
- Query type classification (attribute, contextual, complex, style, compositional)
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field


COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "beige", "navy", "maroon", "teal",
    "olive", "burgundy", "cream", "ivory", "tan", "gold", "silver", "cyan",
    "magenta", "lavender", "coral", "turquoise", "charcoal", "khaki",
}

GARMENTS = {
    "shirt", "t-shirt", "tshirt", "pants", "trousers", "jeans", "dress",
    "skirt", "jacket", "coat", "blazer", "hoodie", "sweater", "pullover",
    "raincoat", "tie", "scarf", "hat", "cap", "shoes", "sneakers", "boots",
    "heels", "sandals", "shorts", "suit", "vest", "cardigan", "blouse",
    "tank", "tanktop", "polo", "button-down", "buttondown", "overcoat",
    "trench", "windbreaker", "gloves", "belt", "bag", "purse", "sunglasses",
}

ENVIRONMENTS = {
    "office": ["office", "workplace", "corporate", "business", "meeting room", "desk"],
    "urban": ["street", "city", "urban", "downtown", "sidewalk", "road", "building"],
    "park": ["park", "garden", "bench", "outdoor", "nature", "grass", "tree", "field"],
    "home": ["home", "house", "living room", "bedroom", "kitchen", "indoor", "couch", "sofa"],
    "formal_setting": ["formal setting", "formal event", "gala", "wedding", "ceremony"],
}

STYLE_KEYWORDS = {
    "casual": ["casual", "weekend", "relaxed", "informal", "everyday", "laid-back", "comfy"],
    "formal": ["formal", "business", "professional", "work", "corporate", "elegant", "sophisticated"],
    "trendy": ["trendy", "fashionable", "stylish", "modern", "chic", "contemporary"],
    "classic": ["classic", "timeless", "traditional", "vintage"],
}

ANATOMICAL_MAP = {
    "shirt": "upper body", "t-shirt": "upper body", "tshirt": "upper body",
    "blouse": "upper body", "hoodie": "upper body", "sweater": "upper body",
    "pullover": "upper body", "cardigan": "upper body", "tank": "upper body",
    "tanktop": "upper body", "polo": "upper body", "button-down": "upper body",
    "buttondown": "upper body", "vest": "upper body", "blazer": "upper body",
    "jacket": "upper body", "coat": "upper body", "overcoat": "upper body",
    "trench": "upper body", "raincoat": "upper body", "windbreaker": "upper body",
    "tie": "upper body", "scarf": "upper body",
    "pants": "lower body", "trousers": "lower body", "jeans": "lower body",
    "shorts": "lower body", "skirt": "lower body",
    "shoes": "feet", "sneakers": "feet", "boots": "feet", "heels": "feet",
    "sandals": "feet",
    "hat": "head", "cap": "head",
    "bag": "accessory", "purse": "accessory", "sunglasses": "accessory",
    "belt": "accessory", "gloves": "accessory",
}


@dataclass
class ParsedQuery:
    """Structured representation of a parsed query."""
    raw: str
    query_type: str  # 'attribute', 'contextual', 'complex', 'style', 'compositional'
    color_item_pairs: List[Tuple[str, str]] = field(default_factory=list)  # [(color, garment), ...]
    colors: List[str] = field(default_factory=list)
    garments: List[str] = field(default_factory=list)
    environment: Optional[str] = None
    environment_keywords: List[str] = field(default_factory=list)
    style: Optional[str] = None
    is_anatomical: bool = False
    prompts: Dict[str, str] = field(default_factory=dict)  # {channel: prompt_text}


class QueryParser:
    """Rule-based query parser for fashion retrieval queries."""

    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured components."""
        query_lower = query.lower().strip()

        # Extract color-item pairs
        color_item_pairs = self._extract_color_item_pairs(query_lower)
        colors = [c for c, _ in color_item_pairs] if color_item_pairs else self._extract_colors(query_lower)
        garments = [g for _, g in color_item_pairs] if color_item_pairs else self._extract_garments(query_lower)

        # Extract environment
        environment, env_keywords = self._extract_environment(query_lower)

        # Extract style
        style = self._extract_style(query_lower)

        # Determine query type
        query_type = self._classify_query(query_lower, color_item_pairs, environment, style)

        # Determine if anatomical (fashion attributes detected)
        is_anatomical = len(garments) > 0

        # Generate channel-specific prompts
        prompts = self._generate_prompts(query, color_item_pairs, garments, colors, environment, env_keywords, style, is_anatomical)

        return ParsedQuery(
            raw=query,
            query_type=query_type,
            color_item_pairs=color_item_pairs,
            colors=colors,
            garments=garments,
            environment=environment,
            environment_keywords=env_keywords,
            style=style,
            is_anatomical=is_anatomical,
            prompts=prompts,
        )

    def _extract_color_item_pairs(self, query: str) -> List[Tuple[str, str]]:
        """Extract (color, garment) pairs from query."""
        pairs = []
        words = query.split()

        for i, word in enumerate(words):
            clean_word = re.sub(r'[^a-z]', '', word)
            if clean_word in COLORS and i + 1 < len(words):
                next_word = re.sub(r'[^a-z-]', '', words[i + 1])
                # Handle multi-word garments like "t-shirt", "button-down"
                if next_word in GARMENTS:
                    pairs.append((clean_word, next_word))
                elif i + 2 < len(words):
                    two_word = next_word + "-" + re.sub(r'[^a-z]', '', words[i + 2])
                    if two_word in GARMENTS:
                        pairs.append((clean_word, two_word))

        return pairs

    def _extract_colors(self, query: str) -> List[str]:
        """Extract standalone color mentions."""
        found = []
        words = re.findall(r'[a-z]+', query)
        for word in words:
            if word in COLORS and word not in found:
                found.append(word)
        return found

    def _extract_garments(self, query: str) -> List[str]:
        """Extract standalone garment mentions."""
        found = []
        words = re.findall(r'[a-z-]+', query)
        for word in words:
            if word in GARMENTS and word not in found:
                found.append(word)
        return found

    def _extract_environment(self, query: str) -> Tuple[Optional[str], List[str]]:
        """Extract environment from query."""
        for env, keywords in ENVIRONMENTS.items():
            for kw in keywords:
                if kw in query:
                    return env, [kw]
        return None, []

    def _extract_style(self, query: str) -> Optional[str]:
        """Extract style intent from query."""
        for style, keywords in STYLE_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    return style
        return None

    def _classify_query(self, query: str, pairs: List, environment: Optional[str], style: Optional[str]) -> str:
        """Classify query into one of 5 categories."""
        has_garments = len(pairs) > 0 or bool(self._extract_garments(query))
        has_env = environment is not None
        has_style = style is not None
        has_multiple_pairs = len(pairs) >= 2

        # Check for explicit location prepositions (indicates contextual intent)
        has_location_phrase = any(
            phrase in query for phrase in
            ["inside", "in a", "in an", "at a", "at an", "on a", "on an"]
        )

        if has_multiple_pairs:
            return "compositional"
        if has_env and has_garments:
            return "complex"
        if has_env and has_location_phrase:
            return "contextual"
        if has_style and not has_garments:
            return "style"
        if has_env:
            return "contextual"
        if has_garments:
            return "attribute"
        return "style"  # default fallback

    def _generate_prompts(
        self,
        query: str,
        pairs: List[Tuple[str, str]],
        garments: List[str],
        colors: List[str],
        environment: Optional[str],
        env_keywords: List[str],
        style: Optional[str],
        is_anatomical: bool,
    ) -> Dict[str, str]:
        """Generate channel-specific prompts for multi-vector retrieval."""
        prompts = {"global": query}

        # Scene prompt: focus on environment
        if environment:
            prompts["scene"] = f"a scene in {environment.replace('_', ' ')}"
        elif env_keywords:
            prompts["scene"] = f"a scene with {env_keywords[0]}"
        else:
            prompts["scene"] = "a scene"

        # Action prompt: focus on person/clothing
        if is_anatomical and pairs:
            # Build anatomical prompt with color-item pairs
            parts = []
            for color, garment in pairs:
                body_part = ANATOMICAL_MAP.get(garment, "a person")
                parts.append(f"{body_part} wearing a {color} {garment}")
            prompts["action"] = " and ".join(parts)
        elif garments:
            garment_str = " and ".join(garments)
            prompts["action"] = f"a person wearing {garment_str}"
        elif style:
            prompts["action"] = f"a person in {style} attire"
        else:
            prompts["action"] = f"a person, {query}"

        return prompts
