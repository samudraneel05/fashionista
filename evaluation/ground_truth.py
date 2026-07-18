"""Ground truth generation for evaluation queries.

Fashionpedia has NO retrieval ground truth — its API evaluates instance segmentation
(AP/AR with IoU + F1 thresholds), not text→image retrieval. We must CONSTRUCT
retrieval ground truth from the dataset's annotations.

Three-tier approach:
1. Annotation-based: Map query terms → actual Fashionpedia category/attribute names
   → find images whose annotations match. Used for attribute/compositional queries.
2. Caption-based: Use VLM-generated captions (Florence-2 + BLIP) + LLM judgment
   for subjective queries (style, context, complex).
3. Hybrid: Combine both when possible.

Fashionpedia annotation structure:
- 46 categories (27 main apparel + 19 parts), each with {id, name, supercategory}
- 294 attributes across 9 supercategories, each with {id, name, supercategory}
- Per annotation: {image_id, category_id, attribute_ids: [int], segmentation, bbox}
- Global attributes task: {image_id, attribute_ids: [int]} (image-level)
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Optional, Any, Tuple
from collections import defaultdict


class FashionpediaAnnotationIndex:
    """Indexes Fashionpedia annotations for efficient ground truth lookup.

    Builds:
    - image_id → {categories: set(str), attributes: set(str), attribute_supercats: set(str)}
    - attribute_name → set(image_id)  (reverse index)
    - category_name → set(image_id)   (reverse index)
    """

    def __init__(self, annotations_path: str):
        self.annotations_path = annotations_path
        self.raw: Dict[str, Any] = {}
        self.categories: Dict[int, Dict[str, str]] = {}
        self.attributes: Dict[int, Dict[str, str]] = {}
        self.image_to_annotations: Dict[str, Dict[str, Set[str]]] = {}
        self.attr_to_images: Dict[str, Set[str]] = defaultdict(set)
        self.cat_to_images: Dict[str, Set[str]] = defaultdict(set)
        self.attr_supercat_to_attrs: Dict[str, Set[str]] = defaultdict(set)
        self.image_file_names: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not Path(self.annotations_path).exists():
            print(f"Warning: Annotations file not found: {self.annotations_path}")
            return

        print(f"Loading Fashionpedia annotations from {self.annotations_path}...")
        with open(self.annotations_path, "r") as f:
            self.raw = json.load(f)

        for cat in self.raw.get("categories", []):
            self.categories[cat["id"]] = {
                "name": cat["name"].lower(),
                "supercategory": cat.get("supercategory", "").lower(),
            }

        for attr in self.raw.get("attributes", []):
            self.attributes[attr["id"]] = {
                "name": attr["name"].lower(),
                "supercategory": attr.get("supercategory", "").lower(),
            }
            self.attr_supercat_to_attrs[attr.get("supercategory", "").lower()].add(
                attr["name"].lower()
            )

        for img in self.raw.get("images", []):
            self.image_file_names[str(img["id"])] = img.get("file_name", "")

        for ann in self.raw.get("annotations", []):
            img_id = str(ann["image_id"])
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = {
                    "categories": set(),
                    "attributes": set(),
                    "attribute_supercats": set(),
                }

            cat_id = ann.get("category_id")
            if cat_id and cat_id in self.categories:
                cat_name = self.categories[cat_id]["name"]
                self.image_to_annotations[img_id]["categories"].add(cat_name)
                self.cat_to_images[cat_name].add(img_id)

            for attr_id in ann.get("attribute_ids", []):
                if attr_id in self.attributes:
                    attr_name = self.attributes[attr_id]["name"]
                    attr_super = self.attributes[attr_id]["supercategory"]
                    self.image_to_annotations[img_id]["attributes"].add(attr_name)
                    self.image_to_annotations[img_id]["attribute_supercats"].add(attr_super)
                    self.attr_to_images[attr_name].add(img_id)

        print(f"  Loaded {len(self.categories)} categories, {len(self.attributes)} attributes")
        print(f"  Indexed {len(self.image_to_annotations)} images with annotations")
        print(f"  Attribute supercategories: {sorted(self.attr_supercat_to_attrs.keys())}")

    def get_all_attribute_names(self) -> List[str]:
        return sorted([a["name"] for a in self.attributes.values()])

    def get_all_category_names(self) -> List[str]:
        return sorted([c["name"] for c in self.categories.values()])

    def get_attributes_by_supercategory(self, supercat: str) -> List[str]:
        return sorted(self.attr_supercat_to_attrs.get(supercat.lower(), set()))

    def get_images_with_category(self, category_name: str) -> Set[str]:
        return self.cat_to_images.get(category_name.lower(), set())

    def get_images_with_attribute(self, attribute_name: str) -> Set[str]:
        return self.attr_to_images.get(attribute_name.lower(), set())

    def get_images_with_attributes(self, attribute_names: Set[str]) -> Set[str]:
        if not attribute_names:
            return set()
        result = None
        for attr in attribute_names:
            imgs = self.attr_to_images.get(attr.lower(), set())
            if result is None:
                result = imgs.copy()
            else:
                result &= imgs
        return result if result else set()

    def get_images_with_category_and_attributes(
        self, category_name: str, attribute_names: Set[str]
    ) -> Set[str]:
        cat_images = self.get_images_with_category(category_name)
        if not attribute_names:
            return cat_images
        attr_images = self.get_images_with_attributes(attribute_names)
        return cat_images & attr_images

    def get_image_attributes(self, image_id: str) -> Dict[str, Set[str]]:
        return self.image_to_annotations.get(str(image_id), {
            "categories": set(), "attributes": set(), "attribute_supercats": set(),
        })

    def to_filename_stems(self, image_ids: Set[str]) -> Set[str]:
        """Convert numeric Fashionpedia image IDs to filename stems.

        Retrievers use filename stems (e.g. 'e14733841b04c64e75789a91fbe549b3')
        as image IDs, but annotations use numeric IDs (e.g. '13297').
        This method bridges the gap.
        """
        from pathlib import Path
        result = set()
        for img_id in image_ids:
            file_name = self.image_file_names.get(str(img_id), "")
            if file_name:
                result.add(Path(file_name).stem)
            else:
                result.add(str(img_id))
        return result


class QueryAnnotationMatcher:
    """Maps natural language query terms to Fashionpedia annotation names.

    Uses fuzzy matching against the actual 294 attributes and 46 categories
    in the dataset, rather than hardcoded guesses.
    """

    GARMENT_TO_CATEGORY = {
        "raincoat": ["coat", "jacket", "outerwear"],
        "coat": ["coat", "jacket"],
        "jacket": ["jacket", "coat"],
        "blazer": ["blazer", "jacket"],
        "shirt": ["shirt, blouse", "top"],
        "blouse": ["shirt, blouse"],
        "t-shirt": ["shirt, blouse", "top"],
        "tshirt": ["shirt, blouse", "top"],
        "hoodie": ["sweater", "shirt, blouse"],
        "sweater": ["sweater"],
        "pullover": ["sweater"],
        "cardigan": ["sweater"],
        "dress": ["dress"],
        "skirt": ["skirt"],
        "pants": ["pants", "trousers"],
        "trousers": ["pants", "trousers"],
        "jeans": ["pants", "trousers"],
        "shorts": ["shorts"],
        "suit": ["blazer", "jacket"],
        "tie": ["tie"],
        "scarf": ["scarf"],
        "hat": ["hat"],
        "cap": ["hat"],
        "shoes": ["shoe"],
        "sneakers": ["shoe"],
        "boots": ["shoe"],
        "heels": ["shoe"],
        "belt": ["belt"],
        "bag": ["bag"],
        "sunglasses": ["sunglasses"],
        "gloves": ["gloves"],
    }

    COLOR_SYNONYMS = {
        "yellow": ["yellow", "lemon", "gold", "mustard"],
        "blue": ["blue", "navy", "azure", "cobalt", "teal", "sky blue", "light blue", "dark blue"],
        "red": ["red", "crimson", "maroon", "burgundy", "wine"],
        "white": ["white", "ivory", "cream", "off-white", "beige"],
        "black": ["black", "charcoal", "grey", "gray", "dark grey"],
        "green": ["green", "olive", "emerald", "lime", "mint", "sage"],
        "pink": ["pink", "rose", "coral", "salmon"],
        "purple": ["purple", "violet", "lavender", "plum", "mauve"],
        "orange": ["orange", "rust", "terracotta"],
        "brown": ["brown", "tan", "chocolate", "coffee", "camel"],
        "silver": ["silver", "metallic", "grey"],
        "gold": ["gold", "metallic", "yellow"],
        "navy": ["navy", "navy blue", "dark blue"],
        "beige": ["beige", "cream", "ivory", "nude"],
    }

    def __init__(self, annotation_index: FashionpediaAnnotationIndex):
        self.index = annotation_index
        self.all_attrs = set(annotation_index.get_all_attribute_names())
        self.all_cats = set(annotation_index.get_all_category_names())

        self.actual_colors = set(annotation_index.get_attributes_by_supercategory("colors"))
        if not self.actual_colors:
            color_words = {"yellow", "blue", "red", "white", "black", "green", "pink",
                          "purple", "orange", "brown", "gray", "grey", "navy", "beige",
                          "gold", "silver", "cream", "ivory", "maroon", "burgundy",
                          "teal", "olive", "coral", "lavender", "charcoal", "tan",
                          "rust", "mustard", "sage", "mint", "rose", "plum", "mauve",
                          "cobalt", "azure", "wine", "salmon", "terracotta", "camel",
                          "chocolate", "coffee", "nude", "metallic"}
            self.actual_colors = {a for a in self.all_attrs if any(c in a for c in color_words)}

        print(f"  Found {len(self.actual_colors)} color attributes: {sorted(self.actual_colors)[:20]}...")

    def match_garment(self, term: str) -> Set[str]:
        term_lower = term.lower()
        candidates = self.GARMENT_TO_CATEGORY.get(term_lower, [term_lower])
        matched = set()
        for cat in candidates:
            if cat in self.all_cats:
                matched.add(cat)
            for actual_cat in self.all_cats:
                if cat in actual_cat or actual_cat in cat:
                    matched.add(actual_cat)
        return matched

    def match_color(self, color_term: str) -> Set[str]:
        color_lower = color_term.lower()
        synonyms = self.COLOR_SYNONYMS.get(color_lower, [color_lower])
        matched = set()
        for syn in synonyms:
            if syn in self.all_attrs:
                matched.add(syn)
            for actual_attr in self.actual_colors:
                if syn in actual_attr or actual_attr in syn:
                    matched.add(actual_attr)
        return matched

    def match_attributes(self, query: str) -> Tuple[Set[str], Set[str]]:
        query_lower = query.lower()
        matched_cats = set()
        matched_attrs = set()

        for color_term in self.COLOR_SYNONYMS:
            if color_term in query_lower:
                matched_attrs |= self.match_color(color_term)

        for garment_term in self.GARMENT_TO_CATEGORY:
            if garment_term in query_lower:
                matched_cats |= self.match_garment(garment_term)

        words = re.findall(r'[a-z]+', query_lower)
        for word in words:
            if len(word) < 3:
                continue
            for attr in self.all_attrs:
                if word == attr or (len(word) > 4 and word in attr):
                    matched_attrs.add(attr)

        return matched_cats, matched_attrs


class GroundTruthGenerator:
    """Generates ground-truth relevant image IDs for evaluation queries.

    Three-tier approach:
    1. Annotation-based: For attribute/compositional queries, use Fashionpedia
       annotations to find images that match the queried categories + attributes.
    2. Caption-based: For subjective queries (style, context), use VLM captions
       + LLM judgment to score relevance.
    3. Hybrid: Combine both when possible.
    """

    def __init__(
        self,
        annotations_path: Optional[str] = None,
        captions_path: Optional[str] = None,
        use_fashionpedia_api: bool = False,
    ):
        self.captions: Dict[str, Any] = {}
        self.annotation_index: Optional[FashionpediaAnnotationIndex] = None
        self.matcher: Optional[QueryAnnotationMatcher] = None
        self._caption_encoder = None
        self._caption_embeddings: Dict[str, Any] = {}

        if annotations_path and Path(annotations_path).exists():
            self.annotation_index = FashionpediaAnnotationIndex(annotations_path)
            self.matcher = QueryAnnotationMatcher(self.annotation_index)

        if captions_path and Path(captions_path).exists():
            with open(captions_path, "r") as f:
                self.captions = json.load(f)

    def _ensure_caption_embeddings(self) -> None:
        """Lazily load BGE encoder and embed all captions once.

        VLM captions describe runway/portrait content (e.g. "model walks the
        runway...") and rarely contain literal query words like "office" or
        "business". Word-overlap (Jaccard) similarity therefore fails almost
        universally for contextual/style queries. Semantic embedding
        similarity via BGE (same encoder used by the V3 retriever) captures
        paraphrase-level relevance instead.
        """
        if self._caption_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._caption_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

        missing = [
            img_id for img_id in self.captions
            if img_id not in self._caption_embeddings
        ]
        if missing:
            texts = [
                self.captions[img_id].get("combined", self.captions[img_id].get("blip", ""))
                for img_id in missing
            ]
            embs = self._caption_encoder.encode(
                texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
            )
            for img_id, emb in zip(missing, embs):
                self._caption_embeddings[img_id] = emb

    def get_annotation_based_gt(self, query_category: str, query: str) -> Set[str]:
        """Get ground-truth image IDs using Fashionpedia annotations.

        Maps query terms to actual Fashionpedia category/attribute names,
        then finds images whose annotations match.
        """
        if not self.annotation_index or not self.matcher:
            return set()

        matched_cats, matched_attrs = self.matcher.match_attributes(query)

        if not matched_cats and not matched_attrs:
            return set()

        print(f"    Query: '{query}'")
        print(f"    Matched categories: {matched_cats}")
        print(f"    Matched attributes: {matched_attrs}")

        if matched_cats and matched_attrs:
            cat_images = set()
            for cat in matched_cats:
                cat_images |= self.annotation_index.get_images_with_category(cat)
            attr_images = set()
            for attr in matched_attrs:
                attr_images |= self.annotation_index.get_images_with_attribute(attr)
            relevant = cat_images & attr_images
            # If intersection is too small (< 5), fall back to category-only match
            # This handles cases where attribute matching is overly restrictive
            # (e.g., "raincoat" attribute matches only 1 image)
            if len(relevant) < 5 and cat_images:
                relevant = cat_images
        elif matched_cats:
            relevant = set()
            for cat in matched_cats:
                relevant |= self.annotation_index.get_images_with_category(cat)
        elif matched_attrs:
            relevant = set()
            for attr in matched_attrs:
                relevant |= self.annotation_index.get_images_with_attribute(attr)
        else:
            relevant = set()

        print(f"    Found {len(relevant)} relevant images")
        return self.annotation_index.to_filename_stems(relevant)

    def get_caption_based_gt(
        self,
        query: str,
        candidate_ids: List[str],
        threshold: float = 0.55,
    ) -> Set[str]:
        """Get ground-truth using VLM caption semantic similarity.

        Encodes the query and all candidate captions with BGE and computes
        cosine similarity. Word-overlap (Jaccard) was tried first but fails
        almost universally here: VLM captions describe runway/portrait scenes
        and rarely contain literal query words like "office" (only 9/3200
        captions mention it), so max achievable Jaccard similarity was ~0.03.
        Semantic embeddings capture paraphrase-level relevance instead.
        """
        if not self.captions or not candidate_ids:
            return set()

        import numpy as np
        self._ensure_caption_embeddings()

        query_vec = self._caption_encoder.encode(
            [query], normalize_embeddings=True
        )[0]

        relevant = set()
        for img_id in candidate_ids:
            emb = self._caption_embeddings.get(img_id)
            if emb is None:
                continue
            similarity = float(np.dot(query_vec, emb))
            if similarity >= threshold:
                relevant.add(img_id)

        return relevant

    def get_llm_based_gt(
        self,
        query: str,
        candidate_ids: List[str],
        threshold: float = 5.0,
    ) -> Set[str]:
        """Get ground-truth using LLM-based relevance judgment.

        Uses Qwen2.5-0.5B-Instruct to score each candidate image's VLM caption
        against the query on a 0-10 scale. Images scoring above threshold are relevant.

        Falls back to caption-based GT if LLM is not available.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            relevant = set()
            for img_id in candidate_ids:
                if img_id not in self.captions:
                    continue

                caption = self.captions[img_id].get("combined", "")
                if not caption:
                    continue

                prompt = (
                    f"Rate how relevant this image description is to the query.\n"
                    f"Query: '{query}'\n"
                    f"Image description: '{caption}'\n"
                    f"Relevance score (0-10, where 10 is perfect match): "
                )

                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                score_match = re.search(r'(\d+(?:\.\d+)?)', response)
                if score_match:
                    score = float(score_match.group(1))
                    if score >= threshold:
                        relevant.add(img_id)

            del model, tokenizer
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return relevant

        except Exception as e:
            print(f"  LLM not available ({e}), falling back to caption-based GT")
            return self.get_caption_based_gt(query, candidate_ids)

    def get_ground_truth(
        self,
        query_id: str,
        query: str,
        category: str,
        candidate_ids: List[str] = None,
    ) -> Set[str]:
        """Get ground-truth relevant image IDs for a query.

        Strategy:
        - attribute/compositional: annotation-based (exact category + attribute match)
        - contextual/complex: annotation-based for garment part + caption-based for environment
        - style: caption-based or LLM-based (subjective, no annotation match possible)
        """
        if self.annotation_index:
            ann_gt = self.get_annotation_based_gt(category, query)
            if ann_gt:
                if category in ("attribute", "compositional"):
                    return ann_gt
                if candidate_ids and self.captions:
                    cap_gt = self.get_caption_based_gt(query, candidate_ids)
                    return ann_gt | cap_gt
                return ann_gt

        if candidate_ids and self.captions:
            if category == "style":
                return self.get_llm_based_gt(query, candidate_ids)
            else:
                return self.get_caption_based_gt(query, candidate_ids)

        return set()
