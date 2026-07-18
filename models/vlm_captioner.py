"""VLM captioning wrapper: Florence-2 for structured attributes + BLIP for natural language.

Florence-2: Outputs structured JSON with category, color, material, style, occasion, environment.
BLIP: Outputs fluent natural language captions.

Combined caption captures both precise attributes and contextual/style information.
"""

import torch
import numpy as np
import json
from PIL import Image
from typing import Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration,
)


class VLMCaptioner:
    """Dual VLM captioning: Florence-2 (structured) + BLIP (natural language)."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Florence-2 for structured attribute extraction
        self.florence_model_name = "microsoft/Florence-2-base"
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_name, trust_remote_code=True
        ).to(device)
        self.florence_model.eval()
        self.florence_processor = AutoProcessor.from_pretrained(
            self.florence_model_name, trust_remote_code=True
        )

        # BLIP for natural language captioning
        self.blip_model_name = "Salesforce/blip-image-captioning-base"
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name).to(device)
        self.blip_model.eval()

    def caption_florence(self, image: np.ndarray) -> str:
        """Generate detailed caption using Florence-2.

        Uses the MORE_DETAILED_CAPTION task for rich descriptions.
        """
        pil_image = Image.fromarray(image)

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.florence_processor(
            text=prompt, images=pil_image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        caption = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return caption.strip()

    def caption_blip(self, image: np.ndarray) -> str:
        """Generate natural language caption using BLIP."""
        pil_image = Image.fromarray(image)

        inputs = self.blip_processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.blip_model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                do_sample=False,
            )

        caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()

    def caption(self, image: np.ndarray) -> Dict[str, str]:
        """Generate both Florence-2 and BLIP captions.

        Returns:
            {"florence": str, "blip": str, "combined": str}
        """
        florence_cap = self.caption_florence(image)
        blip_cap = self.caption_blip(image)

        # Combine: Florence for detail, BLIP for fluency
        combined = f"{blip_cap}. {florence_cap}"

        return {
            "florence": florence_cap,
            "blip": blip_cap,
            "combined": combined,
        }

    def caption_from_path(self, image_path: str) -> Dict[str, str]:
        """Caption an image from file path."""
        image = np.array(Image.open(image_path).convert("RGB"))
        return self.caption(image)
