"""Inférence baseline TrOCR sans fine-tuning (mesure avant entraînement Vinci)."""

from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from htmir.utils.logger import get_logger
from htmir.utils.seeds import fixer_seeds

logger = get_logger(__name__)
MODEL_ID = "microsoft/trocr-base-handwritten"


class TrOCRBaseline:
    """Wrapper léger pour transcrire des lignes ou pages."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        fixer_seeds(42)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"TrOCR baseline chargé ({model_id}) sur {self.device}")

    def transcribe_image(self, image_path: Path | str, max_length: int = 128) -> tuple[str, float]:
        """Transcrit une image de ligne et renvoie (texte, confiance_proxy).

        La confiance est une heuristique basée sur la longueur de la séquence générée
        (à remplacer par scores du décodeur lors du fine-tuning).
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pixel_values, max_length=max_length)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        conf = min(1.0, max(0.3, len(text.strip()) / 40.0))
        return text, conf

    def transcribe_batch(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        return [self.transcribe_image(p) for p in image_paths]
