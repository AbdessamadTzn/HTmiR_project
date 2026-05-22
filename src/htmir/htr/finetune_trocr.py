"""Fine-tuning de TrOCR avec LoRA pour manuscrits médiévaux."""

from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

from htmir.evaluation.metrics import corpus_cer
from htmir.utils.logger import get_logger
from htmir.utils.seeds import fixer_seeds

logger = get_logger(__name__)
MODEL_ID = "microsoft/trocr-base-handwritten"


@dataclass
class TrOCRLoRAConfig:
    """Configuration du fine-tuning TrOCR + LoRA.

    Args:
        lora_r: Rang LoRA (8 ou 16 selon le brief).
        lora_alpha: Paramètre alpha LoRA.
        lora_dropout: Dropout LoRA.
        learning_rate: Taux d'apprentissage.
        num_epochs: Nombre d'époques.
        batch_size: Taille de batch.
        output_dir: Dossier de sortie pour les checkpoints.
    """
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 5e-5
    num_epochs: int = 10
    batch_size: int = 8
    output_dir: Path = Path("outputs/trocr_lora")
    seed: int = 42


class ManuscriptLineDataset(Dataset):
    """Dataset de lignes de manuscrits pour TrOCR.

    Args:
        image_paths: Chemins vers les images de lignes.
        transcriptions: Transcriptions correspondantes.
        processor: TrOCRProcessor pré-chargé.
        max_target_length: Longueur maximale des séquences cible.
    """

    def __init__(
        self,
        image_paths: list[Path],
        transcriptions: list[str],
        processor: TrOCRProcessor,
        max_target_length: int = 128,
    ) -> None:
        self.image_paths = image_paths
        self.transcriptions = transcriptions
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        encoding = self.processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.squeeze()
        labels = self.processor.tokenizer(
            self.transcriptions[idx],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def load_trocr_with_lora(config: TrOCRLoRAConfig) -> tuple:
    """Charge TrOCR et applique les adaptateurs LoRA.

    Args:
        config: Configuration LoRA et fine-tuning.

    Returns:
        Tuple (model, processor) prêts pour l'entraînement.

    Example:
        >>> model, processor = load_trocr_with_lora(TrOCRLoRAConfig(lora_r=8))
    """
    fixer_seeds(config.seed)
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    # Configuration LoRA sur le décodeur
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(f"TrOCR + LoRA chargé (r={config.lora_r})")
    return model, processor


def finetune_trocr(
    train_images: list[Path],
    train_texts: list[str],
    val_images: list[Path],
    val_texts: list[str],
    config: TrOCRLoRAConfig | None = None,
) -> VisionEncoderDecoderModel:
    """Fine-tune TrOCR avec LoRA sur le corpus préparé.

    Args:
        train_images: Images d'entraînement (lignes de manuscrit).
        train_texts: Transcriptions d'entraînement.
        val_images: Images de validation.
        val_texts: Transcriptions de validation.
        config: Configuration du fine-tuning.

    Returns:
        Modèle fine-tuné.

    Example:
        >>> model = finetune_trocr(train_imgs, train_txts, val_imgs, val_txts)
    """
    config = config or TrOCRLoRAConfig()
    model, processor = load_trocr_with_lora(config)

    train_dataset = ManuscriptLineDataset(train_images, train_texts, processor)
    val_dataset = ManuscriptLineDataset(val_images, val_texts, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        seed=config.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    logger.info("Fine-tuning TrOCR terminé.")
    return model
