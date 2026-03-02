from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from legal_pilot.types import QAExample, SectionRecord, WindowExample


class MLMDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }


class QADataset(Dataset):
    def __init__(
        self,
        examples: list[QAExample],
        section_by_id: dict[str, SectionRecord],
        tokenizer: AutoTokenizer,
        max_length: int,
        with_support: bool,
    ) -> None:
        texts: list[str] = []
        labels: list[int] = []
        for example in examples:
            prompt = example.question
            if with_support and example.support_section_ids:
                support = "\n".join(
                    section_by_id[section_id].section_text
                    for section_id in example.support_section_ids
                    if section_id in section_by_id
                )
                prompt = f"{example.question}\n[SEP]\n{support}"
            texts.append(prompt)
            labels.append(example.answer_index)
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


@dataclass
class TrainArtifacts:
    model_dir: Path
    tokenizer_dir: Path


def train_mlm(
    *,
    model_name: str,
    output_dir: str | Path,
    windows: list[WindowExample],
    config: dict,
) -> TrainArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    texts = [window.text for window in windows]
    dataset = MLMDataset(texts, tokenizer, int(config["model"]["max_length"]))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=float(config["model"]["mlm_probability"]),
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        num_train_epochs=float(config["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        logging_steps=int(config["training"]["logging_steps"]),
        save_total_limit=int(config["training"]["save_total_limit"]),
        fp16=bool(config["training"]["fp16"]),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return TrainArtifacts(model_dir=output_dir, tokenizer_dir=output_dir)


def train_qa_classifier(
    *,
    pretrained_dir: str | Path,
    output_dir: str | Path,
    train_examples: list[QAExample],
    dev_examples: list[QAExample],
    section_by_id: dict[str, SectionRecord],
    config: dict,
    with_support: bool,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    num_labels = max(example.answer_index for example in train_examples + dev_examples) + 1
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_dir,
        num_labels=num_labels,
    )
    train_dataset = QADataset(
        train_examples,
        section_by_id,
        tokenizer,
        int(config["model"]["max_length"]),
        with_support=with_support,
    )
    eval_dataset = QADataset(
        dev_examples,
        section_by_id,
        tokenizer,
        int(config["model"]["max_length"]),
        with_support=with_support,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        num_train_epochs=2.0,
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        logging_steps=int(config["training"]["logging_steps"]),
        report_to=[],
    )

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = float((preds == labels).mean())
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer.evaluate()
