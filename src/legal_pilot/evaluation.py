from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from legal_pilot.training import QADataset
from legal_pilot.types import CitationEdge, QAExample, SectionRecord


def compute_probe_metrics(
    *,
    model_dir: str | Path,
    sections: list[SectionRecord],
    probe_edges: list[CitationEdge],
    candidate_pool_size: int,
    recall_ks: list[int],
    seed: int,
) -> dict[str, float]:
    if not probe_edges:
        return {"num_edges": 0}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    section_by_id = {section.section_id: section for section in sections}
    embeddings = _embed_sections(model, tokenizer, sections)
    rng = random.Random(seed)
    section_ids = list(section_by_id.keys())
    ranks: list[int] = []
    for edge in probe_edges:
        if edge.source_id not in embeddings or edge.target_id not in embeddings:
            continue
        source_vec = embeddings[edge.source_id]
        negatives = [item for item in section_ids if item not in {edge.source_id, edge.target_id}]
        rng.shuffle(negatives)
        candidates = [edge.target_id] + negatives[: max(0, candidate_pool_size - 1)]
        scores = []
        for candidate_id in candidates:
            candidate_vec = embeddings[candidate_id]
            score = float(np.dot(source_vec, candidate_vec))
            scores.append((candidate_id, score))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        rank = 1 + next(i for i, (candidate_id, _) in enumerate(ranked) if candidate_id == edge.target_id)
        ranks.append(rank)

    if not ranks:
        return {"num_edges": 0}

    metrics: dict[str, float] = {
        "num_edges": float(len(ranks)),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
    }
    for k in recall_ks:
        metrics[f"recall@{k}"] = float(np.mean([1.0 if rank <= k else 0.0 for rank in ranks]))
    return metrics


def _embed_sections(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    sections: list[SectionRecord],
) -> dict[str, np.ndarray]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for section in sections:
            text = f"{section.section_title}\n{section.section_text}"
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings[section.section_id] = pooled.squeeze(0).cpu().numpy()
    return embeddings


def split_qa_examples(examples: list[QAExample]) -> dict[str, list[QAExample]]:
    buckets: dict[str, list[QAExample]] = defaultdict(list)
    for example in examples:
        buckets[example.split].append(example)
    return buckets


def evaluate_qa_classifier(
    *,
    classifier_dir: str | Path,
    test_examples: list[QAExample],
    section_by_id: dict[str, SectionRecord],
    max_length: int,
    with_support: bool,
) -> dict[str, float]:
    if not test_examples:
        return {"num_examples": 0}
    tokenizer = AutoTokenizer.from_pretrained(classifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_dir)
    dataset = QADataset(
        test_examples,
        section_by_id,
        tokenizer,
        max_length,
        with_support=with_support,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(classifier_dir) / "eval_tmp"),
            per_device_eval_batch_size=8,
            report_to=[],
        ),
    )
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = logits.argmax(axis=-1)
    accuracy = float((preds == labels).mean())
    confidence = _softmax(logits).max(axis=-1)
    ece = _expected_calibration_error(confidence, preds, labels)
    return {"num_examples": float(len(test_examples)), "accuracy": accuracy, "ece": ece}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _expected_calibration_error(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & (confidence < right)
        if not mask.any():
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = (preds[mask] == labels[mask]).mean()
        ece += float(abs(bin_acc - bin_conf) * (mask.sum() / len(labels)))
    return ece
