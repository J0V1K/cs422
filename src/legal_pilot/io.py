from __future__ import annotations

import json
from pathlib import Path

from legal_pilot.sample_data import build_sample_qa, build_sample_sections
from legal_pilot.statecodes import load_statecodes_sections
from legal_pilot.types import JSONDict, QAExample, SectionRecord, WindowExample


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_sections(config: JSONDict) -> list[SectionRecord]:
    mode = config["data"]["mode"]
    if mode == "sample":
        return build_sample_sections()
    if mode == "hf_statecodes":
        return load_statecodes_sections(config)
    if mode != "jsonl":
        raise ValueError(f"Unsupported data mode: {mode}")
    path = config["data"]["sections_path"]
    if not path:
        raise ValueError("data.sections_path is required for jsonl mode")
    return load_section_jsonl(path)


def load_qa(config: JSONDict) -> list[QAExample]:
    mode = config["data"]["mode"]
    if mode == "sample":
        return build_sample_qa()
    path = config["data"].get("qa_path")
    if not path:
        return []
    examples: list[QAExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                QAExample(
                    example_id=row["id"],
                    split=row["split"],
                    question=row["question"],
                    choices=row["choices"],
                    answer_index=int(row["answer_index"]),
                    support_section_ids=row.get("support_section_ids", []),
                )
            )
    return examples


def save_json(path: str | Path, payload: JSONDict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_jsonl(path: str | Path, rows: list[JSONDict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_windows(path: str | Path, windows: list[WindowExample]) -> None:
    rows = [
        {
            "condition": item.condition,
            "anchor_id": item.anchor_id,
            "section_ids": item.section_ids,
            "text": item.text,
            "token_count": item.token_count,
        }
        for item in windows
    ]
    save_jsonl(path, rows)


def section_to_row(section: SectionRecord) -> JSONDict:
    return {
        "section_id": section.section_id,
        "code_name": section.code_name,
        "chapter": section.chapter,
        "article": section.article,
        "section_number": section.section_number,
        "section_title": section.section_title,
        "section_text": section.section_text,
        "effective_date": section.effective_date,
    }


def load_section_jsonl(path: str | Path) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append(
                SectionRecord(
                    section_id=row["section_id"],
                    code_name=row.get("code_name", ""),
                    chapter=row.get("chapter", ""),
                    article=row.get("article", ""),
                    section_number=row["section_number"],
                    section_title=row.get("section_title", ""),
                    section_text=row["section_text"],
                    effective_date=row.get("effective_date"),
                )
            )
    return records


def save_sections_jsonl(path: str | Path, sections: list[SectionRecord]) -> None:
    save_jsonl(path, [section_to_row(section) for section in sections])
