from __future__ import annotations

import re
from typing import Iterable

from datasets import load_dataset

from legal_pilot.types import SectionRecord

SECTION_TITLE_RE = re.compile(r"Section\s+([0-9A-Za-z.\-]+)")


def load_statecodes_sections(config: dict) -> list[SectionRecord]:
    data_config = config["data"]
    hf_config = data_config["hf"]
    state = hf_config.get("state", "CA")
    code_filters = tuple(hf_config.get("code_filters", []))
    include_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in hf_config.get("include_patterns", [])]
    exclude_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in hf_config.get("exclude_patterns", [])]
    limit = hf_config.get("limit")

    dataset = load_dataset(
        hf_config.get("dataset_name", "reglab/statecodes"),
        hf_config.get("config_name", "all_codes"),
        split=hf_config.get("split", "train"),
        streaming=True,
    )

    sections: list[SectionRecord] = []
    for row in dataset:
        if row.get("state") != state:
            continue
        title = row.get("title", "")
        path = row.get("path", "")
        url = row.get("url", "")
        haystack = " ".join([title, path, url])
        if code_filters and not any(code_filter in haystack for code_filter in code_filters):
            continue
        if include_patterns and not any(pattern.search(haystack) for pattern in include_patterns):
            continue
        if exclude_patterns and any(pattern.search(haystack) for pattern in exclude_patterns):
            continue
        section = _normalize_row(row)
        if not section:
            continue
        sections.append(section)
        if limit and len(sections) >= int(limit):
            break
    return sections


def _normalize_row(row: dict) -> SectionRecord | None:
    title = row.get("title", "")
    parts = [part.strip() for part in title.split("›") if part.strip()]
    if len(parts) < 2:
        return None
    code_name = parts[1]
    chapter = _first_prefixed(parts, "CHAPTER")
    article = _first_prefixed(parts, "ARTICLE")
    section_label = parts[-1]
    match = SECTION_TITLE_RE.search(section_label)
    if not match:
        content_match = SECTION_TITLE_RE.search(row.get("content", ""))
        if not content_match:
            return None
        section_number = _normalize_section_number(content_match.group(1))
    else:
        section_number = _normalize_section_number(match.group(1))
    section_id = _build_section_id(code_name, section_number)
    section_title = section_label.rstrip(".")
    return SectionRecord(
        section_id=section_id,
        code_name=code_name,
        chapter=chapter,
        article=article,
        section_number=section_number,
        section_title=section_title,
        section_text=row.get("content", "").strip(),
        effective_date=None,
    )


def _first_prefixed(parts: Iterable[str], prefix: str) -> str:
    for part in parts:
        if part.startswith(prefix):
            return part
    return ""


def _normalize_section_number(value: str) -> str:
    return value.replace("-", ".").rstrip(".")


def _build_section_id(code_name: str, section_number: str) -> str:
    code_stub = code_name.lower()
    code_stub = re.sub(r"[^a-z0-9]+", "_", code_stub).strip("_")
    number_stub = re.sub(r"[^a-z0-9]+", "_", section_number.lower()).strip("_")
    return f"{code_stub}_{number_stub}"
