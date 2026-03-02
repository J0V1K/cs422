from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SectionRecord:
    section_id: str
    code_name: str
    chapter: str
    article: str
    section_number: str
    section_title: str
    section_text: str
    effective_date: str | None = None


@dataclass(frozen=True)
class CitationEdge:
    source_id: str
    target_id: str
    citation_text: str
    citation_type: str


@dataclass(frozen=True)
class QAExample:
    example_id: str
    split: str
    question: str
    choices: list[str]
    answer_index: int
    support_section_ids: list[str] = field(default_factory=list)


@dataclass
class WindowExample:
    condition: str
    anchor_id: str
    section_ids: list[str]
    section_sources: list[str]
    text: str
    token_count: int


JSONDict = Dict[str, Any]
