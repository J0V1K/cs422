from __future__ import annotations

import re

from legal_pilot.types import CitationEdge, SectionRecord

SECTION_PATTERN = re.compile(
    r"(?:Section|Sections|Sec\.|§)\s+(\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def classify_citation(context: str) -> str:
    lowered = context.lower()
    if "defined in" in lowered:
        return "definitional"
    if "except as provided" in lowered:
        return "exception"
    if "subject to" in lowered:
        return "incorporation"
    if "punish" in lowered:
        return "penalty"
    if "procedure" in lowered or "accordance with" in lowered:
        return "procedural"
    return "generic"


def extract_citation_edges(sections: list[SectionRecord]) -> list[CitationEdge]:
    number_to_id = {section.section_number: section.section_id for section in sections}
    edges: list[CitationEdge] = []
    for section in sections:
        for match in SECTION_PATTERN.finditer(section.section_text):
            cited_number = match.group(1)
            target_id = number_to_id.get(cited_number)
            if not target_id or target_id == section.section_id:
                continue
            start = max(0, match.start() - 40)
            end = min(len(section.section_text), match.end() + 40)
            context = section.section_text[start:end]
            edges.append(
                CitationEdge(
                    source_id=section.section_id,
                    target_id=target_id,
                    citation_text=match.group(0),
                    citation_type=classify_citation(context),
                )
            )
    return edges
