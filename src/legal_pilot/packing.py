from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
from transformers import AutoTokenizer

from legal_pilot.graphing import has_edge_type
from legal_pilot.types import SectionRecord, WindowExample

SEPARATOR = "\n[SEP]\n"


def format_section(section: SectionRecord) -> str:
    return (
        f"[SECTION] {section.code_name} {section.section_number}: {section.section_title}\n"
        f"{section.section_text}"
    )


def generate_windows(
    condition: str,
    sections: list[SectionRecord],
    graph: nx.Graph,
    similarity_index: dict[str, list[str]],
    model_name: str,
    max_length: int,
    min_sections_per_window: int,
    max_sections_per_window: int,
    target_exposures_per_section: int,
    seed: int,
) -> list[WindowExample]:
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    section_by_id = {section.section_id: section for section in sections}
    section_ids = list(section_by_id.keys())
    rng.shuffle(section_ids)
    exposure_count: dict[str, int] = defaultdict(int)
    windows: list[WindowExample] = []

    for anchor_id in section_ids:
        anchor = section_by_id[anchor_id]
        chosen_ids = [anchor.section_id]
        exposure_count[anchor.section_id] += 1
        text_parts = [format_section(anchor)]

        for candidate_id in _candidate_ids(condition, anchor_id, graph, similarity_index, section_by_id):
            if candidate_id in chosen_ids:
                continue
            if exposure_count[candidate_id] >= target_exposures_per_section:
                continue
            candidate = section_by_id[candidate_id]
            candidate_text = format_section(candidate)
            joined = SEPARATOR.join(text_parts + [candidate_text])
            token_count = len(tokenizer(joined, truncation=False)["input_ids"])
            if token_count > max_length:
                continue
            chosen_ids.append(candidate_id)
            text_parts.append(candidate_text)
            exposure_count[candidate_id] += 1
            if len(chosen_ids) >= max_sections_per_window:
                break

        if len(chosen_ids) < min_sections_per_window:
            for fallback_id in section_ids:
                if fallback_id in chosen_ids:
                    continue
                if exposure_count[fallback_id] >= target_exposures_per_section:
                    continue
                fallback = section_by_id[fallback_id]
                joined = SEPARATOR.join(text_parts + [format_section(fallback)])
                token_count = len(tokenizer(joined, truncation=False)["input_ids"])
                if token_count > max_length:
                    continue
                chosen_ids.append(fallback_id)
                text_parts.append(format_section(fallback))
                exposure_count[fallback_id] += 1
                if len(chosen_ids) >= min_sections_per_window:
                    break

        final_text = SEPARATOR.join(text_parts)
        final_token_count = len(tokenizer(final_text, truncation=False)["input_ids"])
        windows.append(
            WindowExample(
                condition=condition,
                anchor_id=anchor_id,
                section_ids=chosen_ids,
                text=final_text,
                token_count=final_token_count,
            )
        )
    return windows


def _candidate_ids(
    condition: str,
    anchor_id: str,
    graph: nx.Graph,
    similarity_index: dict[str, list[str]],
    section_by_id: dict[str, SectionRecord],
) -> list[str]:
    if condition == 'random':
        return list(section_by_id.keys())
    if condition == 'embed-sim':
        return similarity_index.get(anchor_id, [])
    if condition == 'cite-graph':
        return _ordered_unique(
            _neighbors_by_edge_type(graph, anchor_id, 'REFERENCES', section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, 'NEXT_SECTION', section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, 'SAME_ARTICLE', section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, 'SAME_CHAPTER', section_by_id),
            section_by_id,
        )
    if condition == 'hierarchy-pack':
        return _ordered_unique(
            _neighbors_by_edge_type(graph, anchor_id, 'SAME_ARTICLE', section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, 'NEXT_SECTION', section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, 'SAME_CHAPTER', section_by_id),
            section_by_id,
        )
    raise ValueError(f'Unsupported condition: {condition}')


def _neighbors_by_edge_type(
    graph: nx.Graph,
    anchor_id: str,
    edge_type: str,
    section_by_id: dict[str, SectionRecord],
) -> list[str]:
    matches = [
        neighbor_id
        for neighbor_id in graph.neighbors(anchor_id)
        if has_edge_type(graph, anchor_id, neighbor_id, edge_type)
    ]
    return _sort_section_ids(matches, section_by_id)


def _ordered_unique(section_ids: list[str], section_by_id: dict[str, SectionRecord]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for section_id in section_ids:
        if section_id in seen:
            continue
        if section_id not in section_by_id:
            continue
        seen.add(section_id)
        ordered.append(section_id)
    return _sort_section_ids(ordered, section_by_id)


def _sort_section_ids(section_ids: list[str], section_by_id: dict[str, SectionRecord]) -> list[str]:
    return sorted(
        section_ids,
        key=lambda section_id: (
            section_by_id[section_id].code_name,
            section_by_id[section_id].chapter,
            section_by_id[section_id].article,
            _section_sort_key(section_by_id[section_id].section_number),
        ),
    )


def _section_sort_key(value: str) -> tuple[int, ...]:
    parts = value.split('.')
    return tuple(int(part) if part.isdigit() else 0 for part in parts)
