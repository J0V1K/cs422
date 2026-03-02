from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
from transformers import AutoTokenizer

from legal_pilot.types import SectionRecord, WindowExample


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
            joined = "\n[SEP]\n".join(text_parts + [candidate_text])
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
                joined = "\n[SEP]\n".join(text_parts + [format_section(fallback)])
                token_count = len(tokenizer(joined, truncation=False)["input_ids"])
                if token_count > max_length:
                    continue
                chosen_ids.append(fallback_id)
                text_parts.append(format_section(fallback))
                exposure_count[fallback_id] += 1
                if len(chosen_ids) >= min_sections_per_window:
                    break

        final_text = "\n[SEP]\n".join(text_parts)
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
    if condition == "random":
        return list(section_by_id.keys())
    if condition == "embed-sim":
        return similarity_index.get(anchor_id, [])
    if condition == "cite-graph":
        references: list[str] = []
        next_sections: list[str] = []
        same_chapter: list[str] = []
        for neighbor_id in graph.neighbors(anchor_id):
            edge_type = graph.edges[anchor_id, neighbor_id].get("edge_type")
            if edge_type == "REFERENCES":
                references.append(neighbor_id)
            elif edge_type == "NEXT_SECTION":
                next_sections.append(neighbor_id)
            elif edge_type == "SAME_CHAPTER":
                same_chapter.append(neighbor_id)
        return references + next_sections + same_chapter
    raise ValueError(f"Unsupported condition: {condition}")
