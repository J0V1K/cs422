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
        anchor_text = format_section(anchor)
        anchor_text, _, anchor_was_truncated = _fit_to_max_length(tokenizer, anchor_text, max_length)
        chosen_ids = [anchor.section_id]
        chosen_sources = ["ANCHOR_TRUNCATED" if anchor_was_truncated else "ANCHOR"]
        exposure_count[anchor.section_id] += 1
        text_parts = [anchor_text]

        for candidate_id, candidate_source in _candidate_entries(
            condition,
            anchor_id,
            graph,
            similarity_index,
            section_by_id,
            section_ids,
        ):
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
            chosen_sources.append(candidate_source)
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
                chosen_sources.append("FALLBACK")
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
                section_sources=chosen_sources,
                text=final_text,
                token_count=final_token_count,
            )
        )
    return windows


def summarize_windows(windows: list[WindowExample]) -> dict[str, object]:
    source_counts: dict[str, int] = defaultdict(int)
    non_anchor_source_counts: dict[str, int] = defaultdict(int)
    num_windows_with_fallback = 0
    total_sections = 0
    total_non_anchor_sections = 0

    for window in windows:
        total_sections += len(window.section_ids)
        if "FALLBACK" in window.section_sources:
            num_windows_with_fallback += 1
        for source in window.section_sources:
            source_counts[source] += 1
        for source in window.section_sources[1:]:
            non_anchor_source_counts[source] += 1
            total_non_anchor_sections += 1

    return {
        "num_windows": len(windows),
        "avg_token_count": (
            sum(window.token_count for window in windows) / len(windows) if windows else 0.0
        ),
        "avg_sections_per_window": (total_sections / len(windows)) if windows else 0.0,
        "num_windows_with_fallback": num_windows_with_fallback,
        "fallback_window_fraction": (
            num_windows_with_fallback / len(windows) if windows else 0.0
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "non_anchor_source_counts": dict(sorted(non_anchor_source_counts.items())),
        "non_anchor_source_fractions": {
            source: count / total_non_anchor_sections
            for source, count in sorted(non_anchor_source_counts.items())
        },
    }


def _candidate_entries(
    condition: str,
    anchor_id: str,
    graph: nx.Graph,
    similarity_index: dict[str, list[str]],
    section_by_id: dict[str, SectionRecord],
    shuffled_section_ids: list[str],
) -> list[tuple[str, str]]:
    if condition == "random":
        return [
            (section_id, "RANDOM_POOL")
            for section_id in shuffled_section_ids
            if section_id in section_by_id
        ]
    if condition == "embed-sim":
        return [
            (section_id, "EMBED_SIM")
            for section_id in similarity_index.get(anchor_id, [])
            if section_id in section_by_id
        ]
    if condition == "citation-only":
        return _ordered_unique(
            _neighbors_by_edge_type(graph, anchor_id, "REFERENCES", section_by_id)
        )
    if condition == "cite-graph":
        return _ordered_unique(
            _neighbors_by_edge_type(graph, anchor_id, "REFERENCES", section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, "NEXT_SECTION", section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, "SAME_ARTICLE", section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, "SAME_CHAPTER", section_by_id)
        )
    if condition == "hierarchy-pack":
        return _ordered_unique(
            _neighbors_by_edge_type(graph, anchor_id, "SAME_ARTICLE", section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, "NEXT_SECTION", section_by_id)
            + _neighbors_by_edge_type(graph, anchor_id, "SAME_CHAPTER", section_by_id)
        )
    raise ValueError(f"Unsupported condition: {condition}")


def _neighbors_by_edge_type(
    graph: nx.Graph,
    anchor_id: str,
    edge_type: str,
    section_by_id: dict[str, SectionRecord],
) -> list[tuple[str, str]]:
    matches = [
        neighbor_id
        for neighbor_id in graph.neighbors(anchor_id)
        if has_edge_type(graph, anchor_id, neighbor_id, edge_type)
    ]
    sorted_matches = _sort_section_ids(matches, section_by_id)
    return [(neighbor_id, edge_type) for neighbor_id in sorted_matches]


def _ordered_unique(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for section_id, source in entries:
        if section_id in seen:
            continue
        seen.add(section_id)
        ordered.append((section_id, source))
    return ordered


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
    parts = value.split(".")
    return tuple(int(part) if part.isdigit() else 0 for part in parts)


def _fit_to_max_length(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
) -> tuple[str, int, bool]:
    input_ids = tokenizer(text, truncation=False)["input_ids"]
    token_count = len(input_ids)
    if token_count <= max_length:
        return text, token_count, False
    truncated_ids = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    truncated_text = tokenizer.decode(
        truncated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return truncated_text, len(truncated_ids), True
