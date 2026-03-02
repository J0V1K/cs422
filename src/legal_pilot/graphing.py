from __future__ import annotations

import networkx as nx

from legal_pilot.types import CitationEdge, SectionRecord


def build_section_graph(sections: list[SectionRecord], edges: list[CitationEdge]) -> nx.Graph:
    graph = nx.Graph()
    by_code: dict[str, list[SectionRecord]] = {}
    by_chapter: dict[tuple[str, str], list[SectionRecord]] = {}
    for section in sections:
        graph.add_node(
            section.section_id,
            code_name=section.code_name,
            chapter=section.chapter,
            article=section.article,
            section_number=section.section_number,
        )
        by_code.setdefault(section.code_name, []).append(section)
        by_chapter.setdefault((section.code_name, section.chapter), []).append(section)
    for edge in edges:
        graph.add_edge(edge.source_id, edge.target_id, edge_type="REFERENCES")
    for code_sections in by_code.values():
        sorted_sections = sorted(code_sections, key=lambda item: _section_sort_key(item.section_number))
        for left, right in zip(sorted_sections, sorted_sections[1:]):
            graph.add_edge(left.section_id, right.section_id, edge_type="NEXT_SECTION")
    for chapter_sections in by_chapter.values():
        ids = [section.section_id for section in chapter_sections]
        for i, source_id in enumerate(ids):
            for target_id in ids[i + 1 :]:
                graph.add_edge(source_id, target_id, edge_type="SAME_CHAPTER")
    return graph


def _section_sort_key(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    return tuple(int(part) if part.isdigit() else 0 for part in parts)
