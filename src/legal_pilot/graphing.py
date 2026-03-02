from __future__ import annotations

from pathlib import Path

import networkx as nx

from legal_pilot.types import CitationEdge, SectionRecord


def build_section_graph(sections: list[SectionRecord], edges: list[CitationEdge]) -> nx.Graph:
    graph = nx.Graph()
    by_code: dict[str, list[SectionRecord]] = {}
    by_chapter: dict[tuple[str, str], list[SectionRecord]] = {}
    by_article: dict[tuple[str, str, str], list[SectionRecord]] = {}
    for section in sections:
        graph.add_node(
            section.section_id,
            code_name=section.code_name,
            chapter=section.chapter,
            article=section.article,
            section_number=section.section_number,
            section_title=section.section_title,
        )
        by_code.setdefault(section.code_name, []).append(section)
        by_chapter.setdefault((section.code_name, section.chapter), []).append(section)
        by_article.setdefault((section.code_name, section.chapter, section.article), []).append(section)
    for edge in edges:
        _add_edge_type(graph, edge.source_id, edge.target_id, 'REFERENCES')
    for code_sections in by_code.values():
        sorted_sections = sorted(code_sections, key=lambda item: _section_sort_key(item.section_number))
        for left, right in zip(sorted_sections, sorted_sections[1:]):
            _add_edge_type(graph, left.section_id, right.section_id, 'NEXT_SECTION')
    for article_sections in by_article.values():
        ids = [section.section_id for section in article_sections if section.article]
        for i, source_id in enumerate(ids):
            for target_id in ids[i + 1 :]:
                _add_edge_type(graph, source_id, target_id, 'SAME_ARTICLE')
    for chapter_sections in by_chapter.values():
        ids = [section.section_id for section in chapter_sections]
        for i, source_id in enumerate(ids):
            for target_id in ids[i + 1 :]:
                _add_edge_type(graph, source_id, target_id, 'SAME_CHAPTER')
    return graph


def compute_graph_stats(graph: nx.Graph) -> dict[str, object]:
    edge_type_counts: dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        for edge_type in attrs.get('edge_types', []):
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    components = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    reference_degrees = []
    for node_id in graph.nodes():
        degree = sum(1 for neighbor_id in graph.neighbors(node_id) if has_edge_type(graph, node_id, neighbor_id, 'REFERENCES'))
        reference_degrees.append(degree)
    reference_degrees.sort(reverse=True)
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'edge_type_counts': edge_type_counts,
        'num_components': len(components),
        'largest_component_size': components[0] if components else 0,
        'top_reference_degrees': reference_degrees[:10],
    }


def save_citation_graph_visualization(
    graph: nx.Graph,
    output_path: str | Path,
    *,
    max_nodes: int = 40,
    seed: int = 42,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_graph = nx.Graph()
    for node_id, attrs in graph.nodes(data=True):
        reference_graph.add_node(node_id, **attrs)
    for source_id, target_id, attrs in graph.edges(data=True):
        if 'REFERENCES' in attrs.get('edge_types', []):
            reference_graph.add_edge(source_id, target_id)

    if reference_graph.number_of_edges() == 0:
        viz_graph = graph.copy()
    else:
        active_nodes = [node_id for node_id, degree in reference_graph.degree() if degree > 0]
        viz_graph = reference_graph.subgraph(active_nodes).copy()

    if viz_graph.number_of_nodes() > max_nodes:
        top_nodes = [node_id for node_id, _ in sorted(viz_graph.degree, key=lambda item: item[1], reverse=True)[:max_nodes]]
        viz_graph = viz_graph.subgraph(top_nodes).copy()

    if viz_graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No citation edges available', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    chapter_to_color: dict[str, int] = {}
    node_colors = []
    for node_id in viz_graph.nodes():
        chapter = viz_graph.nodes[node_id].get('chapter') or 'Unknown'
        if chapter not in chapter_to_color:
            chapter_to_color[chapter] = len(chapter_to_color)
        node_colors.append(chapter_to_color[chapter])

    labels = {
        node_id: viz_graph.nodes[node_id].get('section_number', node_id)
        for node_id in viz_graph.nodes()
    }
    positions = nx.spring_layout(viz_graph, seed=seed)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(
        viz_graph,
        pos=positions,
        node_color=node_colors,
        cmap=plt.cm.Set2,
        node_size=900,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(viz_graph, pos=positions, width=1.5, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(viz_graph, pos=positions, labels=labels, font_size=7, ax=ax)
    ax.set_title('Citation Graph (reference-edge subgraph)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def has_edge_type(graph: nx.Graph, source_id: str, target_id: str, edge_type: str) -> bool:
    if not graph.has_edge(source_id, target_id):
        return False
    return edge_type in graph.edges[source_id, target_id].get('edge_types', [])


def _add_edge_type(graph: nx.Graph, source_id: str, target_id: str, edge_type: str) -> None:
    if graph.has_edge(source_id, target_id):
        edge_types = set(graph.edges[source_id, target_id].get('edge_types', []))
        edge_types.add(edge_type)
        graph.edges[source_id, target_id]['edge_types'] = sorted(edge_types)
        return
    graph.add_edge(source_id, target_id, edge_types=[edge_type])


def _section_sort_key(value: str) -> tuple[int, ...]:
    parts = value.split('.')
    return tuple(int(part) if part.isdigit() else 0 for part in parts)
