from __future__ import annotations

import random

from legal_pilot.types import CitationEdge


def build_leakage_safe_split(
    node_ids: list[str],
    citation_edges: list[CitationEdge],
    holdout_fraction: float,
    exclusion_hops: int,
    seed: int,
    preferred_holdout_ids: list[str] | None = None,
) -> dict[str, set[str]]:
    rng = random.Random(seed)
    preferred = [node_id for node_id in (preferred_holdout_ids or []) if node_id in set(node_ids)]
    target_holdout = max(1, int(len(node_ids) * holdout_fraction), len(preferred))
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in citation_edges:
        adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
        adjacency.setdefault(edge.target_id, set()).add(edge.source_id)

    holdout: set[str] = set()
    for node_id in preferred:
        holdout.add(node_id)

    remaining = [node_id for node_id in node_ids if node_id not in holdout]
    rng.shuffle(remaining)
    for node_id in remaining:
        if len(holdout) >= target_holdout:
            break
        holdout.add(node_id)

    excluded: set[str] = set(holdout)
    frontier = set(holdout)
    for _ in range(exclusion_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(adjacency.get(node, set()))
        excluded.update(next_frontier)
        frontier = next_frontier

    train = set(node_ids) - excluded
    return {"train": train, "holdout": holdout, "excluded": excluded}
