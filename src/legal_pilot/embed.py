from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from legal_pilot.types import SectionRecord


def build_similarity_index(sections: list[SectionRecord]) -> dict[str, list[str]]:
    texts = [f"{section.section_title}\n{section.section_text}" for section in sections]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix)
    neighbors: dict[str, list[str]] = {}
    for i, section in enumerate(sections):
        ranked = sorted(
            range(len(sections)),
            key=lambda j: sim[i, j],
            reverse=True,
        )
        neighbors[section.section_id] = [
            sections[j].section_id for j in ranked if sections[j].section_id != section.section_id
        ]
    return neighbors
