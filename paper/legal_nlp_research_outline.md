# Structure-Aware Pretraining for Statutory Reasoning

## Abstract

Legal reasoning over statutes depends on explicit relationships among sections,
including definitions, exceptions, procedural prerequisites, and cross-
references. Standard language model pretraining rarely preserves these
relationships in a controlled way, especially for short-context encoders that
still underpin many practical legal NLP systems. This paper proposes a
structure-aware pretraining framework for legal text in which masked-language
model windows are packed according to document relationships rather than random
document order alone. We focus on a California housing-law slice derived from
`reglab/statecodes` and compare four ordering conditions: random packing,
embedding-similarity packing, hierarchy-only packing, and citation-aware graph
packing. The central question is whether explicit citation structure improves
statutory reasoning beyond gains attributable to ordinary code hierarchy. We
design the study around leakage-safe splits, a citation-edge probe, and held-
out statutory QA. The resulting paper target is modest but precise: to show
that explicit document relationships can function as a usable pretraining
signal for legal NLP, even under a 512-token encoder budget.

## Motivation

Legal reasoning depends on explicit document relationships that standard
language model pretraining largely treats as noise. Statutes are not just
sequences of tokens. They are organized systems of definitions, exceptions,
procedural prerequisites, remedies, and cross-references. In housing law, for
example, a question about tenantability may require the model to connect a
landlord's duty to repair, a set of habitability standards, a retaliation
rule, and a procedural provision governing unlawful detainer. These links are
often decisive even when lexical overlap is weak.

This motivates a central question for legal NLP: can a model learn better legal
reasoning if pretraining examples preserve explicit relationships among legal
sections rather than treating documents as exchangeable text blocks? The
working hypothesis of this project is that short-context encoders will learn
more useful legal representations when their pretraining windows are packed
according to citation and code-structure signals. The goal is not only to
improve aggregate downstream accuracy, but also to improve the model's ability
to navigate definitional, exception-driven, and multi-hop statutory reasoning.

The paper is targeted at the legal NLP community rather than at general long-
context modeling. The intended contribution is methodological clarity: a clean
comparison between random order, semantic order, hierarchical order, and
explicit citation order, evaluated on legal tasks where those distinctions
should matter.

## Background & Related Work

Prior work on legal language models has shown that domain-adaptive pretraining
can improve performance over general-purpose models, but most approaches still
inherit the random or weakly structured document order of conventional NLP
pipelines. Legal-BERT and related domain-adapted encoders demonstrated that
legal corpora matter, yet they did not directly encode the relational structure
of statutes and cases into the pretraining curriculum. More recent work on
structure-aware or context-aware pretraining in general NLP suggests that
document order can shape learned representations, especially when neighboring
documents reflect semantically coherent or causally linked information.

This project sits at the intersection of three lines of work. The first is
domain-adaptive legal pretraining, which established that legal text is a
distinct language domain. The second is sequence-ordering and curriculum
learning for language models, including work that uses semantic similarity to
create more coherent local training windows. The third is legal knowledge
representation, where citation graphs, statutory hierarchies, and procedural
links have long been recognized as central to legal analysis. What remains
underexplored is whether explicit document relationships, especially citations,
can be turned into a pretraining signal rather than only a retrieval or post-
hoc explanation tool.

Relative to this literature, the project makes a narrower but cleaner claim. It
does not assume that long-context generative reasoning is required. Instead, it
tests whether packing 512-token windows using explicit legal structure changes
what a lightweight encoder learns. The addition of a hierarchy-only baseline is
particularly important because it distinguishes two hypotheses that are often
conflated: first, that local legal organization is useful; second, that
explicit citation edges carry signal beyond mere hierarchy or adjacency.

In conference terms, the novelty is not a new architecture but a new
experimental control. The hierarchy baseline makes it possible to argue for
citation-aware pretraining specifically, rather than reporting gains that could
be explained by ordinary legal document locality.

## Proposed Method & Experiments

The pilot operates over a filtered California housing-law slice drawn from
`reglab/statecodes`, currently centered on landlord-tenant sections in the
Civil Code. Each statutory section is normalized into a record containing its
code name, chapter, article, section number, title, and text. Citation edges
are extracted with a regex-based resolver over references such as `Section
1942.5` or `§ 1942.5`, and each citation is typed coarsely using local context
cues such as "defined in," "except as provided," and "subject to."

The graph is built over section nodes only. Each pair of sections may carry one
or more edge types:

- `REFERENCES`: resolved cross-citation edges
- `NEXT_SECTION`: adjacency in numeric order within the same code
- `SAME_ARTICLE`: co-membership in the same article
- `SAME_CHAPTER`: co-membership in the same chapter

This multi-edge representation is critical because real legal structure is not
exhausted by a single relation. Two sections may be both adjacent in the code
and linked by an explicit citation, and the experimental design should retain
both facts. The implementation now preserves these multi-relational ties rather
than collapsing them into a single edge label.

The current experimental suite includes four ordering conditions:

- `random`: randomly packed windows
- `embed-sim`: windows packed by local semantic similarity
- `hierarchy-pack`: windows packed by article/chapter/code hierarchy only
- `cite-graph`: windows packed by explicit citations first, with hierarchical
  fallback

All conditions use the same model checkpoint, context length, token budget, and
exposure caps. The model is a lightweight encoder such as DistilBERT or
BERT-base. Training examples are fixed-length windows that contain two to four
sections separated by explicit boundary markers. This makes the intervention
interpretable: the only intended difference across conditions is how legal
structure shapes local co-occurrence during pretraining.

The experimental design is deliberately scoped for a legal NLP paper with a
clear causal story. We do not claim full legal reasoning or broad doctrinal
coverage. Instead, we test whether explicit document relationships measurably
change the representation space learned during domain-adaptive pretraining.

The primary evaluation is a citation-edge probe. Given a source section and a
candidate set, the model must rank the true cited section. This directly tests
whether pretraining captured document relationships rather than only broad
domain familiarity. A secondary evaluation uses held-out statutory QA with
leakage-safe support sections. Closed-book and open-book settings are both
relevant: the former tests what the model internalized; the latter tests
whether the resulting representations help when gold support text is provided.

To support a conference submission, the experiments should report at least the
following: probe accuracy or mean reciprocal rank, QA accuracy or macro-F1,
multiple random seeds once the pilot stabilizes, and qualitative figures that
visualize the citation graph and the effect of each packing regime. The repo
already produces graph statistics and a citation-graph image, which should feed
directly into the paper's method and analysis sections.

The central experimental comparison is no longer just `cite-graph` versus
`random`. The more informative comparison is `cite-graph` versus
`hierarchy-pack`. If hierarchy alone explains the gains, then the claim should
be framed as legal structural locality rather than citation-aware reasoning. If
citation-aware packing still outperforms hierarchy-only packing, that is much
stronger evidence that explicit document relationships contribute distinct
learning signal.

## Next Steps

The next phase of the project should strengthen both scale and validity. First,
the California slice should be widened beyond the current bounded pilot to
increase the number of resolved citation edges and the size of the held-out
probe. A probe built from only a handful of held-out edges is enough for
validation, but not enough for a conference-level result. Second, the data
pipeline should be extended to include directly related Code of Civil Procedure
provisions governing unlawful detainer so the graph spans both substantive and
procedural housing law.

Third, citation extraction should move beyond purely regex-based matching. The
current extractor is serviceable for a pilot, but a publication-quality system
should report manual precision estimates and section-level recall estimates on
annotated samples. Fourth, the hierarchy baseline should be treated as a first-
class condition in every future table and figure, because it is now the
cleanest way to separate explicit citation effects from generic structural
locality.

Finally, the project should mature from a runnable pilot into a full paper
package. That means broader sweeps over corpus size, multiple random seeds,
stable TensorBoard logging, stronger held-out QA construction, and figures that
visualize both the citation graph and per-condition performance. The intended
end state is a legal NLP conference paper that makes a precise claim: short-
context structure-aware pretraining improves statutory reasoning when the
structural signal is defined by explicit legal document relationships and
evaluated against leakage-safe baselines.
