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
pipelines. LEGAL-BERT and related domain-adapted encoders demonstrated that
legal corpora matter, yet they did not directly encode the relational structure
of statutes and cases into the pretraining curriculum (Chalkidis et al., 2020).
LexGLUE then provided a standardized benchmark showing that legal-oriented
models consistently outperform generic ones across diverse legal NLU tasks
(Chalkidis et al., 2022). More recent work on structure-aware or context-aware
pretraining in general NLP suggests that document order can shape learned
representations, especially when neighboring documents reflect semantically
coherent or causally linked information (Shi et al., 2024; Zhao et al., 2024).

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

On the evaluation side, several benchmarks motivate the choice of statutory and
reasoning-heavy tasks. SARA and its follow-on analyses frame statutory
reasoning as a difficult entailment and rule-application problem rather than a
simple reading-comprehension task (Holzenberger et al., 2020; Holzenberger and
Van Durme, 2021). CaseHOLD shows that legal-domain pretraining helps most on
tasks that are genuinely close to the underlying legal corpus (Zheng et al.,
2021). LegalBench broadens this picture by organizing legal reasoning into a
large suite of task types designed with lawyers and legal scholars (Guha et
al., 2023), while LegalBench-RAG highlights that retrieval quality is itself a
separate bottleneck in practical legal reasoning systems (Pipitone and Houir
Alami, 2024).

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

This proposal also differs from prior graph-centric legal work. Existing legal
graph methods typically use citation or statute structure at retrieval time,
classification time, or as an explicit graph-learning signal, rather than as a
minimal intervention on language-model pretraining order. Examples include
heterogeneous-graph statute identification in LeSICiN, citation-network
benchmarking in LeCNet, and knowledge-graph-enhanced statute retrieval in
NyayGraph (Paul et al., 2021; Harde et al., 2025; Shukla et al., 2025). The
closest conceptual analog outside law is SPECTER, which uses citation graphs to
learn document representations in science (Cohan et al., 2020). Our setting is
different: the goal is not document embedding from citation supervision, but
encoder pretraining under controlled local co-occurrence regimes.

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

## References

- Chalkidis, Ilias, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos
  Aletras, and Ion Androutsopoulos. 2020. "LEGAL-BERT: The Muppets straight
  out of Law School." Findings of EMNLP 2020.
  https://aclanthology.org/2020.findings-emnlp.261/
- Chalkidis, Ilias, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion
  Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. "LexGLUE: A
  Benchmark Dataset for Legal Language Understanding in English." ACL 2022.
  https://aclanthology.org/2022.acl-long.297/
- Cohan, Arman, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld.
  2020. "SPECTER: Document-level Representation Learning using
  Citation-informed Transformers." ACL 2020.
  https://aclanthology.org/2020.acl-main.207/
- Guha, Neel, Julian Nyarko, Daniel E. Ho, Christopher Re, Adam Chilton, and
  many others. 2023. "LegalBench: A Collaboratively Built Benchmark for
  Measuring Legal Reasoning in Large Language Models." arXiv.
  https://arxiv.org/abs/2308.11462
- Harde, Pooja, Bhavya Jain, and Sarika Jain. 2025. "LeCNet: A Legal Citation
  Network Benchmark Dataset." JUST-NLP 2025.
  https://aclanthology.org/2025.justnlp-main.4/
- Holzenberger, Nils, Andrew Blair-Stanek, and Benjamin Van Durme. 2020. "A
  Dataset for Statutory Reasoning in Tax Law Entailment and Question
  Answering." arXiv.
  https://arxiv.org/abs/2005.05257
- Holzenberger, Nils, and Benjamin Van Durme. 2021. "Factoring Statutory
  Reasoning as Language Understanding Challenges." ACL-IJCNLP 2021.
  https://aclanthology.org/2021.acl-long.213/
- Paul, Shounak, Pawan Goyal, and Saptarshi Ghosh. 2021. "LeSICiN: A
  Heterogeneous Graph-based Approach for Automatic Legal Statute
  Identification from Indian Legal Documents." arXiv.
  https://arxiv.org/abs/2112.14731
- Pipitone, Nicholas, and Ghita Houir Alami. 2024. "LegalBench-RAG: A
  Benchmark for Retrieval-Augmented Generation in the Legal Domain." arXiv.
  https://arxiv.org/abs/2408.10343
- Rabelo, Juliano, Randy Goebel, Mi-Young Kim, Masaharu Yoshioka, Yoshinobu
  Kano, and Ken Satoh. 2022. "Overview and Discussion of the Competition on
  Legal Information Extraction/Entailment (COLIEE) 2021." Review of
  Socionetwork Strategies.
  https://doi.org/10.1007/s12626-022-00105-z
- Shi, Weijia, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Gergely
  Szilvasy, Rich James, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer,
  Scott Yih, and Mike Lewis. 2024. "In-context Pretraining: Language Modeling
  Beyond Document Boundaries." arXiv.
  https://arxiv.org/abs/2310.10638
- Shukla, Siddharth, Tanuj Tyagi, Abhay Singh Bisht, Ashish Sharma, and Basant
  Agarwal. 2025. "NyayGraph: A Knowledge Graph Enhanced Approach for Legal
  Statute Identification in Indian Law using Large Language Models." NLLP
  2025. https://aclanthology.org/2025.nllp-1.11/
- Zheng, Lucia, Neel Guha, Brandon R. Anderson, Peter Henderson, and Daniel E.
  Ho. 2021. "When Does Pretraining Help? Assessing Self-Supervised Learning
  for Law and the CaseHOLD Dataset." ICAIL 2021.
  https://arxiv.org/abs/2104.08671
- Zhao, Yu, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu,
  Piotr Milos, Yuxiang Wu, and Pasquale Minervini. 2024. "Analysing The
  Impact of Sequence Composition on Language Model Pre-Training." arXiv.
  https://arxiv.org/abs/2402.13991
