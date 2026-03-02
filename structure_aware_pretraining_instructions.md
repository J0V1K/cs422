# Agent Instructions: Structure-Aware Pretraining for Legal Language Models

## Project Identity

**Title:** Structure-Aware Pretraining for Legal Language Models: Exploiting Citation Graphs for Better Legal Reasoning  
**Course:** CS 422 — Stanford University, Winter 2026  
**Researchers:** Javokhir Arifov and Joshua Rizo

---

## 1. Project Objective

Determine whether packing short-context pretraining examples along explicit statutory citation edges improves legal language model performance on definitional and multi-hop statutory reasoning, compared to random packing and embedding-similarity packing.

### Core Thesis

Legal text creates a "lexical similarity trap": sections that look similar on the surface may play different legal roles, while sections with little lexical overlap may be tightly connected by explicit cross-references. For a 512-token encoder, this should be tested by packing 2-4 related sections into each training example, not by creating long walks that exceed model context. The claim of this project is therefore narrower and cleaner: citation-aware local packing should improve learning of section-to-section legal relationships beyond random packing and raw embedding similarity.

---

## 2. Experimental Conditions

Train three primary lightweight language models that differ only in how statutory sections are packed into 512-token pretraining examples. All other hyperparameters must be held constant across conditions, including the exact multiset of sections shown during training.

| Condition | Label | Ordering Strategy |
|-----------|-------|-------------------|
| C1 | `random` | Randomly pack sections into short windows (null baseline) |
| C2 | `embed-sim` | Pack sections with nearest neighbors in raw-text embedding space |
| C3 | `cite-graph` | Pack sections using explicit citation edges plus local code adjacency |

**Optional follow-on ablation:** `expand-embed` may be added only after C1-C3 are stable. It is not part of the core overnight or semester-scale minimum viable experiment because it introduces extra supervision from LLM-generated expansions.

---

## 3. Data Sources and Acquisition

### 3.1 Primary Statute Corpus

**Primary source:** California Codes XML  
**URL:** `https://leginfo.legislature.ca.gov` (publicly available XML dumps)  
**Scope for MVP:** California housing / landlord-tenant slice only:
- Civil Code sections in the landlord-tenant range (starting with `1940` and related nearby provisions)
- Directly cited neighbors reachable within 1 citation hop
- Optional addition: unlawful detainer provisions from Code of Civil Procedure if they are directly connected

**Why this is the default:** clean XML, explicit citations, manageable scale, and a realistic overnight compute footprint.

**Deferred scale-up source:** HousingQA statute corpus  
**Scale:** ~1.7 million statutes across US states  
**Access:** Available through RegLab collaboration (coordinate with Lucia Zheng)  
**Use only after MVP works:** multi-state scaling should be a second-stage experiment, not the starting point.

**Agent action:**
1. Attempt to acquire the HousingQA corpus first. If unavailable or access-delayed, proceed with California Codes XML.
2. For California Codes: download the bulk XML export, parse each code into individual sections, and store as structured records with fields: `section_id`, `code_name`, `division`, `part`, `chapter`, `article`, `section_number`, `section_title`, `section_text`, `effective_date`.
3. Restrict the MVP corpus to a coherent legal area before training. Log corpus statistics: total sections, average section length, citation counts, and connected-component sizes.
4. Create leakage-safe splits at the section-cluster level. If a section is in dev/test, its 1-hop citation neighbors must also be excluded from pretraining.

### 3.2 Evaluation Datasets

#### CitationEdgeProbe (Primary structural probe)
- **Source:** Held-out citation edges from the California statute graph
- **Task:** Given a source section and a candidate set, rank the true cited section(s)
- **Why:** Directly tests whether pretraining encoded document relationships, without downstream QA confounds

#### Held-Out Statutory QA (Primary reasoning evaluation)
- **Source:** Small evaluation set built only from dev/test sections excluded from pretraining
- **Task:** Answer yes/no or multiple-choice questions about held-out statutory sections
- **Why:** Measures legal reasoning while avoiding corpus leakage
- **Construction:** Prefer an existing dataset if it can be filtered to held-out sections; otherwise create a small manually verified set for the chosen California subdomain

#### HousingQA / BarExamQA / LegalBench (Optional transfer evaluations)
- Use only after the primary structural probe and held-out QA are working
- HousingQA may be used only with a strict anti-leakage filter: no gold support section, paraphrase, or 1-hop citation neighbor may appear in pretraining
- BarExamQA and LegalBench should be framed as transfer checks, not primary evidence for the citation-ordering claim

**Agent action:**
1. Build the CitationEdgeProbe by holding out a subset of resolved citation edges from dev/test clusters.
2. Build or filter a held-out statutory QA set so that all support sections are absent from pretraining.
3. If using HousingQA, BarExamQA, or LegalBench, treat them as secondary transfer evaluations and document the anti-leakage filter explicitly.
4. Record counts, label distributions, and section coverage for every evaluation set.

### 3.3 Embeddings for Condition C2 and C4

**Embedding model:** Use a lightweight sentence encoder for the MVP. Recommended: `sentence-transformers/all-MiniLM-L6-v2`.

**For Condition C2 (embed-sim):** Embed the raw text of each statutory section. Construct a nearest-neighbor graph over all sections. Order pretraining sequences by traversing this graph (nearest-neighbor chains).

**Agent action:**
1. Compute raw-text embeddings for all sections. Store as numpy arrays or FAISS index.
2. Build approximate nearest-neighbor index (FAISS with IVF or HNSW) for efficient graph construction.
3. Log embedding statistics: dimensionality, mean/std cosine similarity distribution, and a sample of nearest-neighbor pairs for manual inspection.

---

## 4. Citation Graph Construction (Condition C3)

This is the novel contribution. Execute the following pipeline carefully.

### 4.1 Cross-Reference Extraction

Parse statutory text to extract all internal cross-references. Legal citations follow semi-regular patterns but have significant variation.

**California pattern examples:**
- `Section 1942.5` / `§ 1942.5` / `Sec. 1942.5`
- `Civil Code Section 1942.5`
- `subdivision (a) of Section 1942.5`
- `Sections 1941 to 1942.5, inclusive`
- `Chapter 2 (commencing with Section 1940) of Title 5 of Part 4 of Division 3`

**Multi-state pattern examples:**
- `ORS 90.360` (Oregon)
- `N.Y. Real Prop. Law § 235-b` (New York)
- `Tex. Prop. Code Ann. § 92.052` (Texas)

**Agent action:**
1. Build a regex-based citation extractor. Start with the most common patterns and iteratively expand coverage.
2. Run the extractor over the entire corpus. For each extracted citation, resolve it to a specific section in the corpus (this requires a lookup table mapping citation strings to section IDs).
3. Classify each cross-reference into one of these types:
   - **Definitional:** "As defined in Section X" — points to definitions
   - **Exception:** "Except as provided in Section X" — carves out exceptions
   - **Incorporation:** "Subject to Section X" — incorporates requirements
   - **Penalty:** "Shall be punishable under Section X" — points to enforcement
   - **Procedural:** "In accordance with the procedures of Section X" — process references
4. Log extraction statistics: total citations found, resolution rate (% successfully mapped to known sections), distribution across reference types, and a sample of 50 unresolved citations for debugging.
5. **Validation:** Estimate precision by manually inspecting 100 randomly sampled extracted citations. Estimate recall by exhaustively annotating citations in 25-50 sampled source sections and comparing extracted vs. gold citations. Target: >90% precision, >80% recall on the annotated sections.

### 4.2 Graph Construction

**Nodes:** Each statutory section is a node. Metadata: `section_id`, `code_name`, `chapter`, `article`, `section_number`, `text_length`, `num_outgoing_refs`, `num_incoming_refs`.

**Edge types:**
- `REFERENCES` — extracted cross-citations (directed: source → target)
- `NEXT_SECTION` — adjacent section numbers within the same code (directed or undirected)
- `SAME_CHAPTER` — sections within the same chapter (undirected)

Store hierarchy as metadata, not as separate non-text nodes. The training corpus consists only of section text, so every traversed node must correspond to a textual section.

**Implementation:** Use NetworkX for prototyping. Neo4j is unnecessary for the MVP.

**Agent action:**
1. Construct the graph. Log: total nodes, total edges by type, degree distribution (in-degree and out-degree for REFERENCES edges), and connected component count and sizes.
2. Identify hub nodes (high in-degree) — these are likely foundational definitional or procedural sections that many other sections reference.
3. Save the graph in a serializable format (e.g., GraphML, edge list + node attributes CSV, or pickled NetworkX object).

### 4.3 Training Sequence Generation

For each condition, generate fixed-length 512-token training examples by packing 2-4 statutory sections into a single window. Separate sections with `[SEP]` and include a short title/header marker so the model can detect boundaries. No training example may exceed the model context length.

**Condition C1 — Random:**
- Shuffle sections randomly and pack them into windows.

**Condition C2 — Embedding Similarity:**
- Pick a seed section, then pack its nearest neighbors by embedding cosine similarity until the 512-token budget is filled.

**Condition C3 — Citation Graph:**
- Pick a seed section. Pack cited sections first using `REFERENCES` edges; if there is remaining room, back off to `NEXT_SECTION`, then `SAME_CHAPTER`.
- Track global exposure counts and cap repeated appearances of hub nodes so C3 does not win merely by oversampling a few central sections.

**Agent action:**
1. Generate all sequence sets from the same section multiset. Log: total windows per condition, average sections per window, token utilization, coverage, and section exposure histograms.
2. Save tokenized datasets ready for training.
3. **Sanity check:** For a sample of 10 windows from each condition, print section IDs and verify that C3 windows contain actual citation-linked neighbors.

---

## 5. Model Training

### 5.1 Architecture and Scale

**Main study base model:** `nlpaueb/legal-bert-base-uncased` or `bert-base-uncased`  
**Rationale:** encoder-only MLM is appropriate for section packing and classification-style downstream evaluation.

**Overnight pilot model:** DistilBERT or another ~66M-parameter encoder initialized from a single shared checkpoint.

**Do NOT use for the MVP:** long-context claims or decoder-only models. The experiment is about short-context relational packing.

### 5.2 Hyperparameters (Hold Constant Across All 3 Primary Conditions)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max sequence length | 512 tokens | Standard for BERT |
| Batch size | 32 | Effective batch size; use gradient accumulation if needed |
| Learning rate | 2e-5 to 5e-5 | Tune once on the baseline, then hold fixed |
| LR schedule | Linear warmup (10% of steps) + linear decay | |
| Weight decay | 0.01 | |
| Optimizer | AdamW | |
| Total training tokens | Equal across conditions | Critical: each condition must see the same number of tokens |
| Pretraining objective | Masked Language Modeling (MLM) | 15% masking rate, standard BERT setup |
| Random seed | 42 | Minimum: 3 seeds for the final comparison; overnight pilot may use 1 seed |

### 5.3 Training Infrastructure

**MVP target:** 1× modern GPU  
**Semester-scale estimate:** 3 runs × 6-12 hours each on a narrowed corpus  
**Overnight pilot target:** 2-4 hours per run on ~20M to 30M tokens using DistilBERT on the California housing slice

**Agent action:**
1. Initialize all models from the same pretrained checkpoint.
2. Train each model on its respective packed windows using the same hyperparameters.
3. Save checkpoints at regular intervals (every 20% of training is enough for the MVP).
4. Log training loss curves for all conditions on the same plot.

---

## 6. Evaluation Protocol

### 6.1 Primary Evaluation Tasks

Run two required evaluations and one optional extension.

**Task A — CitationEdgeProbe:**
- Input: a source section plus a candidate pool of possible target sections
- Output: rank the true cited section(s)
- **Tests:** whether pretraining encoded document relationships directly

**Task B — Held-Out Statutory QA (closed-book and open-book):**
- Closed-book: fine-tune each pretrained encoder on the QA training split; no external documents at inference
- Open-book: provide the gold support section(s), packed into the same 512-token format across all models
- **Tests:** whether structural packing helps downstream legal reasoning when support sections were absent from pretraining

**Task C — End-to-end retrieval/RAG (optional extension):**
- Only include this if the retriever is trained identically from each checkpoint
- Do not use BM25 or a frozen external encoder as primary evidence for the pretraining claim

### 6.2 Metrics

| Metric | Applicable To | Description |
|--------|---------------|-------------|
| Accuracy | Held-out QA | Correct answer rate |
| F1 (macro) | Held-out QA | Balanced performance across classes |
| MRR (Mean Reciprocal Rank) | CitationEdgeProbe | Rank of first true cited section |
| Recall@k (k=1, 5, 10) | CitationEdgeProbe | Fraction of true cited sections in top-k |
| Confidence calibration | QA tasks | ECE (Expected Calibration Error) |

### 6.3 Stratified Analysis by Reasoning Type

This is the most important analytical contribution. For the MVP, use only reasoning types that can be labeled reliably:

| Reasoning Type | Description | Hypothesis |
|----------------|-------------|------------|
| Direct Lookup | Answer is stated explicitly in a single section | Minimal gain from citation ordering |
| Definitional | Requires following a direct "defined in Section X" reference | Moderate gain |
| Multi-hop | Requires chaining 2+ cross-references to reach the answer | Large gain (primary hypothesis) |
| Analogical (exploratory only) | Requires mapping a novel fact pattern to a legal concept with weak lexical overlap | Do not treat as a primary claim in the MVP |

**Agent action:**
1. Write a fixed annotation rubric before scoring models.
2. Double-annotate at least 100 QA items and report inter-annotator agreement (target Cohen's kappa >= 0.7).
3. Use heuristics only for direct-lookup vs multi-hop pre-labeling; human review is required for the final stratified table.
4. Report performance separately for each reasoning type x condition x task.
5. Treat analogical results as exploratory unless a reliable annotation protocol is in place.

---

## 7. Analysis and Ablations

### 7.1 Primary Analysis

1. **Main effects table:** Condition (C1-C3) x Task (`CitationEdgeProbe`, closed-book QA, open-book QA). Report MRR/Recall@k or accuracy/F1 with 95% confidence intervals.
2. **Stratified results:** Same table broken down by reasoning type for the QA tasks.
3. **Statistical significance:** paired bootstrap over examples plus variance across seeds for the final comparison.

### 7.2 Ablation Studies (if compute allows)

| Ablation | Question Addressed |
|----------|--------------------|
| Citation type ablation | Train C3 variants using only definitional vs all references. Which relationship type contributes most? |
| Packing size ablation | Pack 2 vs 3 vs 4 sections per window. How much local context is useful? |
| Hybrid ordering | Combine citation graph with embedding similarity (weighted). Does the combination outperform either alone? |
| Corpus scale | Train on 25%, 50%, 100% of the narrowed corpus with citation ordering. Does the benefit scale with data? |

### 7.3 Error Analysis

1. For each condition, sample 25-50 incorrectly answered questions from the definitional and multi-hop categories.
2. Categorize errors: retrieval failure, reasoning failure, knowledge gap, lexical confusion, hallucination.
3. Compare error type distributions across conditions. The hypothesis is that C3 should show fewer "reasoning failure" and "lexical confusion" errors.

### 7.4 Graph Analysis

1. Report the correlation between a question's "citation distance" (minimum hops in the citation graph between the question's relevant sections) and accuracy.
2. Visualize subgraphs that correspond to commonly tested legal concepts (e.g., landlord-tenant rights, eviction procedures).
3. Identify whether hub nodes in the citation graph correspond to sections that appear frequently in evaluation questions.

---

## 8. Deliverables

### 8.1 Code Repository

Organize as follows:

```
structure-aware-legal-pretraining/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Downloaded corpora (gitignored)
│   ├── processed/              # Cleaned, parsed sections
│   ├── graphs/                 # Citation graph files
│   ├── sequences/              # Generated training sequences per condition
│   └── eval/                   # Evaluation datasets
├── src/
│   ├── data_acquisition/       # Download and parse statutes
│   ├── citation_extraction/    # Regex-based cross-reference extraction
│   ├── graph_construction/     # Build and analyze citation graph
│   ├── sequence_generation/    # Generate ordered training sequences
│   ├── training/               # BERT continued pretraining scripts
│   ├── evaluation/             # Evaluation pipeline (closed, open, RAG)
│   └── analysis/               # Stratified analysis, visualization, statistics
├── configs/
│   ├── training_config.yaml    # Shared hyperparameters
│   └── eval_config.yaml        # Evaluation settings
├── notebooks/
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_graph_analysis.ipynb
│   ├── 03_sequence_inspection.ipynb
│   └── 04_results_analysis.ipynb
├── results/
│   ├── training_logs/
│   ├── eval_results/
│   └── figures/
└── paper/
    └── main.tex
```

### 8.2 Written Report

The final report should contain:

1. **Introduction:** The lexical gap problem, why standard pretraining misses legal structure, contribution statement.
2. **Related Work:** In-Context Pretraining (Shi et al.), SBP (Yang et al.), Legal-BERT (Zheng et al.), Quiet-STaR, SPICE — and how this project extends or synthesizes them.
3. **Methodology:** Data pipeline (Sections 3–4 of this document), model training (Section 5), evaluation protocol (Section 6).
4. **Results:** Main effects, stratified analysis, ablations (Section 7).
5. **Discussion:** Does citation ordering help? Where do gains concentrate? Does it transfer? Compute/performance tradeoff. Limitations.
6. **Conclusion:** Summary of findings, implications for legal AI, future work.

### 8.3 Key Figures to Produce

1. **Citation graph visualization** — subgraph of a high-traffic legal area (e.g., California landlord-tenant code) showing node types and edge types.
2. **Training loss curves** — all trained conditions on a single plot.
3. **Main results bar chart** — probe/QA performance by condition grouped by task.
4. **Stratified results heatmap** — reasoning type (rows) × condition (columns), cell color = accuracy.
5. **Performance vs. citation distance scatter** — x-axis: minimum citation hops for the question, y-axis: accuracy, separate line per condition.
6. **Error type distribution** — stacked bar chart by condition showing error category proportions.

---

## 9. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Corpus leakage into evaluation | High | Split by section clusters before pretraining; exclude 1-hop neighbors of dev/test sections from training |
| Citation extraction precision too low (<80%) | Medium | Use California XML only; validate on annotated sections before any model training |
| Compute insufficient for 3 full runs | Medium | Use DistilBERT, 20M-30M tokens, and the narrowed California housing slice |
| No significant difference between conditions | Medium | Check the direct CitationEdgeProbe first; if that fails, do not over-interpret QA noise |
| Embedding similarity ordering ~= citation ordering | Low | Measure overlap between C2 and C3 windows; if overlap is high, report that explicitly |
| Analogical labels are unreliable | Medium | Keep analogical reasoning exploratory unless manual agreement is adequate |

---

## 10. Execution Order and Checkpoints

Follow this sequence. Do not proceed past a checkpoint until the checkpoint criteria are met.

### Phase 1: Data (Week 1)

1. Download California Codes XML
2. Parse the housing / landlord-tenant slice into structured section records
3. Build leakage-safe train/dev/test splits at the section-cluster level
4. **Checkpoint 1:** Corpus statistics logged; all split boundaries verified; no dev/test section or 1-hop neighbor appears in pretraining.

### Phase 2: Graph and Embeddings (Week 2)

5. Build citation extractor; validate on 100 samples
6. Construct citation graph; log statistics
7. Compute raw-text embeddings for all sections
8. Build nearest-neighbor indices
9. **Checkpoint 2:** Citation extraction precision >90%; recall estimated on annotated sections; embeddings computed and indexed.

### Phase 3: Window Generation (Week 2-3)

10. Generate packed 512-token training windows for C1-C3
11. Sanity-check 10 sequences per condition
12. Tokenize and save as training-ready datasets
13. **Checkpoint 3:** All 3 window sets generated; total token counts and exposure histograms are matched across conditions.

### Phase 4: Training (Week 3-4)

14. Train C1 (random) — this is your baseline; verify training loop works
15. Train C2 and C3
16. Save checkpoints and training logs
17. **Checkpoint 4:** All 3 models trained to completion; training loss curves are stable.

### Phase 5: Evaluation (Week 4-5)

18. Run CitationEdgeProbe
19. Run closed-book and open-book held-out QA
20. Compute metrics; run stratified analysis
21. Statistical significance tests
22. Error analysis on 25-50 samples per condition
23. **Checkpoint 5:** Structural probe and QA results are both populated; cite-graph either beats or fails to beat the random baseline on the direct probe.

### Phase 6: Writing and Analysis (Week 5)

24. Produce all figures
25. Write report
26. Review and revise

---

## 11. Key Research Questions (Reiterated for Clarity)

At the conclusion of this project, the results should directly address:

1. **Does citation-aware packing help at all?** Compare C3 vs. C1 on CitationEdgeProbe and held-out QA.
2. **Does citation-aware packing outperform embedding similarity?** Compare C3 vs. C2. This is the core novel claim.
3. **Where do gains concentrate?** Stratified analysis — hypothesis: definitional and multi-hop tasks show the largest delta.
4. **How much of the effect is direct structural knowledge vs generic semantic similarity?** Use the CitationEdgeProbe as the primary discriminating test.
5. **What is the compute/performance tradeoff?** Graph construction and packing overhead vs. improvement magnitude.
6. **Does structural pretraining reduce overconfidence?** Calibration analysis on the held-out QA task.

---

## 12. Overnight Pilot Experiment

This is the fastest credible version of the study. It is a pilot, not the final paper result.

### 12.1 Goal

Establish whether citation-aware packing produces a measurable gain over random packing after one night of compute on a narrow California housing-law corpus.

### 12.2 Scope

- **Corpus:** 1 California legal slice only, ideally landlord-tenant sections plus 1-hop citation neighbors
- **Conditions:** `random` (C1) vs `cite-graph` (C3)
- **Optional third arm:** `embed-sim` (C2) only if preprocessing is already built
- **Model:** DistilBERT or similar small encoder
- **Training budget:** 20M-30M total tokens per condition, 512-token windows
- **Seeds:** 1 seed for the overnight pilot

### 12.3 Required Preprocessing

1. Parse the California housing slice into sections.
2. Extract and resolve citations.
3. Hold out 15%-20% of section clusters for evaluation; remove their 1-hop neighbors from pretraining.
4. Build packed windows for C1 and C3 from the same section multiset.

### 12.4 Overnight Evaluation

Run the following in order:

1. **CitationEdgeProbe**
   - Hold out a set of true citation edges from the dev/test clusters.
   - For each source section, rank the true target among 50-100 candidates from the same code.
   - Primary metrics: MRR and Recall@10.
2. **Small held-out QA**
   - Build or filter 100-300 questions whose support sections are entirely in the held-out split.
   - Evaluate closed-book first.
   - Add open-book only if the packing pipeline is stable.

### 12.5 Success Criteria

- C3 beats C1 on CitationEdgeProbe by a clear margin (for example, +3 to +5 absolute Recall@10 or a consistent MRR gain).
- C3 is at least non-inferior to C1 on held-out QA.
- Training curves remain stable and token/exposure controls match across conditions.

### 12.6 What Not to Do Overnight

- Do not include BarExamQA or LegalBench.
- Do not include query-expansion ordering.
- Do not claim analogical reasoning improvements from the pilot.
- Do not use BM25 or a frozen external retriever as evidence that the pretrained encoder learned citation structure.
