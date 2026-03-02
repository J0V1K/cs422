# Structure-Aware Legal Pilot

Minimal runnable scaffold for the overnight pilot in
[structure_aware_pretraining_instructions.md](/Users/jovik/Desktop/cs422_project/structure_aware_pretraining_instructions.md).

## What this does

The pipeline supports three conditions:

- `random`
- `embed-sim`
- `cite-graph`

For each enabled condition it can:

1. Load sections from a built-in sample corpus, from Hugging Face `reglab/statecodes`, or from user-provided JSONL.
2. Extract and resolve statutory citations.
3. Build a section graph and leakage-safe split.
4. Generate packed 512-token MLM windows.
5. Continue pretraining a masked-language model.
6. Evaluate with a citation-edge ranking probe.
7. Optionally fine-tune and score a QA classifier if QA examples are provided.

## Quick start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the built-in sample end to end:

```bash
python3 scripts/run_overnight_pilot.py --config configs/overnight_pilot.yaml
```

Artifacts are written to `runs/overnight_pilot/`.

## State codes mode

Set `data.mode: hf_statecodes` in [configs/overnight_pilot.yaml](/Users/jovik/Desktop/cs422_project/configs/overnight_pilot.yaml).

The current default `hf` block is aimed at the California Civil Code landlord-tenant slice from `reglab/statecodes`. The loader pulls rows from:

- dataset: `reglab/statecodes`
- config: `all_codes`
- split: `train`

It then filters by:

- `state`
- `code_filters`
- regex `include_patterns`
- regex `exclude_patterns`

This is the fastest path to a real pilot run because it avoids writing a separate California XML parser first.

If you want to add Code of Civil Procedure unlawful-detainer sections later, widen the `code_filters` and `include_patterns` in the config rather than changing the loader.

## Real data mode

Set `data.mode: jsonl` in [configs/overnight_pilot.yaml](/Users/jovik/Desktop/cs422_project/configs/overnight_pilot.yaml) and provide:

- `data.sections_path`
- `data.qa_path` if you want QA fine-tuning

### Sections JSONL schema

Each line should contain:

```json
{
  "section_id": "civ_1940",
  "code_name": "Civil Code",
  "chapter": "Landlord and Tenant",
  "article": "General Provisions",
  "section_number": "1940",
  "section_title": "Hiring of Real Property",
  "section_text": "This hiring is regulated by ...",
  "effective_date": "2024-01-01"
}
```

### QA JSONL schema

Each line should contain:

```json
{
  "id": "qa_001",
  "split": "train",
  "question": "Can a landlord retaliate after a tenant complains about habitability issues?",
  "choices": ["Yes", "No"],
  "answer_index": 1,
  "support_section_ids": ["civ_1942_5"]
}
```

`split` must be one of `train`, `dev`, or `test`.

## Notes

- The Hugging Face ingest for `reglab/statecodes` is implemented. California XML ingest is still not implemented.
- The citation probe is the primary overnight metric. QA is optional and requires labeled examples.
- The default embedding backend for `embed-sim` uses TF-IDF to avoid a second large model download during pilot runs.
