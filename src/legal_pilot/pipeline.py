from __future__ import annotations

from pathlib import Path

from legal_pilot.citations import extract_citation_edges
from legal_pilot.embed import build_similarity_index
from legal_pilot.evaluation import compute_probe_metrics, evaluate_qa_classifier, split_qa_examples
from legal_pilot.graphing import build_section_graph, compute_graph_stats, save_citation_graph_visualization
from legal_pilot.io import ensure_dir, load_qa, load_sections, save_json, save_windows
from legal_pilot.packing import generate_windows, summarize_windows
from legal_pilot.splits import build_leakage_safe_split
from legal_pilot.training import train_mlm, train_qa_classifier


def run_pipeline(config: dict) -> None:
    output_dir = ensure_dir(config['output_dir'])
    sections = load_sections(config)
    qa_examples = load_qa(config)
    section_by_id = {section.section_id: section for section in sections}

    citation_edges = extract_citation_edges(sections)
    graph = build_section_graph(sections, citation_edges)
    save_json(output_dir / 'graph_stats.json', compute_graph_stats(graph))
    save_citation_graph_visualization(graph, output_dir / 'citation_graph.png')
    preferred_holdout_ids = sorted(
        {
            section_id
            for example in qa_examples
            if example.split in {'dev', 'test'}
            for section_id in example.support_section_ids
        }
    )
    split = build_leakage_safe_split(
        node_ids=[section.section_id for section in sections],
        citation_edges=citation_edges,
        holdout_fraction=float(config['data']['holdout_fraction']),
        exclusion_hops=int(config['data']['exclusion_hops']),
        seed=int(config['training']['seed']),
        preferred_holdout_ids=preferred_holdout_ids,
    )

    train_sections = [section_by_id[section_id] for section_id in sorted(split['train'])]
    holdout_sections = [section_by_id[section_id] for section_id in sorted(split['holdout'])]
    similarity_index = build_similarity_index(train_sections) if train_sections else {}

    summary = {
        'num_sections': len(sections),
        'num_citation_edges': len(citation_edges),
        'num_train_sections': len(train_sections),
        'num_holdout_sections': len(holdout_sections),
        'num_excluded_sections': len(split['excluded']),
    }
    save_json(output_dir / 'summary.json', summary)

    conditions = list(config['training']['conditions'])
    model_dirs: dict[str, Path] = {}
    train_graph = graph.subgraph(split['train']).copy()

    for condition in conditions:
        condition_dir = ensure_dir(output_dir / condition)
        windows = generate_windows(
            condition=condition,
            sections=train_sections,
            graph=train_graph,
            similarity_index=similarity_index,
            model_name=config['model']['name'],
            max_length=int(config['model']['max_length']),
            min_sections_per_window=int(config['packing']['min_sections_per_window']),
            max_sections_per_window=int(config['packing']['max_sections_per_window']),
            target_exposures_per_section=int(config['packing']['target_exposures_per_section']),
            seed=int(config['training']['seed']),
        )
        save_windows(condition_dir / 'windows.jsonl', windows)
        save_json(condition_dir / 'window_stats.json', summarize_windows(windows))
        if config['training']['enabled']:
            artifacts = train_mlm(
                model_name=config['model']['name'],
                output_dir=condition_dir / 'mlm_model',
                windows=windows,
                config=config,
            )
            model_dirs[condition] = artifacts.model_dir
        else:
            model_dirs[condition] = condition_dir / 'mlm_model'

    probe_edges = [
        edge
        for edge in citation_edges
        if edge.source_id in split['holdout'] and edge.target_id in split['holdout']
    ]
    probe_results = {}
    for condition, model_dir in model_dirs.items():
        probe_results[condition] = compute_probe_metrics(
            model_dir=model_dir,
            sections=holdout_sections,
            probe_edges=probe_edges,
            candidate_pool_size=int(config['probe']['candidate_pool_size']),
            recall_ks=[int(k) for k in config['probe']['recall_ks']],
            seed=int(config['training']['seed']),
        )
    save_json(output_dir / 'probe_results.json', probe_results)

    if config['qa']['enabled'] and qa_examples:
        filtered_examples = []
        for example in qa_examples:
            support_ids = set(example.support_section_ids)
            if example.split == 'train' and support_ids and support_ids.issubset(split['train']):
                filtered_examples.append(example)
            if example.split in {'dev', 'test'} and support_ids and support_ids.issubset(split['holdout']):
                filtered_examples.append(example)
        qa_buckets = split_qa_examples(filtered_examples)
        train_examples = qa_buckets.get('train', [])
        dev_examples = qa_buckets.get('dev', qa_buckets.get('test', []))
        test_examples = qa_buckets.get('test', [])
        for condition, model_dir in model_dirs.items():
            condition_dir = ensure_dir(output_dir / condition)
            if train_examples and test_examples:
                closed_book_dir = condition_dir / 'qa_closed_book'
                closed_dev = dev_examples or test_examples
                closed_metrics = train_qa_classifier(
                    pretrained_dir=model_dir,
                    output_dir=closed_book_dir,
                    train_examples=train_examples,
                    dev_examples=closed_dev,
                    section_by_id=section_by_id,
                    config=config,
                    with_support=False,
                )
                save_json(closed_book_dir / 'dev_metrics.json', closed_metrics)
                closed_test_metrics = evaluate_qa_classifier(
                    classifier_dir=closed_book_dir,
                    test_examples=test_examples,
                    section_by_id=section_by_id,
                    max_length=int(config['model']['max_length']),
                    with_support=False,
                )
                save_json(closed_book_dir / 'test_metrics.json', closed_test_metrics)

                open_book_dir = condition_dir / 'qa_open_book'
                open_metrics = train_qa_classifier(
                    pretrained_dir=model_dir,
                    output_dir=open_book_dir,
                    train_examples=train_examples,
                    dev_examples=closed_dev,
                    section_by_id=section_by_id,
                    config=config,
                    with_support=True,
                )
                save_json(open_book_dir / 'dev_metrics.json', open_metrics)
                open_test_metrics = evaluate_qa_classifier(
                    classifier_dir=open_book_dir,
                    test_examples=test_examples,
                    section_by_id=section_by_id,
                    max_length=int(config['model']['max_length']),
                    with_support=True,
                )
                save_json(open_book_dir / 'test_metrics.json', open_test_metrics)
