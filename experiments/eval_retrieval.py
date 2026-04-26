#!/usr/bin/env python3
"""
Evaluate retrieval quality and cost for baseline vs Grover mode.

Query file format (JSON list):
[
  {
    "id": "q1",
    "query": "What is normalization?",
    "relevant_chunk_ids": [12, 45]
  }
]
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import itertools
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List

_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    GroverRetriever,
    IndexKeywordRetriever,
    filter_retrieved_chunks,
    load_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TokenSmith retrieval modes.")
    parser.add_argument(
        "--queries",
        default="experiments/queries.json",
        help="JSON file with query objects (id, query, relevant_chunk_ids).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "grover", "both"],
        default="both",
        help="Which retrieval mode(s) to evaluate.",
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="Index prefix used by artifacts.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to TokenSmith YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/results",
        help="Directory to write evaluation JSON output.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per configuration.",
    )
    parser.add_argument(
        "--sweep_grover_max_pool",
        default="",
        help="Comma-separated values for grover_max_pool sweep (e.g. 16,32,64).",
    )
    parser.add_argument(
        "--sweep_grover_shots",
        default="",
        help="Comma-separated values for grover_shots sweep (e.g. 32,64,128).",
    )
    parser.add_argument(
        "--sweep_grover_iterations",
        default="",
        help="Comma-separated values for grover_iterations sweep (e.g. 1,2,3).",
    )
    return parser.parse_args()


def _default_weight_template() -> Dict[str, float]:
    return {"faiss": 0.0, "bm25": 0.0, "index_keywords": 0.0, "grover": 0.0}


def _repo_path(path_text: str) -> pathlib.Path:
    p = pathlib.Path(path_text)
    if p.is_absolute():
        return p
    return (_project_root / p).resolve()


def _parse_int_csv(csv_text: str) -> List[int]:
    if not csv_text.strip():
        return []
    values: List[int] = []
    for token in csv_text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _build_sweep_points(args: argparse.Namespace, cfg_base: RAGConfig) -> List[Dict[str, int]]:
    pools = _parse_int_csv(args.sweep_grover_max_pool) or [int(cfg_base.grover_max_pool)]
    shots = _parse_int_csv(args.sweep_grover_shots) or [int(cfg_base.grover_shots)]
    iterations = _parse_int_csv(args.sweep_grover_iterations)
    if not iterations:
        iterations = [
            int(cfg_base.grover_iterations)
            if cfg_base.grover_iterations is not None
            else 1
        ]
    points: List[Dict[str, int]] = []
    for pool, shot, iters in itertools.product(pools, shots, iterations):
        points.append(
            {
                "grover_max_pool": int(pool),
                "grover_shots": int(shot),
                "grover_iterations": int(iters),
            }
        )
    return points


def _build_retrievers(cfg: RAGConfig, faiss_idx: Any, bm25_idx: Any) -> List[Any]:
    retrievers: List[Any] = []
    w = cfg.ranker_weights
    if w.get("faiss", 0) > 0:
        retrievers.append(FAISSRetriever(faiss_idx, cfg.embed_model))
    if w.get("bm25", 0) > 0:
        retrievers.append(BM25Retriever(bm25_idx))
    if w.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )
    if w.get("grover", 0) > 0:
        retrievers.append(GroverRetriever(cfg, faiss_index=faiss_idx, bm25_index=bm25_idx))
    return retrievers


def _precision_at_k(retrieved_ids: List[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for idx in topk if idx in relevant_ids)
    return float(hits) / float(k)


def _recall_at_k(retrieved_ids: List[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for idx in topk if idx in relevant_ids)
    return float(hits) / float(len(relevant_ids))


def _reciprocal_rank(retrieved_ids: List[int], relevant_ids: set[int]) -> float:
    for rank, idx in enumerate(retrieved_ids, start=1):
        if idx in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def _estimate_chunks_accessed(cfg: RAGConfig, num_chunks: int, pool_n: int) -> int:
    w = cfg.ranker_weights
    accessed = 0
    if w.get("faiss", 0) > 0:
        accessed = max(accessed, num_chunks)  # IndexFlatL2 is exhaustive.
    if w.get("bm25", 0) > 0:
        accessed = max(accessed, num_chunks)  # BM25 scores all docs in corpus.
    if w.get("grover", 0) > 0:
        accessed = max(accessed, min(num_chunks, pool_n, int(cfg.grover_max_pool)))
    if w.get("index_keywords", 0) > 0 and accessed == 0:
        accessed = min(num_chunks, pool_n)
    return accessed


def _evaluate_mode(
    mode_name: str,
    cfg: RAGConfig,
    queries: List[Dict[str, Any]],
    faiss_idx: Any,
    bm25_idx: Any,
    chunks: List[str],
) -> Dict[str, Any]:
    retrievers = _build_retrievers(cfg, faiss_idx, bm25_idx)
    if not retrievers:
        raise ValueError(f"No retrievers enabled for mode '{mode_name}'.")

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    per_query: List[Dict[str, Any]] = []
    p_sum = 0.0
    r_sum = 0.0
    rr_sum = 0.0
    t_sum_ms = 0.0
    c_sum = 0
    embed_sum_ms = 0.0
    grover_sim_sum_ms = 0.0
    post_sum_ms = 0.0

    pool_n = max(cfg.num_candidates, cfg.top_k + 10)

    for q in queries:
        qid = str(q["id"])
        text = str(q["query"])
        relevant = set(int(i) for i in q.get("relevant_chunk_ids", []))

        t0 = time.perf_counter()
        raw_scores: Dict[str, Dict[int, float]] = {}
        grover_timing = {
            "embedding_ms": 0.0,
            "grover_sim_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(text, pool_n, chunks)
            if retriever.name == "grover":
                grover_timing = dict(getattr(retriever, "last_timing_ms", grover_timing))
        ordered_ids, _ = ranker.rank(raw_scores=raw_scores)
        top_ids = filter_retrieved_chunks(cfg, chunks, ordered_ids)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        p = _precision_at_k(top_ids, relevant, cfg.top_k)
        r = _recall_at_k(top_ids, relevant, cfg.top_k)
        rr = _reciprocal_rank(top_ids, relevant)
        chunks_accessed = _estimate_chunks_accessed(cfg, len(chunks), pool_n)

        p_sum += p
        r_sum += r
        rr_sum += rr
        t_sum_ms += elapsed_ms
        c_sum += chunks_accessed
        embed_sum_ms += float(grover_timing["embedding_ms"])
        grover_sim_sum_ms += float(grover_timing["grover_sim_ms"])
        post_sum_ms += float(grover_timing["postprocess_ms"])

        per_query.append(
            {
                "id": qid,
                "query": text,
                "relevant_chunk_ids": sorted(relevant),
                "retrieved_topk": [int(i) for i in top_ids],
                "precision_at_k": p,
                "recall_at_k": r,
                "reciprocal_rank": rr,
                "retrieval_time_ms": elapsed_ms,
                "chunks_accessed_estimate": chunks_accessed,
                "grover_timing_ms": grover_timing,
            }
        )

    n = max(1, len(queries))
    summary = {
        "mode": mode_name,
        "num_queries": len(queries),
        "top_k": cfg.top_k,
        "avg_precision_at_k": p_sum / n,
        "avg_recall_at_k": r_sum / n,
        "mrr": rr_sum / n,
        "avg_retrieval_time_ms": t_sum_ms / n,
        "avg_chunks_accessed_estimate": float(c_sum) / float(n),
        "avg_grover_embedding_ms": embed_sum_ms / n,
        "avg_grover_simulation_ms": grover_sim_sum_ms / n,
        "avg_grover_postprocess_ms": post_sum_ms / n,
        "weights": cfg.ranker_weights,
    }

    return {"summary": summary, "queries": per_query}


def _print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Retrieval Evaluation Summary ===")
    print(
        "mode       | p@k      | r@k      | mrr      | avg_ms   | avg_chunks"
    )
    print("-" * 69)
    for mode_name, data in results.items():
        s = data["summary"]
        print(
            f"{mode_name:<10} | "
            f"{s['avg_precision_at_k']:<8.4f} | "
            f"{s['avg_recall_at_k']:<8.4f} | "
            f"{s['mrr']:<8.4f} | "
            f"{s['avg_retrieval_time_ms']:<8.2f} | "
            f"{s['avg_chunks_accessed_estimate']:<10.2f}"
        )
        if mode_name == "grover":
            print(
                " " * 12
                + f"grover timing breakdown (ms): embed={s['avg_grover_embedding_ms']:.2f}, "
                + f"sim={s['avg_grover_simulation_ms']:.2f}, "
                + f"post={s['avg_grover_postprocess_ms']:.2f}"
            )


def _aggregate_mode_runs(mode_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not mode_runs:
        return {}
    metrics = [
        "avg_precision_at_k",
        "avg_recall_at_k",
        "mrr",
        "avg_retrieval_time_ms",
        "avg_chunks_accessed_estimate",
        "avg_grover_embedding_ms",
        "avg_grover_simulation_ms",
        "avg_grover_postprocess_ms",
    ]
    agg: Dict[str, Any] = {"num_runs": len(mode_runs)}
    for metric in metrics:
        vals = [float(r["summary"][metric]) for r in mode_runs]
        agg[f"{metric}_mean"] = statistics.mean(vals)
        agg[f"{metric}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    agg["top_k"] = mode_runs[0]["summary"]["top_k"]
    agg["weights"] = mode_runs[0]["summary"]["weights"]
    return agg


def _print_aggregate_table(aggregate_results: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Aggregated Metrics Across Repeats ===")
    print(
        "mode       | p@k(mean±std)   | r@k(mean±std)   | mrr(mean±std)   | "
        "lat_ms(mean±std)"
    )
    print("-" * 102)
    for mode_name, agg in aggregate_results.items():
        print(
            f"{mode_name:<10} | "
            f"{agg['avg_precision_at_k_mean']:.4f}±{agg['avg_precision_at_k_std']:.4f} | "
            f"{agg['avg_recall_at_k_mean']:.4f}±{agg['avg_recall_at_k_std']:.4f} | "
            f"{agg['mrr_mean']:.4f}±{agg['mrr_std']:.4f} | "
            f"{agg['avg_retrieval_time_ms_mean']:.2f}±{agg['avg_retrieval_time_ms_std']:.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    config_path = _repo_path(args.config)
    queries_path = _repo_path(args.queries)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Queries not found: {queries_path}. "
            "Start by copying experiments/queries.example.json to experiments/queries.json."
        )

    cfg_base = RAGConfig.from_yaml(config_path)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries file must be a non-empty JSON list.")

    artifacts_dir = cfg_base.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, _sources, _meta = load_artifacts(
        artifacts_dir, args.index_prefix
    )
    print(f"Loaded artifacts from {artifacts_dir}. Chunks: {len(chunks)}")

    mode_plan: Dict[str, Dict[str, float]] = {}
    if args.mode in {"baseline", "both"}:
        w = _default_weight_template()
        w["faiss"] = 1.0
        mode_plan["baseline"] = w
    if args.mode in {"grover", "both"}:
        w = _default_weight_template()
        w["grover"] = 1.0
        mode_plan["grover"] = w

    sweep_points = _build_sweep_points(args, cfg_base)
    experiment_runs: List[Dict[str, Any]] = []
    for point_idx, point in enumerate(sweep_points):
        exp_id = (
            f"p{point_idx+1}_pool{point['grover_max_pool']}"
            f"_shots{point['grover_shots']}_iters{point['grover_iterations']}"
        )
        print(f"\n=== Running experiment {exp_id} ({args.repeats} repeat(s)) ===")
        runs_payload: List[Dict[str, Any]] = []
        per_mode_runs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in mode_plan}
        for run_idx in range(args.repeats):
            print(f"  - repeat {run_idx + 1}/{args.repeats}")
            run_results: Dict[str, Dict[str, Any]] = {}
            for mode_name, weights in mode_plan.items():
                cfg = copy.deepcopy(cfg_base)
                cfg.ranker_weights = weights
                cfg.grover_max_pool = point["grover_max_pool"]
                cfg.grover_shots = point["grover_shots"]
                cfg.grover_iterations = point["grover_iterations"]
                result = _evaluate_mode(mode_name, cfg, queries, faiss_idx, bm25_idx, chunks)
                run_results[mode_name] = result
                per_mode_runs[mode_name].append(result)
            runs_payload.append({"run_idx": run_idx, "results": run_results})
            if run_idx == 0:
                _print_summary_table(run_results)
        aggregate = {m: _aggregate_mode_runs(runs) for m, runs in per_mode_runs.items()}
        _print_aggregate_table(aggregate)
        experiment_runs.append(
            {
                "experiment_id": exp_id,
                "grover_params": point,
                "repeats": args.repeats,
                "aggregate": aggregate,
                "runs": runs_payload,
            }
        )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"retrieval_eval_{args.mode}_{ts}.json"
    payload = {
        "config_path": str(config_path),
        "queries_path": str(queries_path),
        "index_prefix": args.index_prefix,
        "repeats": args.repeats,
        "sweep": {
            "grover_max_pool": _parse_int_csv(args.sweep_grover_max_pool),
            "grover_shots": _parse_int_csv(args.sweep_grover_shots),
            "grover_iterations": _parse_int_csv(args.sweep_grover_iterations),
        },
        "experiments": experiment_runs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
