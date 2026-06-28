#!/usr/bin/env python3
"""Fast full-test evaluation for Qwen and TF-IDF MedHEval-Tree-V variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qwen_sanity as ev
from tune_qwen_rerank_weights import TfidfTemplateIndex, retrieval_query, score_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--retrieval-k", type=int, default=40)
    return parser.parse_args()


def direct_query(row: dict) -> str:
    return " ".join([row.get("question", ""), row.get("qwen_direct", "")])


def predict_with_index(
    row: dict,
    index: TfidfTemplateIndex,
    *,
    query: str,
    weights: dict[str, float],
    retrieval_k: int,
) -> str:
    candidates = index.retrieve(row, query, retrieval_k)
    best = candidates[0]
    best_score = -1e9
    for candidate in candidates:
        score = score_candidate(row, candidate, **weights)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def write_predictions(path: Path, rows: list[dict], prediction_sets: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "gold",
        "qwen_direct",
        "complexity",
        "question_class",
        *[f"{name}_prediction" for name in prediction_sets],
        *[f"{name}_token_f1" for name in prediction_sets],
        *[f"{name}_exact" for name in prediction_sets],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {
                "id": row["id"],
                "question": row["question"],
                "gold": row["answer"],
                "qwen_direct": row.get("qwen_direct", ""),
                "complexity": row.get("complexity"),
                "question_class": json.dumps(row.get("question_class", []), ensure_ascii=False),
            }
            for name, preds in prediction_sets.items():
                pred = preds[idx]
                out[f"{name}_prediction"] = pred
                out[f"{name}_token_f1"] = ev.token_f1(pred, row["answer"])
                out[f"{name}_exact"] = int(ev.normalize(pred) == ev.normalize(row["answer"]))
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_majority = ev.build_class_majority()
    by_class, global_answers = ev.build_answer_candidates(top_k_per_class=args.candidate_k)
    index = TfidfTemplateIndex(by_class, global_answers)

    class_preds = [
        class_majority.get((row.get("question_class") or ["unknown"])[0], class_majority["__default__"])
        for row in rows
    ]
    qwen_direct_preds = [row.get("qwen_direct", "") for row in rows]

    direct_weights = {
        "w_evidence": 0.0,
        "w_direct": 1.0,
        "w_question": 0.25,
        "w_penalty": 0.0,
        "w_length": 0.0,
    }
    evidence_weights = {
        "w_evidence": 1.5,
        "w_direct": 1.5,
        "w_question": 0.5,
        "w_penalty": 0.5,
        "w_length": 0.005,
    }
    multiobjective_weights = {
        "w_evidence": 0.0,
        "w_direct": 1.0,
        "w_question": 0.25,
        "w_penalty": 0.0,
        "w_length": 0.0,
    }

    direct_tfidf_preds = [
        predict_with_index(
            row,
            index,
            query=direct_query(row),
            weights=direct_weights,
            retrieval_k=args.retrieval_k,
        )
        for row in rows
    ]
    evidence_tfidf_preds = [
        predict_with_index(
            row,
            index,
            query=retrieval_query(row),
            weights=evidence_weights,
            retrieval_k=args.retrieval_k,
        )
        for row in rows
    ]
    multiobjective_preds = [
        predict_with_index(
            row,
            index,
            query=retrieval_query(row),
            weights=multiobjective_weights,
            retrieval_k=args.retrieval_k,
        )
        for row in rows
    ]

    prediction_sets = {
        "class_majority": class_preds,
        "qwen_direct": qwen_direct_preds,
        "tfidf_direct_template": direct_tfidf_preds,
        "tfidf_evidence_rerank": evidence_tfidf_preds,
        "tfidf_multiobjective": multiobjective_preds,
        "medheval_tree_v_complexity_gate_23": [
            evidence_pred if int(row.get("complexity", 1)) >= 2 else direct_pred
            for row, direct_pred, evidence_pred in zip(rows, direct_tfidf_preds, evidence_tfidf_preds)
        ],
        "medheval_tree_v_complexity_gate_3": [
            evidence_pred if int(row.get("complexity", 1)) >= 3 else direct_pred
            for row, direct_pred, evidence_pred in zip(rows, direct_tfidf_preds, evidence_tfidf_preds)
        ],
    }
    metrics = [
        ev.metric_rows(name, [(pred, row["answer"], row) for pred, row in zip(preds, rows)])
        for name, preds in prediction_sets.items()
    ]

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    write_predictions(args.predictions, rows, prediction_sets)

    evidence = ev.evidence_summary(rows)
    lines = [
        "# Full TF-IDF Fast Evaluation",
        "",
        f"- Input: `{args.input}`",
        f"- Rows: {len(rows)}",
        f"- Candidate K per class: {args.candidate_k}",
        f"- Retrieval K: {args.retrieval_k}",
        "",
        "## Metrics",
        "",
        "| Method | N | Exact Match | Token F1 | Hallucination Proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['method']} | {item['n']} | {item['exact_match']:.4f} | {item['token_f1']:.4f} | {item['hallucination_proxy_rate']:.4f} |"
        )
    lines += [
        "",
        "## Evidence JSON",
        "",
        f"- Parseable evidence: {evidence['parseable']}/{len(rows)} ({evidence['parse_rate']:.2%})",
        f"- Direct/evidence contradiction proxy: {evidence['direct_evidence_contradictions']}",
        f"- Uncertainty distribution: `{json.dumps(evidence['uncertainty'], ensure_ascii=False)}`",
        f"- Unknown fields: `{json.dumps(evidence['unknown_fields'], ensure_ascii=False)}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report)
    print(args.metrics)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
