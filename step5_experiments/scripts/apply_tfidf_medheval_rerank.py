#!/usr/bin/env python3
"""Apply the selected TF-IDF MedHEval-Tree-V reranker to Qwen outputs."""

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
from tune_qwen_rerank_weights import TfidfTemplateIndex, predict


DEFAULT_WEIGHTS = {
    "w_evidence": 1.5,
    "w_direct": 1.5,
    "w_question": 0.5,
    "w_penalty": 0.5,
    "w_length": 0.005,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--retrieval-k", type=int, default=40)
    parser.add_argument("--w-evidence", type=float, default=DEFAULT_WEIGHTS["w_evidence"])
    parser.add_argument("--w-direct", type=float, default=DEFAULT_WEIGHTS["w_direct"])
    parser.add_argument("--w-question", type=float, default=DEFAULT_WEIGHTS["w_question"])
    parser.add_argument("--w-penalty", type=float, default=DEFAULT_WEIGHTS["w_penalty"])
    parser.add_argument("--w-length", type=float, default=DEFAULT_WEIGHTS["w_length"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_class, global_answers = ev.build_answer_candidates(top_k_per_class=args.candidate_k)
    index = TfidfTemplateIndex(by_class, global_answers)
    weights = {
        "w_evidence": args.w_evidence,
        "w_direct": args.w_direct,
        "w_question": args.w_question,
        "w_penalty": args.w_penalty,
        "w_length": args.w_length,
    }
    preds = [predict(row, index, weights, args.retrieval_k) for row in rows]

    method_pairs = [(pred, row["answer"], row) for pred, row in zip(preds, rows)]
    metrics = ev.metric_rows("medheval_tree_v_tfidf_retrieval", method_pairs)
    metrics["weights"] = json.dumps(weights, ensure_ascii=False)
    metrics["candidate_k"] = args.candidate_k
    metrics["retrieval_k"] = args.retrieval_k

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "gold",
                "prediction",
                "qwen_direct",
                "complexity",
                "question_class",
                "token_f1",
                "exact",
            ],
        )
        writer.writeheader()
        for row, pred in zip(rows, preds):
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold": row["answer"],
                    "prediction": pred,
                    "qwen_direct": row.get("qwen_direct", ""),
                    "complexity": row.get("complexity"),
                    "question_class": json.dumps(row.get("question_class", []), ensure_ascii=False),
                    "token_f1": ev.token_f1(pred, row["answer"]),
                    "exact": int(ev.normalize(pred) == ev.normalize(row["answer"])),
                }
            )

    print(args.metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
