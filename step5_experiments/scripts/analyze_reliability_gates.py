#!/usr/bin/env python3
"""Analyze reliability/complexity gates for MedHEval-Tree-V."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qwen_sanity as ev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--qwen-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dev-n", type=int, default=529)
    parser.add_argument("--min-class-dev-n", type=int, default=8)
    return parser.parse_args()


def load_rows(predictions: Path, qwen_jsonl: Path) -> list[dict]:
    qwen_by_id = {}
    for line in qwen_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qwen_by_id[row["id"]] = row

    rows = []
    with predictions.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qwen = qwen_by_id[row["id"]]
            ev_obj = qwen.get("qwen_evidence") if isinstance(qwen.get("qwen_evidence"), dict) else {}
            direct = ev.normalize(qwen.get("qwen_direct", ""))
            lesion = ev.normalize(ev_obj.get("lesion_presence", ""))
            row["primary_class"] = (json.loads(row["question_class"]) or ["unknown"])[0]
            row["uncertainty"] = ev.normalize(ev_obj.get("uncertainty", ""))
            row["direct_evidence_contradiction"] = int("no" in direct.split() and lesion == "yes")
            rows.append(row)
    return rows


def metric(xs: list[dict], pred_fn: Callable[[dict], str]) -> dict:
    preds = [pred_fn(row) for row in xs]
    golds = [row["gold"] for row in xs]
    return {
        "n": len(xs),
        "exact": sum(ev.normalize(p) == ev.normalize(g) for p, g in zip(preds, golds)) / len(xs),
        "token_f1": sum(ev.token_f1(p, g) for p, g in zip(preds, golds)) / len(xs),
        "hallucination_proxy": sum(ev.hallucination_proxy(p, g) for p, g in zip(preds, golds)) / len(xs),
    }


def choose_class_set(dev: list[dict], min_n: int) -> set[str]:
    by_class = defaultdict(list)
    for row in dev:
        by_class[row["primary_class"]].append(row)
    selected = set()
    for cls, xs in by_class.items():
        if len(xs) < min_n:
            continue
        direct = metric(xs, lambda r: r["tfidf_direct_template_prediction"])["token_f1"]
        evidence = metric(xs, lambda r: r["tfidf_evidence_rerank_prediction"])["token_f1"]
        if evidence > direct:
            selected.add(cls)
    return selected


def choose_bucket_policy(dev: list[dict], min_n: int) -> dict[tuple[str, str], str]:
    buckets = defaultdict(list)
    for row in dev:
        buckets[(row["complexity"], row["primary_class"])].append(row)
    policy = {}
    for bucket, xs in buckets.items():
        if len(xs) < min_n:
            continue
        direct = metric(xs, lambda r: r["tfidf_direct_template_prediction"])["token_f1"]
        evidence = metric(xs, lambda r: r["tfidf_evidence_rerank_prediction"])["token_f1"]
        policy[bucket] = "evidence" if evidence > direct else "direct"
    return policy


def main() -> int:
    args = parse_args()
    rows = load_rows(args.predictions, args.qwen_jsonl)
    dev = rows[: args.dev_n]
    eval_rows = rows[args.dev_n :]
    full = rows

    selected_classes = choose_class_set(dev, args.min_class_dev_n)
    bucket_policy = choose_bucket_policy(dev, args.min_class_dev_n)

    strategies: dict[str, Callable[[dict], str]] = {
        "tfidf_direct_template": lambda r: r["tfidf_direct_template_prediction"],
        "tfidf_evidence_rerank": lambda r: r["tfidf_evidence_rerank_prediction"],
        "tfidf_multiobjective": lambda r: r["tfidf_multiobjective_prediction"],
        "complexity_gate_23": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if int(r["complexity"]) >= 2
            else r["tfidf_direct_template_prediction"]
        ),
        "complexity_gate_3": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if int(r["complexity"]) >= 3
            else r["tfidf_direct_template_prediction"]
        ),
        "complexity_gate_3_low_uncertainty": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if int(r["complexity"]) >= 3 and r["uncertainty"] in {"low", "moderate"}
            else r["tfidf_direct_template_prediction"]
        ),
        "complexity_gate_3_no_contradiction": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if int(r["complexity"]) >= 3 and not r["direct_evidence_contradiction"]
            else r["tfidf_direct_template_prediction"]
        ),
        "class_gate_dev_positive": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if r["primary_class"] in selected_classes
            else r["tfidf_direct_template_prediction"]
        ),
        "complexity_or_class_gate": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if int(r["complexity"]) >= 3 or r["primary_class"] in selected_classes
            else r["tfidf_direct_template_prediction"]
        ),
        "bucket_gate_dev_policy": lambda r: (
            r["tfidf_evidence_rerank_prediction"]
            if bucket_policy.get((r["complexity"], r["primary_class"]), "direct") == "evidence"
            else r["tfidf_direct_template_prediction"]
        ),
    }

    records = []
    for split_name, xs in [("dev_first529", dev), ("eval_rest", eval_rows), ("full", full)]:
        for name, pred_fn in strategies.items():
            records.append({"split": split_name, "method": name, **metric(xs, pred_fn)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "method", "n", "exact", "token_f1", "hallucination_proxy"],
        )
        writer.writeheader()
        writer.writerows(records)

    best_eval = sorted(
        [r for r in records if r["split"] == "eval_rest"],
        key=lambda r: (r["token_f1"], r["exact"], -r["hallucination_proxy"]),
        reverse=True,
    )
    lines = [
        "# Reliability Gate Analysis",
        "",
        f"- Predictions: `{args.predictions}`",
        f"- Qwen JSONL: `{args.qwen_jsonl}`",
        f"- Development rows: first {args.dev_n}",
        f"- Evaluation rows: {len(eval_rows)}",
        f"- Dev-positive evidence classes: `{json.dumps(sorted(selected_classes), ensure_ascii=False)}`",
        f"- Bucket policy entries: {len(bucket_policy)}",
        "",
        "## Evaluation Split Ranking",
        "",
        "| Method | N | Exact | Token F1 | Hallucination Proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_eval:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['exact']:.4f} | {row['token_f1']:.4f} | {row['hallucination_proxy']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The split uses early rows only for gate selection and reports the remaining rows as the evaluation split.",
        "- Complexity-only gates are more robust than heavily class-fitted bucket gates if their evaluation gains are comparable.",
        "- The final paper method should prefer the simplest gate that improves Token F1 without a large hallucination-proxy penalty.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report)
    print(args.out)
    print(json.dumps(best_eval[:5], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
