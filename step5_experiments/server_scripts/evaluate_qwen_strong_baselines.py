#!/usr/bin/env python3
"""Evaluate strong Qwen prompt baselines against existing Kvasir-VQA-x1 results."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "step5_experiments" / "results"
REPORT_DIR = ROOT / "step5_experiments" / "reports"
STRONG_PATH = RESULT_DIR / "qwen_strong_baselines.jsonl"
TFIDF_PATH = RESULT_DIR / "qwen_full_tfidf_fast_predictions.csv"
GATE_PATH = RESULT_DIR / "reliability_gate_metrics.csv"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "as", "by",
    "this", "that", "there", "it", "its", "any",
}


def norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return norm(text).split()


def token_f1(pred: str, gold: str) -> float:
    pred_toks = tokens(pred)
    gold_toks = tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact(pred: str, gold: str) -> float:
    return 1.0 if norm(pred) == norm(gold) else 0.0


def lexical_drift(pred: str, gold: str, threshold: float = 0.5) -> float:
    pred_toks = [t for t in tokens(pred) if t not in STOPWORDS]
    gold_set = {t for t in tokens(gold) if t not in STOPWORDS}
    if not pred_toks:
        return 0.0
    unseen = sum(1 for t in pred_toks if t not in gold_set)
    return 1.0 if unseen / len(pred_toks) > threshold else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_strong_rows() -> list[dict]:
    rows = []
    with STRONG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_rows() -> list[dict]:
    with TFIDF_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_predictions(rows: list[dict], field: str) -> dict[str, float | int | str]:
    f1s, exacts, drifts, lengths = [], [], [], []
    errors = 0
    empty = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        pred = row.get(field) or ""
        gold = row.get("answer") or ""
        if not str(pred).strip():
            empty += 1
        f1s.append(token_f1(str(pred), str(gold)))
        exacts.append(exact(str(pred), str(gold)))
        drifts.append(lexical_drift(str(pred), str(gold)))
        lengths.append(len(tokens(str(pred))))
    return {
        "method": field,
        "n": len(rows),
        "errors": errors,
        "empty": empty,
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "avg_len": mean(lengths),
    }


def summarize_existing(rows: list[dict], method: str, pred_field: str, f1_field: str, exact_field: str) -> dict[str, float | int | str]:
    f1s, exacts, drifts, lengths = [], [], [], []
    for row in rows:
        pred = row[pred_field]
        gold = row["gold"]
        f1s.append(float(row[f1_field]))
        exacts.append(float(row[exact_field]))
        drifts.append(lexical_drift(pred, gold))
        lengths.append(len(tokens(pred)))
    return {
        "method": method,
        "n": len(rows),
        "errors": 0,
        "empty": 0,
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "avg_len": mean(lengths),
    }


def load_gate_eval() -> list[dict[str, str]]:
    if not GATE_PATH.exists():
        return []
    with GATE_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def per_complexity(rows: list[dict], field: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["complexity"])].append(row)
    out = []
    for complexity, group in sorted(grouped.items()):
        item = summarize_predictions(group, field)
        item["complexity"] = complexity
        out.append(item)
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    strong_rows = load_strong_rows()
    existing_rows = load_existing_rows()

    strong_methods = [
        "qwen_constrained_answer",
        "qwen_class_constrained_answer",
        "qwen_evidence_then_answer",
    ]
    rows = [
        summarize_existing(
            existing_rows,
            "qwen_direct_original",
            "qwen_direct_prediction",
            "qwen_direct_token_f1",
            "qwen_direct_exact",
        ),
        summarize_existing(
            existing_rows,
            "tfidf_direct_template",
            "tfidf_direct_template_prediction",
            "tfidf_direct_template_token_f1",
            "tfidf_direct_template_exact",
        ),
        summarize_existing(
            existing_rows,
            "tfidf_evidence_rerank",
            "tfidf_evidence_rerank_prediction",
            "tfidf_evidence_rerank_token_f1",
            "tfidf_evidence_rerank_exact",
        ),
    ]
    rows.extend(summarize_predictions(strong_rows, field) for field in strong_methods)

    gate_rows = load_gate_eval()
    for row in gate_rows:
        if row.get("split") == "eval_rest" and row.get("method") == "bucket_gate_dev_policy":
            rows.append({
                "method": "bucket_gate_dev_policy_eval",
                "n": int(row["n"]),
                "errors": 0,
                "empty": 0,
                "exact": float(row["exact"]),
                "token_f1": float(row["token_f1"]),
                "lexical_drift": float(row["hallucination_proxy"]),
                "avg_len": "",
            })
        if row.get("split") == "full" and row.get("method") == "bucket_gate_dev_policy":
            rows.append({
                "method": "bucket_gate_dev_policy_full",
                "n": int(row["n"]),
                "errors": 0,
                "empty": 0,
                "exact": float(row["exact"]),
                "token_f1": float(row["token_f1"]),
                "lexical_drift": float(row["hallucination_proxy"]),
                "avg_len": "",
            })

    rows = sorted(rows, key=lambda r: float(r["token_f1"]), reverse=True)
    out_csv = RESULT_DIR / "qwen_strong_baseline_metrics.csv"
    write_csv(out_csv, rows)

    complexity_rows = []
    for field in strong_methods:
        complexity_rows.extend(per_complexity(strong_rows, field))
    write_csv(RESULT_DIR / "qwen_strong_baseline_by_complexity.csv", complexity_rows)

    report = [
        "# Strong Qwen Prompt Baseline Evaluation",
        "",
        f"Rows evaluated: {len(strong_rows)}.",
        "",
        "## Overall",
        "",
        md_table(rows, ["method", "n", "errors", "empty", "exact", "token_f1", "lexical_drift", "avg_len"]),
        "",
        "## Strong Prompt Baselines by Complexity",
        "",
        md_table(complexity_rows, ["method", "complexity", "n", "exact", "token_f1", "lexical_drift", "avg_len"]),
        "",
    ]
    out_report = REPORT_DIR / "qwen_strong_baseline_evaluation.md"
    out_report.write_text("\n".join(report), encoding="utf-8")
    print(out_report)


if __name__ == "__main__":
    main()
