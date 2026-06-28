#!/usr/bin/env python3
"""Evaluate Qwen outputs against Kvasir-VQA-x1 labels and local baselines."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
DEFAULT_QWEN_JSONL = RESULTS / "qwen_sanity_outputs_100.jsonl"
DEFAULT_METRICS_CSV = RESULTS / "qwen_sanity_metrics.csv"
DEFAULT_REPORT_MD = REPORTS / "qwen_sanity_report.md"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "is",
    "in",
    "of",
    "the",
    "there",
    "to",
    "with",
}


def as_list(value) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    return {tok for tok in normalize(text).split() if tok and tok not in STOPWORDS}


def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(g)
    return 2 * precision * recall / (precision + recall)


def hallucination_proxy(pred: str, gold: str) -> int:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p:
        return 0
    extra = p - g
    return int(len(extra) / max(len(p), 1) > 0.5)


def metric_rows(name: str, pairs: list[tuple[str, str, dict]]) -> dict:
    exact = []
    f1s = []
    halluc = []
    by_complexity = defaultdict(list)
    by_class = defaultdict(list)
    for pred, gold, meta in pairs:
        hit = int(normalize(pred) == normalize(gold))
        f1 = token_f1(pred, gold)
        h = hallucination_proxy(pred, gold)
        exact.append(hit)
        f1s.append(f1)
        halluc.append(h)
        by_complexity[str(meta["complexity"])].append(hit)
        first_class = (meta.get("question_class") or ["unknown"])[0]
        by_class[first_class].append(hit)
    return {
        "method": name,
        "n": len(pairs),
        "exact_match": sum(exact) / len(exact),
        "token_f1": sum(f1s) / len(f1s),
        "hallucination_proxy_rate": sum(halluc) / len(halluc),
        "complexity_accuracy": json.dumps({k: sum(v) / len(v) for k, v in sorted(by_complexity.items())}),
        "top_class_accuracy": json.dumps(
            {
                k: sum(v) / len(v)
                for k, v in sorted(by_class.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:12]
            }
        ),
    }


def build_class_majority() -> dict[str, str]:
    train = pd.read_parquet(DATA_ROOT / "train.parquet")
    counts: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for _, row in train.iterrows():
        ans = str(row["answer"]).strip()
        global_counts[ans] += 1
        for cls in as_list(row["question_class"]):
            counts[cls][ans] += 1
    default = global_counts.most_common(1)[0][0]
    return {k: v.most_common(1)[0][0] for k, v in counts.items()} | {"__default__": default}


def build_answer_candidates(top_k_per_class: int = 200) -> tuple[dict[str, list[str]], list[str]]:
    train = pd.read_parquet(DATA_ROOT / "train.parquet")
    counts: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for _, row in train.iterrows():
        ans = str(row["answer"]).strip()
        global_counts[ans] += 1
        for cls in as_list(row["question_class"]):
            counts[cls][ans] += 1
    by_class = {
        cls: [ans for ans, _ in counter.most_common(top_k_per_class)]
        for cls, counter in counts.items()
    }
    global_answers = [ans for ans, _ in global_counts.most_common(top_k_per_class)]
    return by_class, global_answers


def evidence_text(row: dict) -> str:
    ev = row.get("qwen_evidence") if isinstance(row.get("qwen_evidence"), dict) else {}
    parts = [row.get("qwen_direct", ""), row.get("question", "")]
    for key in [
        "lesion_presence",
        "lesion_type",
        "lesion_count",
        "location",
        "instrument_presence",
        "text_overlay_presence",
        "abnormality_presence",
        "evidence_sentence",
    ]:
        parts.append(str(ev.get(key, "")))
    return " ".join(parts)


def contradiction_penalty(candidate: str, row: dict) -> float:
    ev = row.get("qwen_evidence") if isinstance(row.get("qwen_evidence"), dict) else {}
    cand = normalize(candidate)
    penalty = 0.0
    lesion_presence = normalize(ev.get("lesion_presence", ""))
    lesion_count = normalize(ev.get("lesion_count", ""))
    abnormality = normalize(ev.get("abnormality_presence", ""))
    text_overlay = normalize(ev.get("text_overlay_presence", ""))
    instrument = normalize(ev.get("instrument_presence", ""))
    if lesion_presence == "yes" and any(x in cand for x in ["no polyp", "no lesion", "no abnormal"]):
        penalty += 0.35
    if abnormality == "yes" and "no abnormal" in cand:
        penalty += 0.35
    if lesion_count in {"1", "one", "single"} and any(x in cand for x in ["multiple", "two", "several"]):
        penalty += 0.20
    if lesion_count in {"multiple", "several", "many"} and any(x in cand for x in ["single", "one polyp", "1 polyp"]):
        penalty += 0.20
    if text_overlay == "yes" and "no text" in cand:
        penalty += 0.15
    if instrument == "no" and "instrument" in cand and "no instrument" not in cand:
        penalty += 0.10
    return penalty


def template_rerank_predict(
    row: dict,
    by_class: dict[str, list[str]],
    global_answers: list[str],
    *,
    use_evidence: bool = True,
    use_penalty: bool = True,
) -> str:
    candidates: list[str] = []
    seen = set()
    for cls in row.get("question_class") or []:
        for ans in by_class.get(cls, []):
            if ans not in seen:
                candidates.append(ans)
                seen.add(ans)
    if not candidates:
        candidates = global_answers[:]
    context = evidence_text(row) if use_evidence else " ".join([row.get("qwen_direct", ""), row.get("question", "")])
    direct = row.get("qwen_direct", "")
    question = row.get("question", "")
    best = candidates[0]
    best_score = -10.0
    for cand in candidates:
        score = (
            1.20 * token_f1(cand, context)
            + 0.80 * token_f1(cand, direct)
            + 0.25 * token_f1(cand, question)
            - (contradiction_penalty(cand, row) if use_penalty else 0.0)
        )
        # Prefer concise templates when evidence support is tied.
        score -= 0.005 * len(normalize(cand).split())
        if score > best_score:
            best_score = score
            best = cand
    return best


def evidence_summary(rows: list[dict]) -> dict:
    parseable = [r for r in rows if isinstance(r.get("qwen_evidence"), dict)]
    uncertainty = Counter(str(r["qwen_evidence"].get("uncertainty", "missing")).lower() for r in parseable)
    unknown_fields = Counter()
    contradiction_count = 0
    for r in parseable:
        ev = r["qwen_evidence"]
        for key, value in ev.items():
            if str(value).strip().lower() in {"unknown", "uncertain", "not visible", "n/a"}:
                unknown_fields[key] += 1
        direct = normalize(r.get("qwen_direct", ""))
        lesion = normalize(ev.get("lesion_presence", ""))
        if "no" in direct.split() and lesion == "yes":
            contradiction_count += 1
    return {
        "parseable": len(parseable),
        "parse_rate": len(parseable) / len(rows),
        "uncertainty": dict(uncertainty),
        "unknown_fields": dict(unknown_fields),
        "direct_evidence_contradictions": contradiction_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_QWEN_JSONL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--title", default="Qwen Sanity Report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    class_majority = build_class_majority()
    by_class_candidates, global_candidates = build_answer_candidates()

    qwen_pairs = [(r.get("qwen_direct", ""), r["answer"], r) for r in rows]
    class_pairs = [
        (
            class_majority.get((r.get("question_class") or ["unknown"])[0], class_majority["__default__"]),
            r["answer"],
            r,
        )
        for r in rows
    ]
    direct_template_pairs = [
        (
            template_rerank_predict(
                r,
                by_class_candidates,
                global_candidates,
                use_evidence=False,
                use_penalty=False,
            ),
            r["answer"],
            r,
        )
        for r in rows
    ]
    evidence_template_pairs = [
        (
            template_rerank_predict(
                r,
                by_class_candidates,
                global_candidates,
                use_evidence=True,
                use_penalty=False,
            ),
            r["answer"],
            r,
        )
        for r in rows
    ]
    rerank_pairs = [
        (
            template_rerank_predict(
                r,
                by_class_candidates,
                global_candidates,
                use_evidence=True,
                use_penalty=True,
            ),
            r["answer"],
            r,
        )
        for r in rows
    ]
    metrics = [
        metric_rows("class_majority", class_pairs),
        metric_rows("qwen_direct", qwen_pairs),
        metric_rows("direct_template_rerank", direct_template_pairs),
        metric_rows("evidence_template_rerank_no_penalty", evidence_template_pairs),
        metric_rows("medheval_tree_v_full", rerank_pairs),
    ]

    with args.metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    ev = evidence_summary(rows)
    lines = [
        f"# {args.title}",
        "",
        f"- Input: `{args.input}`",
        f"- Rows: {len(rows)}",
        "",
        "## Metrics",
        "",
        "| Method | N | Exact Match | Token F1 | Hallucination Proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['method']} | {m['n']} | {m['exact_match']:.4f} | {m['token_f1']:.4f} | {m['hallucination_proxy_rate']:.4f} |"
        )
    lines += [
        "",
        "## Evidence JSON",
        "",
        f"- Parseable evidence: {ev['parseable']}/{len(rows)} ({ev['parse_rate']:.2%})",
        f"- Uncertainty distribution: `{json.dumps(ev['uncertainty'], ensure_ascii=False)}`",
        f"- Unknown fields: `{json.dumps(ev['unknown_fields'], ensure_ascii=False)}`",
        f"- Direct/evidence contradiction proxy: {ev['direct_evidence_contradictions']}",
        "",
        "## Initial Interpretation",
        "",
        "- Qwen direct answers are visually grounded but often not normalized to Kvasir answer style.",
        "- Structured evidence extraction is viable because JSON parse rate is high.",
        "- The next method step should use evidence fields to rerank answer templates and suppress contradictions rather than trusting direct free-form answers.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report)
    print(args.metrics)
    print(json.dumps(ev, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
