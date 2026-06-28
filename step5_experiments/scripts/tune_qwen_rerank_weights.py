#!/usr/bin/env python3
"""Tune lightweight MedHEval-Tree-V reranking weights on a Qwen output JSONL.

The core search uses class-aware TF-IDF inverted retrieval before evidence
reranking, avoiding a full scan over all answer templates for every query.
"""

from __future__ import annotations

import argparse
import math
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qwen_sanity as ev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--retrieval-k", type=int, default=80)
    parser.add_argument("--hallucination-alpha", type=float, default=0.0)
    return parser.parse_args()


class TfidfTemplateIndex:
    def __init__(self, by_class: dict[str, list[str]], global_answers: list[str]) -> None:
        self.by_class = by_class
        self.global_answers = global_answers
        self.answers: list[str] = []
        self.answer_class_ids: dict[str, set[int]] = defaultdict(set)
        seen = set()
        for cls, answers in by_class.items():
            for answer in answers:
                if answer not in seen:
                    seen.add(answer)
                    self.answers.append(answer)
                self.answer_class_ids[cls].add(self.answers.index(answer))
        for answer in global_answers:
            if answer not in seen:
                seen.add(answer)
                self.answers.append(answer)

        doc_tokens = [Counter(ev.tokenize(answer)) for answer in self.answers]
        df = Counter()
        for counts in doc_tokens:
            df.update(counts.keys())
        n_docs = max(len(self.answers), 1)
        self.idf = {tok: math.log((1 + n_docs) / (1 + freq)) + 1.0 for tok, freq in df.items()}
        self.doc_vecs: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self.inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_id, counts in enumerate(doc_tokens):
            vec = {tok: (1.0 + math.log(tf)) * self.idf[tok] for tok, tf in counts.items() if tf > 0}
            norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
            self.doc_vecs.append(vec)
            self.doc_norms.append(norm)
            for tok, weight in vec.items():
                self.inverted[tok].append((doc_id, weight))

    def allowed_doc_ids(self, row: dict) -> set[int] | None:
        allowed: set[int] = set()
        for cls in row.get("question_class") or []:
            allowed.update(self.answer_class_ids.get(cls, set()))
        return allowed or None

    def retrieve(self, row: dict, query: str, top_k: int) -> list[str]:
        counts = Counter(ev.tokenize(query))
        if not counts:
            return self.class_fallback(row, top_k)
        qvec = {tok: (1.0 + math.log(tf)) * self.idf.get(tok, 1.0) for tok, tf in counts.items() if tf > 0}
        qnorm = math.sqrt(sum(weight * weight for weight in qvec.values())) or 1.0
        allowed = self.allowed_doc_ids(row)
        scores: dict[int, float] = defaultdict(float)
        for tok, qweight in qvec.items():
            for doc_id, dweight in self.inverted.get(tok, []):
                if allowed is not None and doc_id not in allowed:
                    continue
                scores[doc_id] += qweight * dweight
        if not scores:
            return self.class_fallback(row, top_k)
        ranked = sorted(
            scores.items(),
            key=lambda item: item[1] / (qnorm * self.doc_norms[item[0]]),
            reverse=True,
        )
        return [self.answers[doc_id] for doc_id, _ in ranked[:top_k]]

    def class_fallback(self, row: dict, top_k: int) -> list[str]:
        candidates = []
        seen = set()
        for cls in row.get("question_class") or []:
            for answer in self.by_class.get(cls, []):
                if answer not in seen:
                    candidates.append(answer)
                    seen.add(answer)
                if len(candidates) >= top_k:
                    return candidates
        for answer in self.global_answers:
            if answer not in seen:
                candidates.append(answer)
                seen.add(answer)
            if len(candidates) >= top_k:
                break
        return candidates


def retrieval_query(row: dict) -> str:
    evidence = row.get("qwen_evidence") if isinstance(row.get("qwen_evidence"), dict) else {}
    parts = [
        row.get("question", ""),
        row.get("qwen_direct", ""),
        str(evidence.get("lesion_presence", "")),
        str(evidence.get("lesion_type", "")),
        str(evidence.get("lesion_count", "")),
        str(evidence.get("location", "")),
        str(evidence.get("instrument_presence", "")),
        str(evidence.get("text_overlay_presence", "")),
        str(evidence.get("abnormality_presence", "")),
        str(evidence.get("evidence_sentence", "")),
    ]
    return " ".join(parts)


def score_candidate(
    row: dict,
    candidate: str,
    *,
    w_evidence: float,
    w_direct: float,
    w_question: float,
    w_penalty: float,
    w_length: float,
) -> float:
    evidence_context = ev.evidence_text(row)
    direct = row.get("qwen_direct", "")
    question = row.get("question", "")
    return (
        w_evidence * ev.token_f1(candidate, evidence_context)
        + w_direct * ev.token_f1(candidate, direct)
        + w_question * ev.token_f1(candidate, question)
        - w_penalty * ev.contradiction_penalty(candidate, row)
        - w_length * len(ev.normalize(candidate).split())
    )


def predict(row: dict, index: TfidfTemplateIndex, weights: dict[str, float], retrieval_k: int) -> str:
    candidates = index.retrieve(row, retrieval_query(row), retrieval_k)
    best = candidates[0]
    best_score = -1e9
    for candidate in candidates:
        score = score_candidate(row, candidate, **weights)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def summarize(name: str, rows: list[dict], preds: list[str]) -> dict:
    pairs = [(pred, row["answer"], row) for pred, row in zip(preds, rows)]
    return ev.metric_rows(name, pairs)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_class, global_answers = ev.build_answer_candidates(top_k_per_class=args.candidate_k)
    index = TfidfTemplateIndex(by_class, global_answers)

    grid = []
    for w_evidence, w_direct, w_question, w_penalty, w_length in product(
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 0.25, 0.5],
        [0.0, 0.5, 1.0],
        [0.0, 0.005],
    ):
        if w_evidence == 0.0 and w_direct == 0.0 and w_question == 0.0:
            continue
        weights = {
            "w_evidence": w_evidence,
            "w_direct": w_direct,
            "w_question": w_question,
            "w_penalty": w_penalty,
            "w_length": w_length,
        }
        preds = [predict(row, index, weights, args.retrieval_k) for row in rows]
        metrics = summarize("grid", rows, preds)
        objective = metrics["token_f1"] - args.hallucination_alpha * metrics["hallucination_proxy_rate"]
        grid.append({**weights, **metrics, "objective": objective})

    grid.sort(key=lambda x: (x["objective"], x["token_f1"], x["exact_match"], -x["hallucination_proxy_rate"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "w_evidence",
            "w_direct",
            "w_question",
            "w_penalty",
            "w_length",
            "n",
            "objective",
            "exact_match",
            "token_f1",
            "hallucination_proxy_rate",
            "complexity_accuracy",
            "top_class_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(grid[: args.top_k], 1):
            writer.writerow({"rank": rank, **{k: row[k] for k in fieldnames if k != "rank"}})

    if args.predictions:
        best_weights = {k: grid[0][k] for k in ["w_evidence", "w_direct", "w_question", "w_penalty", "w_length"]}
        preds = [predict(row, index, best_weights, args.retrieval_k) for row in rows]
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

    print(args.out)
    print(json.dumps(grid[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
