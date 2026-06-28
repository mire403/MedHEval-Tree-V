#!/usr/bin/env python3
"""SLAKE external validation for Reliability-Gated MedHEval-Tree-V.

The script has three subcommands:
- inspect: summarize SLAKE splits and image coverage.
- qwen: run compact Qwen visual prompts on a split/language subset.
- eval: evaluate template retrieval and, when Qwen outputs exist, answer
  normalization/routing variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


try:
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
except Exception:  # imported only for qwen subcommand
    torch = None
    AutoProcessor = None
    Qwen3_5ForConditionalGeneration = None


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "models/Qwen3.5-9B")


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokens(text: str) -> list[str]:
    text = normalize_text(text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    latin = re.findall(r"[a-z0-9]+", text)
    return latin + cjk


def token_f1(pred: str, gold: str) -> float:
    pt = tokens(pred)
    gt = tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc = Counter(pt)
    gc = Counter(gt)
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def image_path(base: Path, item: dict[str, Any]) -> Path:
    return base / "imgs_unzipped" / "imgs" / str(item["img_name"])


def split_rows(data_dir: Path, split: str, lang: str) -> list[dict[str, Any]]:
    rows = load_json(data_dir / f"{split}.json")
    if lang != "all":
        rows = [r for r in rows if r.get("q_lang") == lang]
    return rows


def class_keys(item: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(item.get("content_type", "unknown")).lower(), str(item.get("answer_type", "unknown")).lower()),
        (str(item.get("content_type", "unknown")).lower(), "*"),
        ("*", str(item.get("answer_type", "unknown")).lower()),
        ("*", "*"),
    ]


class TemplateRetriever:
    def __init__(self, train_rows: list[dict[str, Any]], top_per_group: int = 200):
        grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        global_counts: Counter[str] = Counter()
        for row in train_rows:
            ans = normalize_text(str(row.get("answer", "")))
            if not ans:
                continue
            global_counts[ans] += 1
            for key in class_keys(row):
                grouped[key][ans] += 1
        self.grouped = {
            key: [ans for ans, _ in counts.most_common(top_per_group)]
            for key, counts in grouped.items()
        }
        self.global_templates = [ans for ans, _ in global_counts.most_common(top_per_group)]
        vocab_docs = list(dict.fromkeys(self.global_templates))
        self.templates = vocab_docs
        self.idf = self._build_idf(vocab_docs)
        self.template_vecs = {t: self._tfidf(t) for t in vocab_docs}
        self.avgdl = mean([float(len(tokens(t))) for t in vocab_docs]) or 1.0

    def _build_idf(self, docs: list[str]) -> dict[str, float]:
        df: Counter[str] = Counter()
        for doc in docs:
            for tok in set(tokens(doc)):
                df[tok] += 1
        n = max(len(docs), 1)
        return {tok: math.log((1 + n) / (1 + freq)) + 1 for tok, freq in df.items()}

    def _tfidf(self, text: str) -> dict[str, float]:
        counts = Counter(tokens(text))
        vec: dict[str, float] = {}
        for tok, count in counts.items():
            vec[tok] = (1 + math.log(count)) * self.idf.get(tok, 1.0)
        return vec

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(tok, 0.0) for tok, value in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def candidates(self, item: dict[str, Any]) -> list[str]:
        for key in class_keys(item):
            vals = self.grouped.get(key)
            if vals:
                return vals
        return self.global_templates

    def majority(self, item: dict[str, Any]) -> str:
        cands = self.candidates(item)
        return cands[0] if cands else ""

    def retrieve(self, item: dict[str, Any], query: str, top_k: int = 40) -> str:
        cands = self.candidates(item)
        if not cands:
            return ""
        qvec = self._tfidf(query)
        scored = []
        for cand in cands[: max(top_k, len(cands))]:
            scored.append((self._cos(qvec, self.template_vecs.get(cand, self._tfidf(cand))), cand))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else self.majority(item)

    def bm25(self, item: dict[str, Any], query: str, top_k: int = 40, k1: float = 1.5, b: float = 0.75) -> str:
        cands = self.candidates(item)
        if not cands:
            return ""
        q_tokens = set(tokens(query))
        scored = []
        for cand in cands[: max(top_k, len(cands))]:
            doc_tokens = tokens(cand)
            counts = Counter(doc_tokens)
            dl = len(doc_tokens) or 1
            score = 0.0
            for tok in q_tokens:
                tf = counts.get(tok, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(tok, 1.0)
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                score += idf * (tf * (k1 + 1)) / denom
            scored.append((score, cand))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else self.majority(item)


def summarize(rows: list[dict[str, Any]], pred_field: str) -> dict[str, float | int | str]:
    return {
        "method": pred_field,
        "n": len(rows),
        "exact": mean([exact(str(r.get(pred_field, "")), str(r.get("answer", ""))) for r in rows]),
        "token_f1": mean([token_f1(str(r.get(pred_field, "")), str(r.get("answer", ""))) for r in rows]),
    }


def bucket_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("content_type", "unknown")).lower(), str(row.get("answer_type", "unknown")).lower())


def fit_bucket_gate(rows: list[dict[str, Any]], direct_field: str, evidence_field: str, min_count: int) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if direct_field in row and evidence_field in row:
            grouped[bucket_key(row)].append(row)
    policy: dict[tuple[str, str], str] = {}
    for key, group in grouped.items():
        if len(group) < min_count:
            policy[key] = "direct"
            continue
        direct = mean([token_f1(str(r.get(direct_field, "")), str(r.get("answer", ""))) for r in group])
        evidence = mean([token_f1(str(r.get(evidence_field, "")), str(r.get("answer", ""))) for r in group])
        policy[key] = "evidence" if evidence > direct else "direct"
    return policy


def apply_bucket_gate(rows: list[dict[str, Any]], policy: dict[tuple[str, str], str], direct_field: str, evidence_field: str, out_field: str) -> None:
    for row in rows:
        route = policy.get(bucket_key(row), "direct")
        row[out_field] = row.get(evidence_field, "") if route == "evidence" else row.get(direct_field, "")
        row[out_field + "_route"] = route


def strip_thinking(text: str) -> str:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def parse_json_object(text: str) -> dict[str, Any] | None:
    clean = strip_thinking(text)
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()
    match = re.search(r"\{.*\}", clean, flags=re.S)
    if match:
        clean = match.group(0)
    try:
        obj = json.loads(clean)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def final_from_json(text: str) -> str:
    obj = parse_json_object(text)
    if not obj:
        return strip_thinking(text)
    for key in ["final_answer", "answer", "final"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return strip_thinking(text)


def concise_prompt(question: str) -> str:
    return (
        "You are answering a medical visual question. Return only the final answer. "
        "Use the shortest benchmark-style answer possible: yes/no, a number, a modality, "
        "an organ, a location, or a short abnormality name. Do not explain.\n\n"
        f"Question: {question}\nAnswer:"
    )


def class_prompt(question: str, item: dict[str, Any]) -> str:
    return (
        "You are answering a medical VQA benchmark question. Return only the final answer. "
        "Match the expected answer style implied by the metadata and avoid explanations.\n\n"
        f"Answer type: {item.get('answer_type')}\n"
        f"Question category: {item.get('content_type')}\n"
        f"Modality: {item.get('modality')}\n"
        f"Body region: {item.get('location')}\n"
        f"Question: {question}\nAnswer:"
    )


def evidence_prompt(question: str, item: dict[str, Any]) -> str:
    return (
        "Inspect the medical image and answer the question. Return exactly one compact JSON object "
        "with keys: modality, body_region, organ, abnormality, count, position, uncertainty, final_answer. "
        "Use unknown for uncertain visual fields. The final_answer must be concise.\n\n"
        f"Question category: {item.get('content_type')}\n"
        f"Question: {question}"
    )


def build_inputs(processor, img: Path, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(img)}, {"type": "text", "text": prompt}]}]
    try:
        return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")


def generate(model, processor, img: Path, prompt: str, max_new_tokens: int) -> str:
    inputs = build_inputs(processor, img, prompt).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    return strip_thinking(processor.batch_decode(output_ids, skip_special_tokens=True)[0])


def cmd_inspect(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    for split in ["train", "validation", "test"]:
        rows = load_json(data_dir / f"{split}.json")
        print(split, "n", len(rows), "lang", Counter(r.get("q_lang") for r in rows))
        print("answer_type", Counter(str(r.get("answer_type")).lower() for r in rows).most_common())
        print("content_type", Counter(str(r.get("content_type")).lower() for r in rows).most_common())
        missing = sum(1 for r in rows if not image_path(data_dir, r).exists())
        print("missing_images", missing, "unique_answers", len({normalize_text(str(r.get("answer", ""))) for r in rows}))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = split_rows(data_dir, "train", args.lang)
    val = split_rows(data_dir, "validation", args.lang)
    test = split_rows(data_dir, "test", args.lang)
    retriever = TemplateRetriever(train, top_per_group=args.top_per_group)

    qwen_rows: dict[str, dict[str, Any]] = {}
    qwen_paths = [Path(p) for p in args.qwen_jsonl.split(",") if p.strip()]
    for qwen_path in qwen_paths:
        if not qwen_path.exists():
            continue
        with qwen_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qwen_rows[str(row["qid"])] = row

    eval_rows: list[dict[str, Any]] = []
    for item in val + test:
        row = dict(item)
        row["split_eval"] = "validation" if item in val else "test"
        row["majority_by_class"] = retriever.majority(item)
        row["question_only_template"] = retriever.retrieve(item, str(item.get("question", "")), top_k=args.top_k)
        row["question_only_bm25"] = retriever.bm25(item, str(item.get("question", "")), top_k=args.top_k)
        qrow = qwen_rows.get(str(item["qid"]))
        if qrow:
            row["qwen_constrained"] = qrow.get("qwen_constrained_answer", "")
            row["qwen_class"] = qrow.get("qwen_class_answer", "")
            row["qwen_evidence_final"] = qrow.get("qwen_evidence_final", "")
            row["qwen_constrained_template"] = retriever.retrieve(item, f"{item.get('question','')} {row['qwen_constrained']}", top_k=args.top_k)
            row["qwen_class_template"] = retriever.retrieve(item, f"{item.get('question','')} {row['qwen_class']}", top_k=args.top_k)
            row["qwen_evidence_template"] = retriever.retrieve(item, f"{item.get('question','')} {row['qwen_constrained']} {row['qwen_evidence_final']}", top_k=args.top_k)
            row["qwen_constrained_bm25"] = retriever.bm25(item, f"{item.get('question','')} {row['qwen_constrained']}", top_k=args.top_k)
            row["qwen_class_bm25"] = retriever.bm25(item, f"{item.get('question','')} {row['qwen_class']}", top_k=args.top_k)
            row["qwen_evidence_bm25"] = retriever.bm25(item, f"{item.get('question','')} {row['qwen_constrained']} {row['qwen_evidence_final']}", top_k=args.top_k)
        eval_rows.append(row)

    fields = ["majority_by_class", "question_only_template", "question_only_bm25"]
    if qwen_rows:
        fields += [
            "qwen_constrained", "qwen_class", "qwen_evidence_final",
            "qwen_constrained_template", "qwen_class_template", "qwen_evidence_template",
            "qwen_constrained_bm25", "qwen_class_bm25", "qwen_evidence_bm25",
        ]
        val_qwen = [r for r in eval_rows if r["split_eval"] == "validation" and str(r["qid"]) in qwen_rows]
        if val_qwen:
            policy = fit_bucket_gate(
                val_qwen,
                direct_field="qwen_constrained",
                evidence_field="qwen_evidence_template",
                min_count=args.gate_min_count,
            )
            apply_bucket_gate(eval_rows, policy, "qwen_constrained", "qwen_evidence_template", "qwen_bucket_gate")
            fields.append("qwen_bucket_gate")

    summary_rows = []
    for split_name in ["validation", "test"]:
        subset = [r for r in eval_rows if r["split_eval"] == split_name]
        for field in fields:
            metric_subset = subset
            if field.startswith("qwen_"):
                metric_subset = [r for r in subset if str(r["qid"]) in qwen_rows]
            if not metric_subset:
                continue
            item = summarize(metric_subset, field)
            item["split"] = split_name
            item["lang"] = args.lang
            summary_rows.append(item)

    with (out_dir / f"slake_external_eval_{args.lang}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    with (out_dir / f"slake_external_summary_{args.lang}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "lang", "method", "n", "exact", "token_f1"])
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(row)
    return 0


def cmd_qwen(args: argparse.Namespace) -> int:
    if torch is None or AutoProcessor is None:
        raise RuntimeError("torch/transformers are required for qwen subcommand")
    data_dir = Path(args.data_dir)
    rows = split_rows(data_dir, args.split, args.lang)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(str(json.loads(line).get("qid")))
                except Exception:
                    pass

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    mode = "a" if args.resume else "w"
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    with out_path.open(mode, encoding="utf-8") as f:
        for idx, item in enumerate(rows, 1):
            if str(item["qid"]) in done:
                continue
            started = time.time()
            row = {
                "qid": item["qid"],
                "split": args.split,
                "q_lang": item.get("q_lang"),
                "img_name": item.get("img_name"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "answer_type": item.get("answer_type"),
                "content_type": item.get("content_type"),
                "error": None,
            }
            try:
                img = image_path(data_dir, item)
                if not img.exists():
                    raise FileNotFoundError(str(img))
                if "constrained" in modes:
                    raw = generate(model, processor, img, concise_prompt(str(item.get("question", ""))), 48)
                    row["qwen_constrained_raw"] = raw
                    row["qwen_constrained_answer"] = raw
                if "class" in modes:
                    raw = generate(model, processor, img, class_prompt(str(item.get("question", "")), item), 48)
                    row["qwen_class_raw"] = raw
                    row["qwen_class_answer"] = raw
                if "evidence" in modes:
                    raw = generate(model, processor, img, evidence_prompt(str(item.get("question", "")), item), 144)
                    row["qwen_evidence_raw"] = raw
                    row["qwen_evidence_json"] = parse_json_object(raw)
                    row["qwen_evidence_final"] = final_from_json(raw)
            except Exception as exc:
                row["error"] = repr(exc)
            row["elapsed_sec"] = round(time.time() - started, 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            preview = row.get("qwen_constrained_answer") or row.get("qwen_class_answer") or ""
            print(f"[{idx}/{len(rows)}] {item['qid']} {row['elapsed_sec']}s {preview[:60]!r}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--data-dir", default="data/slake")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("eval")
    p.add_argument("--data-dir", default="data/slake")
    p.add_argument("--out-dir", default="outputs/slake/results")
    p.add_argument("--lang", default="en", choices=["en", "zh", "all"])
    p.add_argument("--top-per-group", type=int, default=200)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--gate-min-count", type=int, default=5)
    p.add_argument("--qwen-jsonl", default="")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("qwen")
    p.add_argument("--data-dir", default="data/slake")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--lang", default="en", choices=["en", "zh", "all"])
    p.add_argument("--out", default="outputs/slake/results/slake_qwen_test_en.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--modes", default="constrained,class,evidence")
    p.set_defaults(func=cmd_qwen)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
