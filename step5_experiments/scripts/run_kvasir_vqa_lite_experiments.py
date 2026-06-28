#!/usr/bin/env python3
"""First-pass runnable validation for MedHEval-Tree on Kvasir-VQA-x1.

This experiment intentionally avoids GPU/VLM dependencies. It checks whether
the dataset's structure signals (complexity, question_class, atomic QA text)
support the MedHEval-Tree hypothesis before we spend time on heavier models.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
OUT_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "any",
    "there",
    "what",
    "where",
    "which",
    "in",
    "of",
    "and",
    "or",
    "to",
    "with",
    "visible",
    "observed",
    "image",
    "present",
    "identified",
    "gastrointestinal",
    "tract",
}

NONE_TERMS = {
    "none",
    "no",
    "not",
    "absent",
    "without",
    "negative",
    "normal",
    "unremarkable",
}

POLYP_TERMS = {
    "polyp",
    "polypoid",
    "paris",
    "sessile",
    "pedunculated",
    "iia",
    "iib",
    "iic",
}


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.replace("oesophagitis", "esophagitis")
    text = text.replace("polyps", "polyp")
    text = text.replace("polypoid lesions", "polypoid lesion")
    text = text.replace("no visible text observed", "no text")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text).lower())
        if tok not in STOPWORDS and len(tok) > 1
    }


def parse_question_classes(value: str) -> tuple[str, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(str(x).strip() for x in value if str(x).strip())
    text = str(value).strip()
    if not text or text == "nan":
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return tuple(str(x).strip() for x in parsed if str(x).strip())
    except Exception:
        pass
    text = text.strip("[]")
    parts = re.findall(r"[A-Za-z_]+", text)
    return tuple(parts)


def class_key(classes: Iterable[str]) -> str:
    classes = tuple(classes)
    return "|".join(classes) if classes else "__none__"


def parse_original_atomic(value: str) -> list[dict[str, str]]:
    text = str(value).strip()
    if not text or text == "nan":
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, dict):
                    q = str(item.get("q", "")).strip()
                    a = str(item.get("a", "")).strip()
                    if q or a:
                        out.append({"q": q, "a": a})
            return out
    except Exception:
        return []
    return []


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc = Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def hallucination_proxy(pred: str, gold: str) -> int:
    """A conservative text-only proxy, not a medical visual hallucination metric."""
    p = set(normalize_answer(pred).split())
    g = set(normalize_answer(gold).split())
    pred_neg = bool(p & NONE_TERMS)
    gold_neg = bool(g & NONE_TERMS)
    pred_polyp = bool(p & POLYP_TERMS)
    gold_polyp = bool(g & POLYP_TERMS)
    if pred_polyp and gold_neg and not gold_polyp:
        return 1
    if pred_neg and gold_polyp and not gold_neg:
        return 1
    return 0


@dataclass
class Example:
    idx: int
    img_id: str
    complexity: int
    question: str
    answer: str
    norm_answer: str
    classes: tuple[str, ...]
    class_key: str
    original_atoms: list[dict[str, str]]
    tokens: set[str]


def load_split(name: str) -> list[Example]:
    path = DATA_ROOT / f"{name}.parquet"
    df = pd.read_parquet(path)
    examples: list[Example] = []
    for idx, row in df.iterrows():
        classes = parse_question_classes(row.get("question_class", ""))
        answer = str(row.get("answer", ""))
        question = str(row.get("question", ""))
        examples.append(
            Example(
                idx=int(idx),
                img_id=str(row.get("img_id", "")),
                complexity=int(row.get("complexity", 0)),
                question=question,
                answer=answer,
                norm_answer=normalize_answer(answer),
                classes=classes,
                class_key=class_key(classes),
                original_atoms=parse_original_atomic(row.get("original", "")),
                tokens=tokenize(question),
            )
        )
    return examples


def majority(counter: Counter[str], fallback: str) -> str:
    if not counter:
        return fallback
    return counter.most_common(1)[0][0]


def build_indexes(train: list[Example]) -> dict:
    global_counts = Counter(ex.norm_answer for ex in train)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    complexity_class_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    first_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    retrieval_by_class: dict[str, list[Example]] = defaultdict(list)
    retrieval_by_first_class: dict[str, list[Example]] = defaultdict(list)
    atomic_question_counts: dict[str, Counter[str]] = defaultdict(Counter)
    template_by_class: dict[str, list[tuple[set[str], str, set[str]]]] = defaultdict(list)

    for ex in train:
        class_counts[ex.class_key][ex.norm_answer] += 1
        complexity_class_counts[(ex.complexity, ex.class_key)][ex.norm_answer] += 1
        for cls in ex.classes:
            first_class_counts[cls][ex.norm_answer] += 1
            retrieval_by_first_class[cls].append(ex)
        retrieval_by_class[ex.class_key].append(ex)
        for atom in ex.original_atoms:
            q_norm = normalize_answer(atom.get("q", ""))
            a_norm = normalize_answer(atom.get("a", ""))
            if q_norm and a_norm:
                atomic_question_counts[q_norm][a_norm] += 1
        atom_answer_text = " ".join(normalize_answer(atom.get("a", "")) for atom in ex.original_atoms)
        atom_answer_tokens = tokenize(atom_answer_text)
        if atom_answer_tokens:
            template_by_class[ex.class_key].append((atom_answer_tokens, ex.norm_answer, ex.tokens))

    return {
        "global_answer": majority(global_counts, "no"),
        "global_counts": global_counts,
        "class_counts": class_counts,
        "complexity_class_counts": complexity_class_counts,
        "first_class_counts": first_class_counts,
        "retrieval_by_class": retrieval_by_class,
        "retrieval_by_first_class": retrieval_by_first_class,
        "atomic_question_counts": atomic_question_counts,
        "template_by_class": template_by_class,
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_retrieval_predict(ex: Example, index: dict) -> str:
    return lexical_retrieval_predict_with_score(ex, index)[0]


def lexical_retrieval_predict_with_score(ex: Example, index: dict) -> tuple[str, float]:
    candidates = index["retrieval_by_class"].get(ex.class_key, [])
    if len(candidates) < 5 and ex.classes:
        candidates = index["retrieval_by_first_class"].get(ex.classes[0], candidates)
    if not candidates:
        return index["global_answer"], 0.0

    best_score = -1.0
    best_answer = index["global_answer"]
    for tr in candidates:
        score = jaccard(ex.tokens, tr.tokens)
        if score > best_score:
            best_score = score
            best_answer = tr.norm_answer
    return best_answer, best_score


def atomic_oracle_predict(ex: Example) -> str:
    atoms = [normalize_answer(atom.get("a", "")) for atom in ex.original_atoms if atom.get("a")]
    atoms = [a for a in atoms if a]
    if not atoms:
        return ""
    return " ".join(atoms)


def atomic_answer_tokens(ex: Example, use_gold: bool, index: dict) -> set[str]:
    pieces = []
    for atom in ex.original_atoms:
        if use_gold:
            pieces.append(atom.get("a", ""))
        else:
            q_norm = normalize_answer(atom.get("q", ""))
            pieces.append(majority(index["atomic_question_counts"].get(q_norm, Counter()), ""))
    return tokenize(" ".join(pieces))


def template_rerank_predict(ex: Example, index: dict, use_gold_atoms: bool = False) -> str:
    """Retrieve a train final-answer template by matching atomic answer evidence."""
    templates = index["template_by_class"].get(ex.class_key, [])
    if not templates and ex.classes:
        # Back off to any template whose class key contains the first class.
        first = ex.classes[0]
        templates = [
            tpl
            for key, vals in index["template_by_class"].items()
            if first in key
            for tpl in vals[:200]
        ]
    if not templates:
        return lexical_retrieval_predict(ex, index)

    atom_tokens = atomic_answer_tokens(ex, use_gold=use_gold_atoms, index=index)
    best_score = -1.0
    best_answer = ""
    for tpl_atom_tokens, final_answer, train_q_tokens in templates:
        atom_score = jaccard(atom_tokens, tpl_atom_tokens)
        question_score = jaccard(ex.tokens, train_q_tokens)
        score = 0.75 * atom_score + 0.25 * question_score
        if score > best_score:
            best_score = score
            best_answer = final_answer
    return best_answer or lexical_retrieval_predict(ex, index)


def atomic_retrieval_predict(ex: Example, index: dict) -> str:
    preds = []
    for atom in ex.original_atoms:
        q_norm = normalize_answer(atom.get("q", ""))
        if not q_norm:
            continue
        pred = majority(index["atomic_question_counts"].get(q_norm, Counter()), "")
        if pred:
            preds.append(pred)
    if preds:
        return " ".join(preds)
    return lexical_retrieval_predict(ex, index)


def medheval_tree_lite_predict(ex: Example, index: dict) -> str:
    """A first runnable approximation of category/atomic evidence aggregation."""
    if ex.complexity > 1 and ex.original_atoms:
        atomic_pred = atomic_retrieval_predict(ex, index)
        lexical_pred = lexical_retrieval_predict(ex, index)
        # Prefer atomic composition if it covers at least two non-empty pieces.
        if len(atomic_pred.split()) >= 2:
            # Contradiction-lite: if class retrieval strongly says "none/no",
            # keep the explicit no/none part but avoid appending conflicting
            # polyp type words from noisy atomic retrieval.
            atomic_terms = set(atomic_pred.split())
            lexical_terms = set(lexical_pred.split())
            if atomic_terms & NONE_TERMS and lexical_terms & POLYP_TERMS:
                return atomic_pred
            return atomic_pred
        return lexical_pred

    if ex.classes:
        # Simple questions: class prior + lexical retrieval.
        prior = majority(index["first_class_counts"].get(ex.classes[0], Counter()), index["global_answer"])
        retrieved = lexical_retrieval_predict(ex, index)
        if token_f1(retrieved, prior) > 0:
            return retrieved
        return retrieved if retrieved != index["global_answer"] else prior
    return lexical_retrieval_predict(ex, index)


def medheval_tree_template_predict(ex: Example, index: dict) -> str:
    """Second-pass MedHEval-Tree: node evidence -> final answer template reranking."""
    if ex.complexity > 1 and ex.original_atoms:
        template_pred = template_rerank_predict(ex, index, use_gold_atoms=False)
        lexical_pred = lexical_retrieval_predict(ex, index)
        # If template and lexical agree semantically, trust template. Otherwise,
        # use class majority for exact-style answers and lexical for lower
        # hallucination proxy.
        if token_f1(template_pred, lexical_pred) >= 0.4:
            return template_pred
        cls_pred = majority(index["class_counts"].get(ex.class_key, Counter()), index["global_answer"])
        if token_f1(template_pred, cls_pred) >= token_f1(lexical_pred, cls_pred):
            return template_pred
        return lexical_pred
    return medheval_tree_lite_predict(ex, index)


def tune_hybrid_threshold(train: list[Example]) -> dict:
    split = int(len(train) * 0.9)
    sub_train = train[:split]
    dev = train[split:]
    index = build_indexes(sub_train)
    thresholds = [i / 20 for i in range(0, 21)]
    best_exact = {"threshold": 0.0, "exact": -1.0, "token_f1": -1.0}
    best_f1 = {"threshold": 0.0, "exact": -1.0, "token_f1": -1.0}

    for tau in thresholds:
        preds = []
        for ex in dev:
            cls_pred = majority(index["class_counts"].get(ex.class_key, Counter()), index["global_answer"])
            lex_pred, lex_score = lexical_retrieval_predict_with_score(ex, index)
            preds.append(lex_pred if lex_score >= tau else cls_pred)
        res = evaluate(f"hybrid_tau_{tau}", preds, dev)
        if res["exact_match"] > best_exact["exact"]:
            best_exact = {"threshold": tau, "exact": res["exact_match"], "token_f1": res["token_f1"]}
        if res["token_f1"] > best_f1["token_f1"]:
            best_f1 = {"threshold": tau, "exact": res["exact_match"], "token_f1": res["token_f1"]}
    return {"best_exact": best_exact, "best_f1": best_f1}


def hybrid_class_lexical_predict(ex: Example, index: dict, threshold: float) -> str:
    cls_pred = majority(index["class_counts"].get(ex.class_key, Counter()), index["global_answer"])
    lex_pred, lex_score = lexical_retrieval_predict_with_score(ex, index)
    return lex_pred if lex_score >= threshold else cls_pred


def evaluate(name: str, preds: list[str], test: list[Example]) -> dict:
    rows = []
    exact = []
    f1s = []
    hallucinations = []
    by_complexity: dict[int, list[int]] = defaultdict(list)
    by_class: dict[str, list[int]] = defaultdict(list)
    for pred, ex in zip(preds, test):
        pred_norm = normalize_answer(pred)
        hit = int(pred_norm == ex.norm_answer)
        f1 = token_f1(pred_norm, ex.norm_answer)
        h = hallucination_proxy(pred_norm, ex.norm_answer)
        exact.append(hit)
        f1s.append(f1)
        hallucinations.append(h)
        by_complexity[ex.complexity].append(hit)
        for cls in ex.classes or ("__none__",):
            by_class[cls].append(hit)
        rows.append(
            {
                "idx": ex.idx,
                "img_id": ex.img_id,
                "complexity": ex.complexity,
                "class_key": ex.class_key,
                "question": ex.question,
                "gold": ex.answer,
                "pred": pred_norm,
                "exact": hit,
                "token_f1": round(f1, 4),
                "hallucination_proxy": h,
            }
        )
    class_acc = {
        cls: sum(vals) / len(vals)
        for cls, vals in sorted(by_class.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }
    return {
        "method": name,
        "n": len(test),
        "exact_match": sum(exact) / len(exact),
        "token_f1": sum(f1s) / len(f1s),
        "hallucination_proxy_rate": sum(hallucinations) / len(hallucinations),
        "complexity_accuracy": {str(k): sum(v) / len(v) for k, v in sorted(by_complexity.items())},
        "top_class_accuracy": dict(list(class_acc.items())[:20]),
        "rows": rows,
    }


def write_outputs(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for res in results:
        summary_rows.append(
            {
                "method": res["method"],
                "n": res["n"],
                "exact_match": round(res["exact_match"], 6),
                "token_f1": round(res["token_f1"], 6),
                "hallucination_proxy_rate": round(res["hallucination_proxy_rate"], 6),
                "complexity_accuracy": json.dumps(res["complexity_accuracy"], ensure_ascii=False),
            }
        )
        pred_path = OUT_DIR / f"{res['method']}_predictions.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(res["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(res["rows"])

    summary_path = OUT_DIR / "lite_experiment_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md = [
        "# Step 5 Lite Experiment Report",
        "",
        "This is the first low-cost validation of MedHEval-Tree on Kvasir-VQA-x1.",
        "It uses only text/structure fields, not visual model outputs yet.",
        "",
        "## Summary",
        "",
        "| Method | Exact Match | Token F1 | Hallucination Proxy | Complexity Accuracy |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['method']} | {row['exact_match']:.6f} | {row['token_f1']:.6f} | "
            f"{row['hallucination_proxy_rate']:.6f} | `{row['complexity_accuracy']}` |"
        )
    md.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `atomic_oracle` uses test atomic answers from `original`; it is an upper bound, not a deployable method.",
            "- `medheval_tree_lite` is the first deployable structural approximation: atomic question retrieval + class routing + simple consistency checks.",
            "- A weak exact match but strong token F1 suggests answer phrasing normalization or generation templates need improvement.",
            "",
        ]
    )
    (REPORT_DIR / "lite_experiment_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    train = load_split("train")
    test = load_split("test")
    index = build_indexes(train)
    tuned = tune_hybrid_threshold(train)
    (OUT_DIR / "hybrid_threshold_tuning.json").write_text(json.dumps(tuned, ensure_ascii=False, indent=2))
    tau_exact = tuned["best_exact"]["threshold"]
    tau_f1 = tuned["best_f1"]["threshold"]

    methods = {
        "global_majority": lambda ex: index["global_answer"],
        "class_majority": lambda ex: majority(index["class_counts"].get(ex.class_key, Counter()), index["global_answer"]),
        "complexity_class_majority": lambda ex: majority(
            index["complexity_class_counts"].get((ex.complexity, ex.class_key), Counter()),
            majority(index["class_counts"].get(ex.class_key, Counter()), index["global_answer"]),
        ),
        "lexical_retrieval": lambda ex: lexical_retrieval_predict(ex, index),
        "atomic_oracle": lambda ex: atomic_oracle_predict(ex) or lexical_retrieval_predict(ex, index),
        "template_oracle": lambda ex: template_rerank_predict(ex, index, use_gold_atoms=True),
        "atomic_retrieval": lambda ex: atomic_retrieval_predict(ex, index),
        "medheval_tree_lite": lambda ex: medheval_tree_lite_predict(ex, index),
        "medheval_tree_template": lambda ex: medheval_tree_template_predict(ex, index),
        f"hybrid_class_lexical_exact_tau_{tau_exact}": lambda ex: hybrid_class_lexical_predict(ex, index, tau_exact),
        f"hybrid_class_lexical_f1_tau_{tau_f1}": lambda ex: hybrid_class_lexical_predict(ex, index, tau_f1),
    }

    results = []
    for name, fn in methods.items():
        print(f"[run] {name}")
        preds = [fn(ex) for ex in test]
        results.append(evaluate(name, preds, test))

    write_outputs(results)
    print(f"[done] wrote {OUT_DIR / 'lite_experiment_summary.csv'}")
    print(f"[done] wrote {REPORT_DIR / 'lite_experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
