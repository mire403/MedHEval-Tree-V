#!/usr/bin/env python3
"""Reliability-calibrated evidence-tree routing for MedHEval-Tree-V.

This cache-only experiment upgrades the original bucket lookup gate into a
sample-adaptive route selector. It uses the frozen Qwen evidence cache and the
stored direct/evidence template predictions; no MLLM inference is performed.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "step5_experiments" / "results"
REPORT_DIR = ROOT / "step5_experiments" / "reports"
PRED_PATH = RESULT_DIR / "qwen_full_tfidf_fast_predictions.csv"
EVIDENCE_PATH = RESULT_DIR / "qwen_full_test_outputs.jsonl"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "as", "by",
    "this", "that", "there", "it", "its", "any", "from", "into",
}
UNKNOWN = {"", "unknown", "uncertain", "not visible", "n/a", "none", "null"}


def norm(text: object) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: object) -> list[str]:
    return [t for t in norm(text).split() if t and t not in STOPWORDS]


def token_f1(pred: str, gold: str) -> float:
    pt = tokens(pred)
    gt = tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc: dict[str, int] = defaultdict(int)
    gc: dict[str, int] = defaultdict(int)
    for t in pt:
        pc[t] += 1
    for t in gt:
        gc[t] += 1
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact(pred: str, gold: str) -> float:
    return 1.0 if norm(pred) == norm(gold) else 0.0


def lexical_drift(pred: str, gold: str, threshold: float = 0.5) -> float:
    pt = [t for t in tokens(pred) if t not in STOPWORDS]
    gt = {t for t in tokens(gold) if t not in STOPWORDS}
    if not pt:
        return 0.0
    unseen = sum(1 for t in pt if t not in gt)
    return 1.0 if unseen / len(pt) > threshold else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def primary_class(row: dict[str, str]) -> str:
    try:
        value = json.loads(row["question_class"])
        if isinstance(value, list) and value:
            return str(value[0])
    except Exception:
        pass
    return row.get("question_class", "unknown") or "unknown"


def load_evidence() -> dict[str, dict[str, object]]:
    by_id = {}
    with EVIDENCE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ev = row.get("qwen_evidence")
            by_id[str(row["id"])] = ev if isinstance(ev, dict) else {}
    return by_id


def load_rows() -> list[dict[str, object]]:
    evidence_by_id = load_evidence()
    rows: list[dict[str, object]] = []
    with PRED_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            item: dict[str, object] = dict(row)
            item["_idx"] = i
            item["_complexity"] = str(row["complexity"])
            item["_primary_class"] = primary_class(row)
            item["_direct_pred"] = row["tfidf_direct_template_prediction"]
            item["_evidence_pred"] = row["tfidf_evidence_rerank_prediction"]
            item["_direct_f1"] = float(row["tfidf_direct_template_token_f1"])
            item["_evidence_f1"] = float(row["tfidf_evidence_rerank_token_f1"])
            item["_direct_exact"] = float(row["tfidf_direct_template_exact"])
            item["_evidence_exact"] = float(row["tfidf_evidence_rerank_exact"])
            item["_evidence"] = evidence_by_id.get(str(row["id"]), {})
            rows.append(item)
    return rows


def bucket_key(row: dict[str, object]) -> tuple[str, str]:
    return (str(row["_complexity"]), str(row["_primary_class"]))


def fit_bucket_stats(rows: list[dict[str, object]], indices: list[int], min_count: int = 8) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx in indices:
        grouped[bucket_key(rows[idx])].append(idx)
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        direct = mean([float(rows[i]["_direct_f1"]) for i in group])
        evidence = mean([float(rows[i]["_evidence_f1"]) for i in group])
        wins = mean([1.0 if float(rows[i]["_evidence_f1"]) > float(rows[i]["_direct_f1"]) else 0.0 for i in group])
        usable = float(len(group) >= min_count)
        stats[key] = {
            "n": float(len(group)),
            "direct": direct,
            "evidence": evidence,
            "delta": evidence - direct,
            "win_rate": wins,
            "usable": usable,
            "route": 1.0 if usable and evidence > direct else 0.0,
        }
    return stats


def bucket_prior(row: dict[str, object], stats: dict[tuple[str, str], dict[str, float]]) -> dict[str, float]:
    s = stats.get(bucket_key(row))
    if not s:
        return {"bucket_delta": 0.0, "bucket_win_rate": 0.5, "bucket_route": 0.0, "bucket_seen": 0.0}
    n = s["n"]
    # Smoothed prior keeps small buckets from dominating the per-sample gate.
    return {
        "bucket_delta": s["delta"],
        "bucket_win_rate": (s["win_rate"] * n + 0.5 * 8.0) / (n + 8.0),
        "bucket_route": s["route"],
        "bucket_seen": 1.0,
    }


def field_value(row: dict[str, object], key: str) -> str:
    ev = row.get("_evidence")
    if isinstance(ev, dict):
        return norm(ev.get(key, ""))
    return ""


def field_completeness(row: dict[str, object]) -> float:
    keys = [
        "lesion_presence", "lesion_type", "lesion_count", "location",
        "instrument_presence", "text_overlay_presence", "abnormality_presence",
    ]
    vals = [field_value(row, k) for k in keys]
    return sum(1 for v in vals if v not in UNKNOWN) / len(keys)


def uncertainty_score(row: dict[str, object]) -> float:
    u = field_value(row, "uncertainty")
    if u in {"low", "none"}:
        return 0.0
    if u in {"moderate", "medium"}:
        return 0.5
    if u in {"high", "uncertain"}:
        return 1.0
    return 0.35


def class_field_weights(primary: str) -> dict[str, float]:
    cls = norm(primary)
    weights = {
        "presence": 0.9,
        "type": 0.9,
        "count": 0.8,
        "location": 0.6,
        "instrument": 0.5,
        "text": 0.4,
        "abnormality": 0.7,
    }
    if "count" in cls or "number" in cls:
        weights["count"] += 1.0
    if "type" in cls or "diagnosis" in cls or "polyp" in cls:
        weights["type"] += 0.8
        weights["abnormality"] += 0.4
    if "location" in cls or "where" in cls:
        weights["location"] += 1.0
    if "instrument" in cls or "tool" in cls:
        weights["instrument"] += 1.0
    if "text" in cls or "overlay" in cls:
        weights["text"] += 1.0
    if "removal" in cls or "status" in cls:
        weights["presence"] += 0.5
        weights["instrument"] += 0.3
    return weights


def any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {norm(text)} "
    return any(f" {p} " in padded or p in padded for p in phrases)


def overlap_score(value: str, candidate: str) -> float:
    vt = set(tokens(value))
    ct = set(tokens(candidate))
    if not vt:
        return 0.0
    return len(vt & ct) / len(vt)


def evidence_tree_compatibility(candidate: str, row: dict[str, object]) -> tuple[float, float]:
    """Return (compatibility, contradiction) for candidate against evidence tree."""
    cand = norm(candidate)
    weights = class_field_weights(str(row["_primary_class"]))
    score = 0.0
    denom = 0.0
    contradiction = 0.0

    lesion = field_value(row, "lesion_presence")
    abnormal = field_value(row, "abnormality_presence")
    count = field_value(row, "lesion_count")
    lesion_type = field_value(row, "lesion_type")
    location = field_value(row, "location")
    instrument = field_value(row, "instrument_presence")
    text_overlay = field_value(row, "text_overlay_presence")

    no_lesion = any_phrase(cand, ("no lesion", "no polyp", "no abnormality", "no polypoid lesion"))
    yes_lesion = any_phrase(cand, ("lesion", "polyp", "polypoid", "abnormality", "adenoma", "adenomatous"))
    if lesion not in UNKNOWN:
        w = weights["presence"]
        denom += w
        if lesion in {"yes", "present", "true"}:
            score += w * (1.0 if yes_lesion and not no_lesion else -1.0 if no_lesion else 0.0)
            contradiction += 1.0 if no_lesion else 0.0
        elif lesion in {"no", "absent", "false"}:
            score += w * (1.0 if no_lesion else -0.5 if yes_lesion else 0.0)
            contradiction += 1.0 if yes_lesion and not no_lesion else 0.0

    if abnormal not in UNKNOWN:
        w = weights["abnormality"]
        denom += w
        no_abn = any_phrase(cand, ("no abnormality", "normal", "no abnormalities"))
        yes_abn = any_phrase(cand, ("abnormal", "lesion", "polyp", "adenoma"))
        if abnormal in {"yes", "present", "true"}:
            score += w * (1.0 if yes_abn and not no_abn else -1.0 if no_abn else 0.0)
            contradiction += 1.0 if no_abn else 0.0
        elif abnormal in {"no", "absent", "false"}:
            score += w * (1.0 if no_abn else -0.5 if yes_abn else 0.0)
            contradiction += 1.0 if yes_abn and not no_abn else 0.0

    if lesion_type not in UNKNOWN:
        w = weights["type"]
        denom += w
        score += w * overlap_score(lesion_type, cand)

    if count not in UNKNOWN:
        w = weights["count"]
        denom += w
        single_cand = any_phrase(cand, ("single", "one", "1 polyp", "one polyp"))
        multi_cand = any_phrase(cand, ("multiple", "several", "many", "two", "2 polyps", "three"))
        if count in {"1", "one", "single"}:
            score += w * (1.0 if single_cand else -1.0 if multi_cand else 0.0)
            contradiction += 1.0 if multi_cand else 0.0
        elif count in {"multiple", "several", "many", "2", "two", "3", "three"}:
            score += w * (1.0 if multi_cand else -0.8 if single_cand else 0.0)
            contradiction += 1.0 if single_cand else 0.0

    if location not in UNKNOWN:
        w = weights["location"]
        denom += w
        score += w * overlap_score(location, cand)

    if instrument not in UNKNOWN:
        w = weights["instrument"]
        denom += w
        instrument_word = any_phrase(cand, ("instrument", "tool", "scope", "forceps", "snare", "colonoscope"))
        no_inst = any_phrase(cand, ("no instrument", "without instrument", "no tool"))
        if instrument in {"yes", "present", "true"}:
            score += w * (1.0 if instrument_word and not no_inst else -0.8 if no_inst else 0.0)
            contradiction += 1.0 if no_inst else 0.0
        elif instrument in {"no", "absent", "false"}:
            score += w * (1.0 if no_inst else -0.5 if instrument_word else 0.0)
            contradiction += 1.0 if instrument_word and not no_inst else 0.0

    if text_overlay not in UNKNOWN:
        w = weights["text"]
        denom += w
        text_word = any_phrase(cand, ("text", "overlay", "label", "caption"))
        no_text = any_phrase(cand, ("no text", "without text", "no overlay"))
        if text_overlay in {"yes", "present", "true"}:
            score += w * (1.0 if text_word and not no_text else -0.8 if no_text else 0.0)
            contradiction += 1.0 if no_text else 0.0
        elif text_overlay in {"no", "absent", "false"}:
            score += w * (1.0 if no_text else -0.4 if text_word else 0.0)
            contradiction += 1.0 if text_word and not no_text else 0.0

    if denom == 0:
        return 0.0, contradiction
    reliability = field_completeness(row) * (1.0 - 0.5 * uncertainty_score(row))
    return reliability * (score / denom), contradiction / max(1.0, denom)


def jaccard(a: str, b: str) -> float:
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def route_label(row: dict[str, object]) -> float:
    return 1.0 if float(row["_evidence_f1"]) > float(row["_direct_f1"]) else 0.0


def build_feature_space(rows: list[dict[str, object]], indices: list[int]) -> dict[str, int]:
    names = [
        "bias", "complexity", "direct_len", "evidence_len", "len_delta",
        "direct_evidence_jaccard", "question_direct_jaccard", "question_evidence_jaccard",
        "tree_direct", "tree_evidence", "tree_delta", "tree_abs_delta",
        "direct_contra", "evidence_contra", "contra_delta",
        "evidence_completeness", "uncertainty", "evidence_quality",
        "bucket_delta", "bucket_win_rate", "bucket_route", "bucket_seen",
    ]
    classes = sorted({str(rows[i]["_primary_class"]) for i in indices})
    for cls in classes:
        names.append(f"class={cls}")
    return {name: i for i, name in enumerate(names)}


def featurize(
    row: dict[str, object],
    space: dict[str, int],
    stats: dict[tuple[str, str], dict[str, float]],
    *,
    use_tree: bool = True,
    use_bucket: bool = True,
) -> list[float]:
    vec = [0.0] * len(space)
    direct = str(row["_direct_pred"])
    evidence = str(row["_evidence_pred"])
    question = str(row["question"])
    direct_len = len(tokens(direct))
    evidence_len = len(tokens(evidence))
    tree_direct, direct_contra = evidence_tree_compatibility(direct, row) if use_tree else (0.0, 0.0)
    tree_evidence, evidence_contra = evidence_tree_compatibility(evidence, row) if use_tree else (0.0, 0.0)
    completeness = field_completeness(row)
    uncertainty = uncertainty_score(row)
    prior = bucket_prior(row, stats) if use_bucket else {
        "bucket_delta": 0.0,
        "bucket_win_rate": 0.5,
        "bucket_route": 0.0,
        "bucket_seen": 0.0,
    }
    values = {
        "bias": 1.0,
        "complexity": (float(row["_complexity"]) - 2.0),
        "direct_len": min(direct_len, 24) / 24.0,
        "evidence_len": min(evidence_len, 24) / 24.0,
        "len_delta": max(min(evidence_len - direct_len, 24), -24) / 24.0,
        "direct_evidence_jaccard": jaccard(direct, evidence),
        "question_direct_jaccard": jaccard(question, direct),
        "question_evidence_jaccard": jaccard(question, evidence),
        "tree_direct": tree_direct,
        "tree_evidence": tree_evidence,
        "tree_delta": tree_evidence - tree_direct,
        "tree_abs_delta": abs(tree_evidence - tree_direct),
        "direct_contra": direct_contra,
        "evidence_contra": evidence_contra,
        "contra_delta": direct_contra - evidence_contra,
        "evidence_completeness": completeness,
        "uncertainty": uncertainty,
        "evidence_quality": completeness * (1.0 - uncertainty),
        f"class={row['_primary_class']}": 1.0,
        **prior,
    }
    for key, value in values.items():
        if key in space:
            vec[space[key]] = float(value)
    return vec


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def train_logistic(xs: list[list[float]], ys: list[float], epochs: int = 450, lr: float = 0.05, l2: float = 0.02) -> list[float]:
    if not xs:
        return []
    w = [0.0] * len(xs[0])
    n = len(xs)
    for _ in range(epochs):
        grad = [0.0] * len(w)
        for x, y in zip(xs, ys):
            p = sigmoid(sum(wi * xi for wi, xi in zip(w, x)))
            err = p - y
            for j, xj in enumerate(x):
                grad[j] += err * xj
        for j in range(len(w)):
            reg = 0.0 if j == 0 else l2 * w[j]
            w[j] -= lr * (grad[j] / n + reg)
    return w


def dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def evaluate_fixed_route(rows: list[dict[str, object]], indices: list[int], route: str) -> dict[str, float]:
    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    pred_key = "_evidence_pred" if route == "evidence" else "_direct_pred"
    f1_key = "_evidence_f1" if route == "evidence" else "_direct_f1"
    exact_key = "_evidence_exact" if route == "evidence" else "_direct_exact"
    for idx in indices:
        row = rows[idx]
        f1s.append(float(row[f1_key]))
        exacts.append(float(row[exact_key]))
        drifts.append(lexical_drift(str(row[pred_key]), str(row["gold"])))
    return {"n": len(indices), "exact": mean(exacts), "token_f1": mean(f1s), "lexical_drift": mean(drifts)}


def evaluate_bucket_gate(rows: list[dict[str, object]], train_idx: list[int], eval_idx: list[int], min_count: int = 8) -> dict[str, object]:
    stats = fit_bucket_stats(rows, train_idx, min_count=min_count)
    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    routed = 0
    for idx in eval_idx:
        row = rows[idx]
        route_evidence = bucket_prior(row, stats)["bucket_route"] >= 0.5
        if route_evidence:
            routed += 1
            pred = str(row["_evidence_pred"])
            f1s.append(float(row["_evidence_f1"]))
            exacts.append(float(row["_evidence_exact"]))
        else:
            pred = str(row["_direct_pred"])
            f1s.append(float(row["_direct_f1"]))
            exacts.append(float(row["_direct_exact"]))
        drifts.append(lexical_drift(pred, str(row["gold"])))
    return {
        "method": "bucket_gate",
        "n": len(eval_idx),
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "evidence_route_rate": routed / len(eval_idx),
    }


def eval_rcetr(
    rows: list[dict[str, object]],
    train_idx: list[int],
    eval_idx: list[int],
    *,
    method: str = "rcetr_v2",
    use_tree: bool = True,
    use_bucket: bool = True,
) -> dict[str, object]:
    stats = fit_bucket_stats(rows, train_idx, min_count=8)
    space = build_feature_space(rows, train_idx)
    xs = [featurize(rows[i], space, stats, use_tree=use_tree, use_bucket=use_bucket) for i in train_idx]
    ys = [route_label(rows[i]) for i in train_idx]
    w = train_logistic(xs, ys)

    train_probs = [sigmoid(dot(w, featurize(rows[i], space, stats, use_tree=use_tree, use_bucket=use_bucket))) for i in train_idx]
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best_threshold = 0.5
    best_dev_f1 = -1.0
    for threshold in thresholds:
        f1s = []
        for idx, prob in zip(train_idx, train_probs):
            row = rows[idx]
            f1s.append(float(row["_evidence_f1"]) if prob >= threshold else float(row["_direct_f1"]))
        score = mean(f1s)
        if score > best_dev_f1:
            best_dev_f1 = score
            best_threshold = threshold

    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    routed = 0
    probs: list[float] = []
    for idx in eval_idx:
        row = rows[idx]
        prob = sigmoid(dot(w, featurize(row, space, stats, use_tree=use_tree, use_bucket=use_bucket)))
        probs.append(prob)
        if prob >= best_threshold:
            routed += 1
            pred = str(row["_evidence_pred"])
            f1s.append(float(row["_evidence_f1"]))
            exacts.append(float(row["_evidence_exact"]))
        else:
            pred = str(row["_direct_pred"])
            f1s.append(float(row["_direct_f1"]))
            exacts.append(float(row["_direct_exact"]))
        drifts.append(lexical_drift(pred, str(row["gold"])))
    return {
        "method": method,
        "n": len(eval_idx),
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "evidence_route_rate": routed / len(eval_idx),
        "threshold": best_threshold,
        "dev_token_f1": best_dev_f1,
        "feature_count": len(space),
        "mean_route_probability": mean(probs),
    }


def run_random(rows: list[dict[str, object]], dev_size: int, seeds: int = 10) -> list[dict[str, object]]:
    all_idx = list(range(len(rows)))
    out: list[dict[str, object]] = []
    for seed in range(seeds):
        rng = random.Random(9000 + seed)
        perm = all_idx[:]
        rng.shuffle(perm)
        dev = perm[:dev_size]
        eval_idx = perm[dev_size:]
        result = eval_rcetr(rows, dev, eval_idx)
        result.update({"seed": seed, "dev_size": dev_size})
        out.append(result)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(f, "")) for f in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_idx = list(range(529))
    eval_idx = list(range(529, len(rows)))

    direct = evaluate_fixed_route(rows, eval_idx, "direct")
    direct.update({"method": "direct_template", "evidence_route_rate": 0.0})
    evidence = evaluate_fixed_route(rows, eval_idx, "evidence")
    evidence.update({"method": "uniform_evidence_rerank", "evidence_route_rate": 1.0})
    bucket = evaluate_bucket_gate(rows, train_idx, eval_idx)
    rcetr_no_tree = eval_rcetr(rows, train_idx, eval_idx, method="rcetr_no_tree", use_tree=False, use_bucket=True)
    rcetr_no_bucket = eval_rcetr(rows, train_idx, eval_idx, method="rcetr_no_bucket", use_tree=True, use_bucket=False)
    rcetr = eval_rcetr(rows, train_idx, eval_idx)
    main_rows = [direct, evidence, bucket, rcetr_no_tree, rcetr_no_bucket, rcetr]

    random_raw = run_random(rows, 529, seeds=10) + run_random(rows, 1000, seeds=10)
    summary: list[dict[str, object]] = []
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in random_raw:
        grouped[int(row["dev_size"])].append(row)
    for dev_size, group in sorted(grouped.items()):
        summary.append({
            "dev_size": dev_size,
            "runs": len(group),
            "token_f1_mean": mean([float(g["token_f1"]) for g in group]),
            "token_f1_std": stdev([float(g["token_f1"]) for g in group]),
            "exact_mean": mean([float(g["exact"]) for g in group]),
            "lexical_drift_mean": mean([float(g["lexical_drift"]) for g in group]),
            "evidence_route_rate_mean": mean([float(g["evidence_route_rate"]) for g in group]),
            "threshold_mean": mean([float(g["threshold"]) for g in group]),
        })

    write_csv(RESULT_DIR / "rcetr_v2_main.csv", main_rows)
    write_csv(RESULT_DIR / "rcetr_v2_random_raw.csv", random_raw)
    write_csv(RESULT_DIR / "rcetr_v2_random_summary.csv", summary)

    report = [
        "# Reliability-Calibrated Evidence Tree Routing V2",
        "",
        "This cache-only experiment upgrades the bucket lookup gate into a sample-adaptive reliability estimator. The feature set combines a coarse bucket prior with fine evidence-tree compatibility signals computed from lesion presence, lesion type, count, location, instrument visibility, text overlay, abnormality, and uncertainty fields. Gold answers are used only to define route-supervision labels and threshold selection on the development prefix; evaluation features use only question metadata, predictions, and structured evidence.",
        "",
        "## First-529 Development Prefix",
        "",
        markdown_table(main_rows, ["method", "n", "exact", "token_f1", "lexical_drift", "evidence_route_rate", "threshold", "dev_token_f1", "feature_count", "mean_route_probability"]),
        "",
        "## Random Development Prefixes",
        "",
        markdown_table(summary, ["dev_size", "runs", "token_f1_mean", "token_f1_std", "exact_mean", "lexical_drift_mean", "evidence_route_rate_mean", "threshold_mean"]),
        "",
        "## Algorithmic Interpretation",
        "",
        "- The original bucket gate acts as a coarse reliability prior.",
        "- Evidence-tree compatibility supplies a fine per-sample signal over structured evidence fields.",
        "- A calibrated logistic router selects between direct-template and evidence-reranked answers.",
        "- This mirrors a coarse-to-fine routing pattern: do not activate evidence merely because it exists; activate it when the current sample's evidence tree is compatible and the bucket prior supports evidence use.",
        "",
    ]
    out_path = REPORT_DIR / "rcetr_v2_report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
