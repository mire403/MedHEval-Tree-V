#!/usr/bin/env python3
"""Cache-only strengthening analyses for Reliability-Gated MedHEval-Tree-V.

This script intentionally uses only the frozen full-test prediction CSV. It
does not call the MLLM and does not rebuild the template bank.
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

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "as", "by",
    "this", "that", "there", "it", "its", "no", "not", "any",
}


def norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return norm(text).split()


def token_f1(pred: str, gold: str) -> float:
    pt = tokens(pred)
    gt = tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc = defaultdict(int)
    gc = defaultdict(int)
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


def ci95(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    xs = sorted(xs)
    lo = xs[int(0.025 * (len(xs) - 1))]
    hi = xs[int(0.975 * (len(xs) - 1))]
    return lo, hi


def primary_class(row: dict[str, str]) -> str:
    try:
        value = json.loads(row["question_class"])
        if isinstance(value, list) and value:
            return str(value[0])
    except Exception:
        pass
    return row.get("question_class", "unknown") or "unknown"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with PRED_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            item = dict(row)
            item["_idx"] = i
            item["_complexity"] = str(row["complexity"])
            item["_primary_class"] = primary_class(row)
            item["_direct_pred"] = row["tfidf_direct_template_prediction"]
            item["_evidence_pred"] = row["tfidf_evidence_rerank_prediction"]
            item["_direct_f1"] = float(row["tfidf_direct_template_token_f1"])
            item["_evidence_f1"] = float(row["tfidf_evidence_rerank_token_f1"])
            item["_direct_exact"] = float(row["tfidf_direct_template_exact"])
            item["_evidence_exact"] = float(row["tfidf_evidence_rerank_exact"])
            rows.append(item)
    return rows


def key_for(row: dict[str, object], mode: str) -> tuple[str, ...]:
    if mode == "global":
        return ("global",)
    if mode == "complexity":
        return (str(row["_complexity"]),)
    if mode == "class":
        return (str(row["_primary_class"]),)
    if mode == "bucket":
        return (str(row["_complexity"]), str(row["_primary_class"]))
    raise ValueError(f"unknown mode: {mode}")


def key_from_values(complexity: str, primary_cls: str, mode: str) -> tuple[str, ...]:
    if mode == "global":
        return ("global",)
    if mode == "complexity":
        return (complexity,)
    if mode == "class":
        return (primary_cls,)
    if mode == "bucket":
        return (complexity, primary_cls)
    raise ValueError(f"unknown mode: {mode}")


def fit_policy(rows: list[dict[str, object]], indices: list[int], mode: str, min_count: int = 8) -> dict[tuple[str, ...], str]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx in indices:
        grouped[key_for(rows[idx], mode)].append(idx)
    policy: dict[tuple[str, ...], str] = {}
    for key, group in grouped.items():
        if len(group) < min_count:
            policy[key] = "direct"
            continue
        direct = mean([float(rows[i]["_direct_f1"]) for i in group])
        evidence = mean([float(rows[i]["_evidence_f1"]) for i in group])
        policy[key] = "evidence" if evidence > direct else "direct"
    return policy


def choose_route(row: dict[str, object], mode: str, policy: dict[tuple[str, ...], str]) -> str:
    return policy.get(key_for(row, mode), "direct")


def choose_route_from_values(complexity: str, primary_cls: str, mode: str, policy: dict[tuple[str, ...], str]) -> str:
    return policy.get(key_from_values(complexity, primary_cls, mode), "direct")


def evaluate_route(rows: list[dict[str, object]], indices: list[int], mode: str, policy: dict[tuple[str, ...], str]) -> dict[str, float]:
    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    evidence_count = 0
    for idx in indices:
        row = rows[idx]
        route = choose_route(row, mode, policy)
        if route == "evidence":
            pred = str(row["_evidence_pred"])
            f1s.append(float(row["_evidence_f1"]))
            exacts.append(float(row["_evidence_exact"]))
            evidence_count += 1
        else:
            pred = str(row["_direct_pred"])
            f1s.append(float(row["_direct_f1"]))
            exacts.append(float(row["_direct_exact"]))
        drifts.append(lexical_drift(pred, str(row["gold"])))
    return {
        "n": len(indices),
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "evidence_route_rate": evidence_count / len(indices) if indices else 0.0,
    }


def evaluate_prediction_field(rows: list[dict[str, object]], indices: list[int], field: str) -> dict[str, float]:
    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    for idx in indices:
        row = rows[idx]
        pred = str(row[field])
        gold = str(row["gold"])
        f1s.append(token_f1(pred, gold))
        exacts.append(exact(pred, gold))
        drifts.append(lexical_drift(pred, gold))
    return {"n": len(indices), "exact": mean(exacts), "token_f1": mean(f1s), "lexical_drift": mean(drifts)}


def evaluate_stored_route(rows: list[dict[str, object]], indices: list[int], route: str) -> dict[str, float]:
    f1_key = "_evidence_f1" if route == "evidence" else "_direct_f1"
    exact_key = "_evidence_exact" if route == "evidence" else "_direct_exact"
    pred_key = "_evidence_pred" if route == "evidence" else "_direct_pred"
    f1s: list[float] = []
    exacts: list[float] = []
    drifts: list[float] = []
    for idx in indices:
        row = rows[idx]
        f1s.append(float(row[f1_key]))
        exacts.append(float(row[exact_key]))
        drifts.append(lexical_drift(str(row[pred_key]), str(row["gold"])))
    return {"n": len(indices), "exact": mean(exacts), "token_f1": mean(f1s), "lexical_drift": mean(drifts)}


def run_fixed_metadata_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dev = list(range(529))
    eval_idx = list(range(529, len(rows)))
    out: list[dict[str, object]] = []
    for mode in ["global", "complexity", "class", "bucket"]:
        policy = fit_policy(rows, dev, mode, min_count=8)
        metrics = evaluate_route(rows, eval_idx, mode, policy)
        metrics.update({"split": "first529_rest", "method": f"{mode}_policy", "policy_entries": len(policy)})
        out.append(metrics)
    direct = evaluate_stored_route(rows, eval_idx, "direct")
    direct.update({"split": "first529_rest", "method": "direct_template", "policy_entries": 0, "evidence_route_rate": 0.0})
    evidence = evaluate_stored_route(rows, eval_idx, "evidence")
    evidence.update({"split": "first529_rest", "method": "uniform_evidence", "policy_entries": 0, "evidence_route_rate": 1.0})
    return [direct, evidence] + out


def text_features(question: str) -> list[str]:
    toks = [t for t in tokens(question) if t not in STOPWORDS]
    feats = [f"u={t}" for t in toks]
    feats += [f"b={a}_{b}" for a, b in zip(toks, toks[1:])]
    if "how many" in norm(question):
        feats.append("phrase=how_many")
    if "where" in tokens(question):
        feats.append("phrase=where")
    if "what" in tokens(question):
        feats.append("phrase=what")
    if "is" in tokens(question) or "are" in tokens(question):
        feats.append("phrase=yesno")
    return feats


def train_nb(rows: list[dict[str, object]], indices: list[int], label_key: str) -> dict[str, object]:
    label_counts: dict[str, int] = defaultdict(int)
    token_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_tokens: dict[str, int] = defaultdict(int)
    vocab: set[str] = set()
    for idx in indices:
        row = rows[idx]
        label = str(row[label_key])
        label_counts[label] += 1
        for feat in text_features(str(row["question"])):
            token_counts[label][feat] += 1
            total_tokens[label] += 1
            vocab.add(feat)
    majority = max(label_counts, key=label_counts.get)
    return {
        "label_counts": dict(label_counts),
        "token_counts": {k: dict(v) for k, v in token_counts.items()},
        "total_tokens": dict(total_tokens),
        "vocab": vocab,
        "majority": majority,
        "n": len(indices),
    }


def predict_nb(model: dict[str, object], question: str) -> str:
    label_counts: dict[str, int] = model["label_counts"]  # type: ignore[assignment]
    token_counts: dict[str, dict[str, int]] = model["token_counts"]  # type: ignore[assignment]
    total_tokens: dict[str, int] = model["total_tokens"]  # type: ignore[assignment]
    vocab: set[str] = model["vocab"]  # type: ignore[assignment]
    if not label_counts:
        return str(model["majority"])
    feats = text_features(question)
    total_docs = sum(label_counts.values())
    vocab_size = max(1, len(vocab))
    best_label = str(model["majority"])
    best_score = -1e100
    for label, count in label_counts.items():
        score = math.log((count + 1) / (total_docs + len(label_counts)))
        denom = total_tokens.get(label, 0) + vocab_size
        counts = token_counts.get(label, {})
        for feat in feats:
            score += math.log((counts.get(feat, 0) + 1) / denom)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def run_predicted_metadata_gate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dev = list(range(529))
    eval_idx = list(range(529, len(rows)))
    complexity_model = train_nb(rows, dev, "_complexity")
    class_model = train_nb(rows, dev, "_primary_class")
    true_complexity = [str(rows[i]["_complexity"]) for i in eval_idx]
    pred_complexity = [predict_nb(complexity_model, str(rows[i]["question"])) for i in eval_idx]
    true_class = [str(rows[i]["_primary_class"]) for i in eval_idx]
    pred_class = [predict_nb(class_model, str(rows[i]["question"])) for i in eval_idx]
    complexity_acc = mean([float(a == b) for a, b in zip(true_complexity, pred_complexity)])
    class_acc = mean([float(a == b) for a, b in zip(true_class, pred_class)])

    out: list[dict[str, object]] = []
    for mode in ["complexity", "class", "bucket"]:
        policy = fit_policy(rows, dev, mode, min_count=8)
        f1s: list[float] = []
        exacts: list[float] = []
        drifts: list[float] = []
        routed = 0
        for local_pos, idx in enumerate(eval_idx):
            row = rows[idx]
            c = pred_complexity[local_pos] if mode in {"complexity", "bucket"} else str(row["_complexity"])
            cls = pred_class[local_pos] if mode in {"class", "bucket"} else str(row["_primary_class"])
            route = choose_route_from_values(c, cls, mode, policy)
            if route == "evidence":
                pred = str(row["_evidence_pred"])
                f1s.append(float(row["_evidence_f1"]))
                exacts.append(float(row["_evidence_exact"]))
                routed += 1
            else:
                pred = str(row["_direct_pred"])
                f1s.append(float(row["_direct_f1"]))
                exacts.append(float(row["_direct_exact"]))
            drifts.append(lexical_drift(pred, str(row["gold"])))
        out.append({
            "method": f"predicted_{mode}_policy",
            "n": len(eval_idx),
            "exact": mean(exacts),
            "token_f1": mean(f1s),
            "lexical_drift": mean(drifts),
            "evidence_route_rate": routed / len(eval_idx),
            "complexity_acc": complexity_acc,
            "primary_class_acc": class_acc,
            "policy_entries": len(policy),
        })
    return out


def run_min_count_sensitivity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dev = list(range(529))
    eval_idx = list(range(529, len(rows)))
    out: list[dict[str, object]] = []
    for min_count in [1, 3, 5, 8, 12, 20]:
        for mode in ["class", "bucket"]:
            policy = fit_policy(rows, dev, mode, min_count=min_count)
            metrics = evaluate_route(rows, eval_idx, mode, policy)
            metrics.update({
                "method": f"{mode}_policy",
                "min_count": min_count,
                "policy_entries": len(policy),
            })
            out.append(metrics)
    return out


def run_random_splits(rows: list[dict[str, object]], seeds: int = 10, dev_size: int = 529) -> list[dict[str, object]]:
    n = len(rows)
    all_idx = list(range(n))
    results: list[dict[str, object]] = []
    for seed in range(seeds):
        rng = random.Random(seed)
        perm = all_idx[:]
        rng.shuffle(perm)
        dev = perm[:dev_size]
        eval_idx = perm[dev_size:]
        for mode in ["complexity", "class", "bucket"]:
            policy = fit_policy(rows, dev, mode, min_count=8)
            metrics = evaluate_route(rows, eval_idx, mode, policy)
            metrics.update({"seed": seed, "dev_size": dev_size, "method": f"{mode}_policy", "policy_entries": len(policy)})
            results.append(metrics)
    return results


def summarize_random_splits(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    raw = run_random_splits(rows, seeds=10, dev_size=529) + run_random_splits(rows, seeds=10, dev_size=1000)
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        grouped[(int(row["dev_size"]), str(row["method"]))].append(row)
    out: list[dict[str, object]] = []
    for (dev_size, method), group in sorted(grouped.items()):
        f1s = [float(g["token_f1"]) for g in group]
        exacts = [float(g["exact"]) for g in group]
        drifts = [float(g["lexical_drift"]) for g in group]
        rates = [float(g["evidence_route_rate"]) for g in group]
        out.append({
            "dev_size": dev_size,
            "method": method,
            "runs": len(group),
            "token_f1_mean": mean(f1s),
            "token_f1_std": stdev(f1s),
            "exact_mean": mean(exacts),
            "lexical_drift_mean": mean(drifts),
            "evidence_route_rate_mean": mean(rates),
        })
    write_csv(RESULT_DIR / "cache_only_random_split_raw.csv", raw)
    return out


def run_bootstrap(rows: list[dict[str, object]], samples: int = 2000) -> dict[str, object]:
    dev = list(range(529))
    eval_idx = list(range(529, len(rows)))
    policy = fit_policy(rows, dev, "bucket", min_count=8)
    diffs = []
    per_row_diff = []
    for idx in eval_idx:
        row = rows[idx]
        route = choose_route(row, "bucket", policy)
        gate_f1 = float(row["_evidence_f1"]) if route == "evidence" else float(row["_direct_f1"])
        per_row_diff.append(gate_f1 - float(row["_direct_f1"]))
    rng = random.Random(2026)
    for _ in range(samples):
        diffs.append(mean([per_row_diff[rng.randrange(len(per_row_diff))] for _ in per_row_diff]))
    lo, hi = ci95(diffs)
    observed = mean(per_row_diff)
    p_two_sided = 2 * min(
        sum(1 for d in diffs if d <= 0) / len(diffs),
        sum(1 for d in diffs if d >= 0) / len(diffs),
    )
    return {
        "comparison": "bucket_gate_vs_direct_template",
        "eval_n": len(eval_idx),
        "observed_token_f1_gain": observed,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "bootstrap_two_sided_p": min(1.0, p_two_sided),
        "bootstrap_samples": samples,
    }


def run_shuffle_controls(rows: list[dict[str, object]], seeds: int = 5) -> list[dict[str, object]]:
    dev = list(range(529))
    eval_idx = list(range(529, len(rows)))
    policy = fit_policy(rows, dev, "bucket", min_count=8)
    base = evaluate_route(rows, eval_idx, "bucket", policy)
    base.update({"method": "bucket_gate_aligned", "seed": -1})
    direct = evaluate_stored_route(rows, eval_idx, "direct")
    direct.update({"method": "direct_template", "seed": -1, "evidence_route_rate": 0.0})
    out = [direct, base]
    for seed in range(seeds):
        rng = random.Random(1000 + seed)
        shuffled = eval_idx[:]
        rng.shuffle(shuffled)
        f1s = []
        exacts = []
        drifts = []
        routed = 0
        for idx, shuffled_idx in zip(eval_idx, shuffled):
            row = rows[idx]
            route = choose_route(row, "bucket", policy)
            if route == "evidence":
                pred = str(rows[shuffled_idx]["_evidence_pred"])
                routed += 1
            else:
                pred = str(row["_direct_pred"])
            gold = str(row["gold"])
            f1s.append(token_f1(pred, gold))
            exacts.append(exact(pred, gold))
            drifts.append(lexical_drift(pred, gold))
        out.append({
            "method": "bucket_gate_shuffled_evidence",
            "seed": seed,
            "n": len(eval_idx),
            "exact": mean(exacts),
            "token_f1": mean(f1s),
            "lexical_drift": mean(drifts),
            "evidence_route_rate": routed / len(eval_idx),
        })
    return out


def route_label(row: dict[str, object]) -> float:
    return 1.0 if float(row["_evidence_f1"]) > float(row["_direct_f1"]) else 0.0


def jaccard(a: str, b: str) -> float:
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def contains_any(text: str, terms: tuple[str, ...]) -> float:
    text = norm(text)
    return 1.0 if any(term in text for term in terms) else 0.0


def build_feature_space(rows: list[dict[str, object]], indices: list[int]) -> dict[str, int]:
    names = [
        "bias",
        "complexity",
        "direct_len",
        "evidence_len",
        "len_delta",
        "direct_evidence_jaccard",
        "question_direct_jaccard",
        "question_evidence_jaccard",
        "direct_no",
        "evidence_no",
        "evidence_multiple",
        "direct_multiple",
    ]
    classes = sorted({str(rows[i]["_primary_class"]) for i in indices})
    for cls in classes:
        names.append(f"class={cls}")
    return {name: i for i, name in enumerate(names)}


def featurize(row: dict[str, object], space: dict[str, int]) -> list[float]:
    vec = [0.0] * len(space)
    direct = str(row["_direct_pred"])
    evidence = str(row["_evidence_pred"])
    question = str(row["question"])
    direct_len = len(tokens(direct))
    evidence_len = len(tokens(evidence))
    values = {
        "bias": 1.0,
        "complexity": (float(row["_complexity"]) - 2.0) / 1.0,
        "direct_len": min(direct_len, 20) / 20.0,
        "evidence_len": min(evidence_len, 20) / 20.0,
        "len_delta": max(min(evidence_len - direct_len, 20), -20) / 20.0,
        "direct_evidence_jaccard": jaccard(direct, evidence),
        "question_direct_jaccard": jaccard(question, direct),
        "question_evidence_jaccard": jaccard(question, evidence),
        "direct_no": contains_any(direct, (" no ", "no ", "none", "absent")),
        "evidence_no": contains_any(evidence, (" no ", "no ", "none", "absent")),
        "evidence_multiple": contains_any(evidence, ("multiple", "several", "many")),
        "direct_multiple": contains_any(direct, ("multiple", "several", "many")),
        f"class={row['_primary_class']}": 1.0,
    }
    for key, value in values.items():
        if key in space:
            vec[space[key]] = value
    return vec


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def train_logistic(xs: list[list[float]], ys: list[float], epochs: int = 300, lr: float = 0.08, l2: float = 0.01) -> list[float]:
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


def eval_calibrated_gate(rows: list[dict[str, object]], train_idx: list[int], eval_idx: list[int]) -> dict[str, object]:
    space = build_feature_space(rows, train_idx)
    xs = [featurize(rows[i], space) for i in train_idx]
    ys = [route_label(rows[i]) for i in train_idx]
    w = train_logistic(xs, ys)

    dev_probs = [sigmoid(dot(w, featurize(rows[i], space))) for i in train_idx]
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best_threshold = 0.5
    best_dev_f1 = -1.0
    for threshold in thresholds:
        f1s = []
        for idx, prob in zip(train_idx, dev_probs):
            row = rows[idx]
            f1s.append(float(row["_evidence_f1"]) if prob >= threshold else float(row["_direct_f1"]))
        value = mean(f1s)
        if value > best_dev_f1:
            best_dev_f1 = value
            best_threshold = threshold

    f1s = []
    exacts = []
    drifts = []
    routed = 0
    for idx in eval_idx:
        row = rows[idx]
        prob = sigmoid(dot(w, featurize(row, space)))
        if prob >= best_threshold:
            pred = str(row["_evidence_pred"])
            f1s.append(float(row["_evidence_f1"]))
            exacts.append(float(row["_evidence_exact"]))
            routed += 1
        else:
            pred = str(row["_direct_pred"])
            f1s.append(float(row["_direct_f1"]))
            exacts.append(float(row["_direct_exact"]))
        drifts.append(lexical_drift(pred, str(row["gold"])))
    return {
        "method": "sample_calibrated_gate",
        "n": len(eval_idx),
        "exact": mean(exacts),
        "token_f1": mean(f1s),
        "lexical_drift": mean(drifts),
        "evidence_route_rate": routed / len(eval_idx),
        "threshold": best_threshold,
        "dev_token_f1": best_dev_f1,
        "feature_count": len(space),
    }


def run_calibrated_gate(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fixed = eval_calibrated_gate(rows, list(range(529)), list(range(529, len(rows))))
    fixed["split"] = "first529_rest"

    random_results: list[dict[str, object]] = []
    all_idx = list(range(len(rows)))
    for dev_size in [529, 1000]:
        for seed in range(10):
            rng = random.Random(3000 + seed)
            perm = all_idx[:]
            rng.shuffle(perm)
            dev = perm[:dev_size]
            eval_idx = perm[dev_size:]
            result = eval_calibrated_gate(rows, dev, eval_idx)
            result.update({"seed": seed, "dev_size": dev_size})
            random_results.append(result)
    return [fixed], random_results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    metadata = run_fixed_metadata_ablation(rows)
    random_summary = summarize_random_splits(rows)
    bootstrap = run_bootstrap(rows, samples=2000)
    shuffle = run_shuffle_controls(rows, seeds=5)
    calibrated_fixed, calibrated_random = run_calibrated_gate(rows)
    predicted_metadata = run_predicted_metadata_gate(rows)
    min_count_sensitivity = run_min_count_sensitivity(rows)

    write_csv(RESULT_DIR / "cache_only_metadata_ablation.csv", metadata)
    write_csv(RESULT_DIR / "cache_only_random_split_summary.csv", random_summary)
    write_csv(RESULT_DIR / "cache_only_bootstrap.csv", [bootstrap])
    write_csv(RESULT_DIR / "cache_only_shuffle_controls.csv", shuffle)
    write_csv(RESULT_DIR / "cache_only_calibrated_gate_fixed.csv", calibrated_fixed)
    write_csv(RESULT_DIR / "cache_only_calibrated_gate_random_raw.csv", calibrated_random)
    write_csv(RESULT_DIR / "cache_only_predicted_metadata_gate.csv", predicted_metadata)
    write_csv(RESULT_DIR / "cache_only_min_count_sensitivity.csv", min_count_sensitivity)

    shuffle_summary = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shuffle:
        grouped[str(row["method"])].append(row)
    for method, group in grouped.items():
        shuffle_summary.append({
            "method": method,
            "runs": len(group),
            "token_f1_mean": mean([float(g["token_f1"]) for g in group]),
            "token_f1_std": stdev([float(g["token_f1"]) for g in group]),
            "exact_mean": mean([float(g["exact"]) for g in group]),
            "lexical_drift_mean": mean([float(g["lexical_drift"]) for g in group]),
            "evidence_route_rate_mean": mean([float(g.get("evidence_route_rate", 0.0)) for g in group]),
        })
    write_csv(RESULT_DIR / "cache_only_shuffle_summary.csv", shuffle_summary)

    calibrated_summary = []
    by_dev: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in calibrated_random:
        by_dev[int(row["dev_size"])].append(row)
    for dev_size, group in sorted(by_dev.items()):
        calibrated_summary.append({
            "dev_size": dev_size,
            "runs": len(group),
            "token_f1_mean": mean([float(g["token_f1"]) for g in group]),
            "token_f1_std": stdev([float(g["token_f1"]) for g in group]),
            "exact_mean": mean([float(g["exact"]) for g in group]),
            "lexical_drift_mean": mean([float(g["lexical_drift"]) for g in group]),
            "evidence_route_rate_mean": mean([float(g["evidence_route_rate"]) for g in group]),
            "threshold_mean": mean([float(g["threshold"]) for g in group]),
        })
    write_csv(RESULT_DIR / "cache_only_calibrated_gate_summary.csv", calibrated_summary)

    report = [
        "# Cache-Only Experiment Strengthening Report",
        "",
        "This report uses the frozen full-test prediction cache only. No MLLM generation or template-bank rebuilding is performed. Token F1 and exact match for aligned direct/evidence routes use the stored per-row metrics from the original full run. Lexical drift values are recomputed here as a diagnostic and should be treated as local relative diagnostics rather than replacements for the paper tables.",
        "",
        "## Fixed First-529 Metadata Ablation",
        "",
        markdown_table(metadata, ["method", "n", "exact", "token_f1", "lexical_drift", "evidence_route_rate", "policy_entries"]),
        "",
        "## Random Split Stability",
        "",
        markdown_table(random_summary, ["dev_size", "method", "runs", "token_f1_mean", "token_f1_std", "exact_mean", "lexical_drift_mean", "evidence_route_rate_mean"]),
        "",
        "## Paired Bootstrap",
        "",
        markdown_table([bootstrap], ["comparison", "eval_n", "observed_token_f1_gain", "bootstrap_ci95_low", "bootstrap_ci95_high", "bootstrap_two_sided_p", "bootstrap_samples"]),
        "",
        "## Evidence Shuffle Control",
        "",
        markdown_table(shuffle_summary, ["method", "runs", "token_f1_mean", "token_f1_std", "exact_mean", "lexical_drift_mean", "evidence_route_rate_mean"]),
        "",
        "## Sample-Adaptive Calibrated Gate",
        "",
        markdown_table(calibrated_fixed, ["method", "split", "n", "exact", "token_f1", "lexical_drift", "evidence_route_rate", "threshold", "dev_token_f1", "feature_count"]),
        "",
        markdown_table(calibrated_summary, ["dev_size", "runs", "token_f1_mean", "token_f1_std", "exact_mean", "lexical_drift_mean", "evidence_route_rate_mean", "threshold_mean"]),
        "",
        "## Predicted Metadata Gate",
        "",
        markdown_table(predicted_metadata, ["method", "n", "exact", "token_f1", "lexical_drift", "evidence_route_rate", "complexity_acc", "primary_class_acc", "policy_entries"]),
        "",
        "## Minimum Bucket Count Sensitivity",
        "",
        markdown_table(min_count_sensitivity, ["method", "min_count", "n", "exact", "token_f1", "lexical_drift", "evidence_route_rate", "policy_entries"]),
        "",
    ]
    out_path = REPORT_DIR / "cache_only_strengthening_report.md"
    out_path.write_text("\n".join(report))
    print(out_path)


if __name__ == "__main__":
    main()
