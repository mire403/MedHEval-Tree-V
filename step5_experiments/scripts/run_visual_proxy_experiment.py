#!/usr/bin/env python3
"""Visual proxy evidence experiment for Kvasir-VQA-x1.

This is not a substitute for a real VLM. It tests whether even lightweight
image-derived evidence can improve/validate category-specific answers.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
IMG_ROOT = ROOT / "cache" / "kvasir_vqa_images"
OUT_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"

TARGET_CLASSES = [
    "text_presence",
    "box_artifact_presence",
    "polyp_count",
    "instrument_count",
]


def normalize(text: str) -> str:
    text = str(text).lower().replace("polyps", "polyp")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_binary(answer: str, cls: str) -> str | None:
    a = normalize(answer)
    neg = any(x in a for x in [" no ", "no ", "none", "not ", "without", "absent"])
    if cls in {"text_presence", "box_artifact_presence"}:
        pos_words = {
            "text_presence": ["text present", "text is present", "text detected", "text observed", "textual content present"],
            "box_artifact_presence": ["artifact", "artefact", "box artifacts", "box artefact", "green and black box"],
        }[cls]
        pos = any(w in a for w in pos_words)
        if neg and not pos:
            return "no"
        if pos and not a.startswith("no evidence") and "no green" not in a and "no box" not in a:
            return "yes"
    if cls == "polyp_count":
        if "no polyp" in a or "none" in a:
            return "none"
        if any(x in a for x in ["one polyp", "single polyp", "1 polyp"]):
            return "one"
    if cls == "instrument_count":
        if "no instrument" in a or "no surgical instrument" in a or "none" in a:
            return "none"
        if any(x in a for x in ["one surgical instrument", "one instrument", "single instrument", "1 instrument"]):
            return "one"
    return None


def contains_class(value, cls: str) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return isinstance(value, (list, tuple)) and cls in set(map(str, value))


def feature_from_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    hsv_like = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]], axis=2)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    bright = np.array([gray.mean(), gray.std(), (gray > 0.8).mean(), (gray < 0.1).mean()])
    red = ((arr[:, :, 0] > 0.45) & (arr[:, :, 0] > arr[:, :, 1] * 1.15)).mean()
    green = ((arr[:, :, 1] > 0.35) & (arr[:, :, 1] > arr[:, :, 0] * 1.1)).mean()
    black_green = ((gray < 0.18) | ((arr[:, :, 1] > 0.4) & (arr[:, :, 0] < 0.35))).mean()

    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    earr = np.asarray(edges).astype(np.float32) / 255.0
    edge_stats = np.array([earr.mean(), earr.std(), (earr > 0.25).mean()])
    # A crude text-like proxy: high edge density near top/bottom overlays.
    top_edge = (earr[:24, :] > 0.25).mean()
    bottom_edge = (earr[-24:, :] > 0.25).mean()
    center_edge = (earr[40:88, 40:88] > 0.25).mean()
    return np.concatenate([mean, std, bright, [red, green, black_green], edge_stats, [top_edge, bottom_edge, center_edge]])


def build_rows(split: str, cls: str) -> list[dict]:
    df = pd.read_parquet(DATA_ROOT / f"{split}.parquet")
    rows = []
    for _, row in df.iterrows():
        if not contains_class(row["question_class"], cls):
            continue
        img_id = str(row["img_id"])
        path = IMG_ROOT / split / f"{img_id}.jpg"
        if not path.exists() or path.stat().st_size == 0:
            continue
        label = answer_binary(str(row["answer"]), cls)
        if label is None:
            continue
        rows.append({"img_id": img_id, "answer": str(row["answer"]), "label": label, "path": path})
    return rows


def nearest_centroid_predict(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    labels = sorted(set(train_y))
    centroids = []
    for label in labels:
        centroids.append(train_x[[i for i, y in enumerate(train_y) if y == label]].mean(axis=0))
    centroids = np.vstack(centroids)
    # Standardize with train statistics.
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    c = (centroids - mean) / std
    x = (test_x - mean) / std
    dists = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return [labels[int(i)] for i in dists.argmin(axis=1)]


def majority_predict(train_y: list[str], n: int) -> list[str]:
    label = Counter(train_y).most_common(1)[0][0]
    return [label] * n


def accuracy(preds: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(preds, gold)) / len(gold) if gold else 0.0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for cls in TARGET_CLASSES:
        train_rows = build_rows("train", cls)
        test_rows = build_rows("test", cls)
        if len(train_rows) < 10 or len(test_rows) < 5:
            results.append({"class": cls, "status": "too_few_images", "train_n": len(train_rows), "test_n": len(test_rows)})
            continue
        train_x = np.vstack([feature_from_image(r["path"]) for r in train_rows])
        test_x = np.vstack([feature_from_image(r["path"]) for r in test_rows])
        train_y = [r["label"] for r in train_rows]
        test_y = [r["label"] for r in test_rows]
        majority = majority_predict(train_y, len(test_y))
        visual = nearest_centroid_predict(train_x, train_y, test_x)
        results.append(
            {
                "class": cls,
                "status": "ok",
                "train_n": len(train_y),
                "test_n": len(test_y),
                "train_label_dist": dict(Counter(train_y)),
                "test_label_dist": dict(Counter(test_y)),
                "majority_acc": accuracy(majority, test_y),
                "visual_proxy_acc": accuracy(visual, test_y),
            }
        )

    out_json = OUT_DIR / "visual_proxy_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = OUT_DIR / "visual_proxy_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["class", "status", "train_n", "test_n", "majority_acc", "visual_proxy_acc", "train_label_dist", "test_label_dist"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), dict) else row.get(k, "") for k in fieldnames})

    md = ["# Step 5 Round 2 Visual Proxy Report", "", "| Class | Train | Test | Majority Acc | Visual Proxy Acc |", "|---|---:|---:|---:|---:|"]
    for row in results:
        md.append(
            f"| {row['class']} | {row.get('train_n', 0)} | {row.get('test_n', 0)} | "
            f"{row.get('majority_acc', 0):.4f} | {row.get('visual_proxy_acc', 0):.4f} |"
        )
    md.extend(
        [
            "",
            "Interpretation: this uses only crude image statistics, not a VLM. If it beats majority on a class, visual evidence is likely worth adding. If not, we need a real VLM or class-specific detectors.",
        ]
    )
    (REPORT_DIR / "visual_proxy_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_csv)
    print(REPORT_DIR / "visual_proxy_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
