#!/usr/bin/env python3
"""Build a small Kvasir-VQA-x1 bundle for GPU-server VLM sanity tests."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
IMAGE_ROOT = ROOT / "cache" / "kvasir_vqa_images"
BUNDLE_ROOT = ROOT / "server_bundle"
IMAGE_OUT = BUNDLE_ROOT / "images" / "test"
MANIFEST = BUNDLE_ROOT / "sample_manifest.jsonl"
SUMMARY = BUNDLE_ROOT / "bundle_summary.json"

TARGET_CLASSES = [
    "abnormality_presence",
    "polyp_count",
    "instrument_count",
    "text_presence",
    "procedure_type",
    "polyp_type",
    "anatomical_landmark",
    "finding_presence",
]


def as_list(value) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def norm_answer(value) -> str:
    return str(value).strip()


def main() -> int:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_ROOT / "test.parquet").copy()
    df["classes"] = df["question_class"].apply(as_list)
    df["primary_class"] = df["classes"].apply(lambda xs: xs[0] if xs else "unknown")
    df["image_path"] = df["img_id"].apply(lambda x: IMAGE_ROOT / "test" / f"{x}.jpg")
    df = df[df["image_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)]
    df = df[df["answer"].apply(lambda x: bool(norm_answer(x)))]

    selected = []
    seen_questions = set()
    per_class_limit = 12
    per_complexity_limit = 45
    class_counts: Counter[str] = Counter()
    complexity_counts: Counter[int] = Counter()

    # Prefer classes that give meaningful visual evidence, then add coverage.
    priority = {name: i for i, name in enumerate(TARGET_CLASSES)}
    df["priority"] = df["classes"].apply(
        lambda xs: min([priority.get(x, len(priority) + 1) for x in xs] or [len(priority) + 1])
    )
    df = df.sort_values(["priority", "complexity", "primary_class", "img_id"], ascending=[True, False, True, True])

    for _, row in df.iterrows():
        classes = row["classes"]
        primary = next((c for c in classes if c in TARGET_CLASSES), row["primary_class"])
        complexity = int(row["complexity"])
        qkey = (str(row["img_id"]), str(row["question"]))
        if qkey in seen_questions:
            continue
        if class_counts[primary] >= per_class_limit:
            continue
        if complexity_counts[complexity] >= per_complexity_limit:
            continue
        dest = IMAGE_OUT / f"{row['img_id']}.jpg"
        shutil.copy2(row["image_path"], dest)
        item = {
            "id": f"test-{len(selected):06d}",
            "split": "test",
            "img_id": str(row["img_id"]),
            "image_path": f"images/test/{row['img_id']}.jpg",
            "question": str(row["question"]),
            "answer": norm_answer(row["answer"]),
            "complexity": complexity,
            "question_class": classes,
        }
        selected.append(item)
        seen_questions.add(qkey)
        class_counts[primary] += 1
        complexity_counts[complexity] += 1
        if len(selected) >= 100:
            break

    # If the priority classes were too restrictive, fill with remaining samples.
    if len(selected) < 100:
        selected_ids = {(x["img_id"], x["question"]) for x in selected}
        for _, row in df.iterrows():
            qkey = (str(row["img_id"]), str(row["question"]))
            if qkey in selected_ids:
                continue
            dest = IMAGE_OUT / f"{row['img_id']}.jpg"
            shutil.copy2(row["image_path"], dest)
            selected.append(
                {
                    "id": f"test-{len(selected):06d}",
                    "split": "test",
                    "img_id": str(row["img_id"]),
                    "image_path": f"images/test/{row['img_id']}.jpg",
                    "question": str(row["question"]),
                    "answer": norm_answer(row["answer"]),
                    "complexity": int(row["complexity"]),
                    "question_class": row["classes"],
                }
            )
            if len(selected) >= 100:
                break

    with MANIFEST.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "n": len(selected),
        "complexity_counts": dict(Counter(x["complexity"] for x in selected)),
        "class_counts": dict(Counter((x["question_class"] or ["unknown"])[0] for x in selected)),
        "image_count": len(list(IMAGE_OUT.glob("*.jpg"))),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
