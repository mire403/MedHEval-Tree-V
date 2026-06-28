#!/usr/bin/env python3
"""Build full Kvasir-VQA-x1 test manifest for server-side Qwen experiments."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
LOCAL_IMAGE_ROOT = ROOT / "cache" / "kvasir_vqa_images"
BUNDLE_ROOT = ROOT / "server_bundle_full"
IMAGE_OUT = BUNDLE_ROOT / "images"
MANIFEST = BUNDLE_ROOT / "test_full_manifest.jsonl"
IMAGE_MANIFEST = BUNDLE_ROOT / "test_image_manifest.jsonl"
SUMMARY = BUNDLE_ROOT / "full_bundle_summary.json"


def as_list(value) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def clean_answer(value) -> str:
    return str(value).strip()


def main() -> int:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    (IMAGE_OUT / "test").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_ROOT / "test.parquet")

    rows = []
    images = {}
    for idx, row in df.iterrows():
        img_id = str(row["img_id"])
        image_rel = f"images/test/{img_id}.jpg"
        image_url = str(row["image"])
        local_image = LOCAL_IMAGE_ROOT / "test" / f"{img_id}.jpg"
        if local_image.exists() and local_image.stat().st_size > 0:
            dest = BUNDLE_ROOT / image_rel
            if not dest.exists():
                shutil.copy2(local_image, dest)
        item = {
            "id": f"test-{idx:06d}",
            "split": "test",
            "img_id": img_id,
            "image_path": image_rel,
            "image_url": image_url,
            "question": str(row["question"]),
            "answer": clean_answer(row["answer"]),
            "complexity": int(row["complexity"]),
            "question_class": as_list(row["question_class"]),
        }
        rows.append(item)
        images[img_id] = {
            "img_id": img_id,
            "image_path": image_rel,
            "image_url": image_url,
            "local_exists": (BUNDLE_ROOT / image_rel).exists(),
        }

    with MANIFEST.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with IMAGE_MANIFEST.open("w", encoding="utf-8") as f:
        for item in sorted(images.values(), key=lambda x: x["img_id"]):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    existing_images = len([x for x in images.values() if x["local_exists"]])
    class_counter = Counter()
    for item in rows:
        class_counter.update(item["question_class"])
    summary = {
        "qa_rows": len(rows),
        "unique_images": len(images),
        "existing_images_in_bundle": existing_images,
        "missing_images": len(images) - existing_images,
        "complexity_counts": dict(Counter(item["complexity"] for item in rows)),
        "top_classes": class_counter.most_common(30),
        "manifest": str(MANIFEST),
        "image_manifest": str(IMAGE_MANIFEST),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
