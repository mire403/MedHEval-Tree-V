#!/usr/bin/env python3
"""Build a compact curl manifest for Kvasir-VQA-x1 image subset."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
OUT_DIR = ROOT / "cache" / "kvasir_vqa_images"
MANIFEST = ROOT / "cache" / "kvasir_vqa_curl_manifest.csv"
URL_LIST = ROOT / "cache" / "kvasir_vqa_curl_commands.tsv"

TARGET_CLASSES = {
    "text_presence",
    "box_artifact_presence",
    "polyp_count",
    "instrument_count",
    "abnormality_presence",
    "procedure_type",
}


def contains_target(value) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return False
    return bool(set(map(str, value)) & TARGET_CLASSES)


def collect(split: str, max_images: int) -> list[dict]:
    df = pd.read_parquet(DATA_ROOT / f"{split}.parquet")
    sub = df[df["question_class"].apply(contains_target)]
    # Keep deterministic spread: sort by image id and take unique images.
    sub = sub.sort_values(["img_id", "complexity"], ascending=[True, False])
    rows = []
    seen = set()
    for _, row in sub.iterrows():
        img_id = str(row["img_id"])
        if img_id in seen:
            continue
        seen.add(img_id)
        local = OUT_DIR / split / f"{img_id}.jpg"
        rows.append(
            {
                "split": split,
                "img_id": img_id,
                "url": str(row["image"]),
                "local_path": str(local),
                "exists": local.exists() and local.stat().st_size > 0,
            }
        )
        if len(rows) >= max_images:
            break
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect("train", 300) + collect("test", 300)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with URL_LIST.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["exists"]:
                continue
            Path(row["local_path"]).parent.mkdir(parents=True, exist_ok=True)
            f.write(f"{row['url']}\t{row['local_path']}\n")
    print(f"[done] rows={len(rows)} missing={sum(not r['exists'] for r in rows)}")
    print(MANIFEST)
    print(URL_LIST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
