#!/usr/bin/env python3
"""Download a focused Kvasir-VQA-x1 image subset for visual-evidence tests."""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "step3_dataset_migration" / "raw" / "kvasir_vqa_x1"
OUT_DIR = ROOT / "cache" / "kvasir_vqa_images"
META_PATH = ROOT / "cache" / "kvasir_vqa_image_subset_manifest.json"

TARGET_CLASSES = {
    "text_presence",
    "box_artifact_presence",
    "polyp_count",
    "instrument_count",
    "abnormality_presence",
    "procedure_type",
}


def classes_contain(value, targets: set[str]) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return False
    return bool(set(map(str, value)) & targets)


def collect(split: str, max_images: int) -> list[dict]:
    df = pd.read_parquet(DATA_ROOT / f"{split}.parquet")
    sub = df[df["question_class"].apply(lambda x: classes_contain(x, TARGET_CLASSES))]
    # Keep deterministic, balanced-ish coverage by taking the first occurrence
    # of each image after sorting by complexity descending.
    sub = sub.sort_values(["complexity", "img_id"], ascending=[False, True])
    seen = set()
    rows = []
    for _, row in sub.iterrows():
        img_id = str(row["img_id"])
        if img_id in seen:
            continue
        seen.add(img_id)
        rows.append({"split": split, "img_id": img_id, "url": str(row["image"])})
        if len(rows) >= max_images:
            break
    return rows


def download(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 medheval-tree/0.1"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if tmp.stat().st_size == 0:
                raise RuntimeError("zero-byte download")
            tmp.replace(dest)
            return True
        except Exception as exc:
            print(f"[warn] {dest.name} attempt {attempt}/{retries}: {exc}", flush=True)
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            time.sleep(attempt)
    return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect("train", 1200) + collect("test", 900)
    ok = 0
    for i, row in enumerate(rows, 1):
        dest = OUT_DIR / row["split"] / f"{row['img_id']}.jpg"
        row["local_path"] = str(dest)
        row["downloaded"] = download(row["url"], dest)
        ok += int(row["downloaded"])
        if i % 100 == 0:
            print(f"[progress] {i}/{len(rows)} downloaded={ok}", flush=True)
    META_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] downloaded={ok}/{len(rows)} manifest={META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
