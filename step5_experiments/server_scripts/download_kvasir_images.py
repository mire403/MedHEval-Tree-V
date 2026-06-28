#!/usr/bin/env python3
"""Download Kvasir-VQA images on the GPU server from a JSONL image manifest."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download(url: str, dest: Path, retries: int) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True, "exists"
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 medheval-tree/0.2"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response, tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if tmp.stat().st_size <= 0:
                raise RuntimeError("zero-byte download")
            tmp.replace(dest)
            return True, "downloaded"
        except Exception as exc:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt == retries:
                return False, repr(exc)
            time.sleep(min(10, attempt * 2))
    return False, "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="step5_experiments/server_bundle_full")
    parser.add_argument("--image-manifest", default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    manifest = Path(args.image_manifest) if args.image_manifest else bundle / "test_image_manifest.jsonl"
    rows = list(iter_jsonl(manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    ok = 0
    failed = []
    for i, row in enumerate(rows, 1):
        dest = bundle / row["image_path"]
        success, status = download(row["image_url"], dest, args.retries)
        ok += int(success)
        if not success:
            failed.append({"img_id": row["img_id"], "status": status})
        if i % 100 == 0 or not success:
            print(f"[progress] {i}/{len(rows)} ok={ok} failed={len(failed)} last={status}", flush=True)
    report = {
        "total": len(rows),
        "ok": ok,
        "failed": len(failed),
        "failed_items": failed[:200],
    }
    out = bundle / "download_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
