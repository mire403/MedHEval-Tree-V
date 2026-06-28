#!/usr/bin/env python3
"""Download full Kvasir-VQA test images locally using curl with parallel workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "server_bundle_full"
IMAGE_MANIFEST = BUNDLE_ROOT / "test_image_manifest.jsonl"
REPORT = BUNDLE_ROOT / "local_download_report.json"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_one(row: dict, bundle_root: Path, timeout: int) -> dict:
    dest = bundle_root / row["image_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"img_id": row["img_id"], "ok": True, "status": "exists", "bytes": dest.stat().st_size}
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl",
        "-L",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "20",
        "--max-time",
        str(timeout),
        "--retry",
        "2",
        "--retry-delay",
        "2",
        "-o",
        str(tmp),
        row["image_url"],
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return {"img_id": row["img_id"], "ok": True, "status": "downloaded", "bytes": dest.stat().st_size}
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    return {
        "img_id": row["img_id"],
        "ok": False,
        "status": proc.stderr.strip() or f"curl_return_{proc.returncode}",
        "bytes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = list(iter_jsonl(IMAGE_MANIFEST))
    if args.limit > 0:
        rows = rows[: args.limit]
    started = time.time()
    ok = 0
    failed = []
    statuses = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, row, BUNDLE_ROOT, args.timeout) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            ok += int(result["ok"])
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            if not result["ok"]:
                failed.append(result)
            if i % 100 == 0 or not result["ok"]:
                print(f"[progress] {i}/{len(rows)} ok={ok} failed={len(failed)} status={result['status']}", flush=True)
    report = {
        "total": len(rows),
        "ok": ok,
        "failed": len(failed),
        "statuses": statuses,
        "failed_items": failed[:200],
        "elapsed_sec": round(time.time() - started, 3),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
