#!/usr/bin/env python3
"""Run Qwen direct VQA and structured evidence extraction on a Kvasir sanity bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "models/Qwen3.5-9B")


DIRECT_SYSTEM = (
    "You answer endoscopic visual questions. Do not show reasoning. "
    "Return only the final short answer."
)

EVIDENCE_SYSTEM = (
    "You inspect endoscopic images and return structured visual evidence. "
    "Do not show reasoning. Return only valid compact JSON."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_thinking(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = re.sub(r"(?is)^thinking process:\s*", "", text).strip()
    return text.strip()


def build_inputs(processor, image_path: Path, system_prompt: str, user_prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
            ],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )


def generate(model, processor, image_path: Path, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    inputs = build_inputs(processor, image_path, system_prompt, user_prompt).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    output_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return strip_thinking(processor.batch_decode(output_ids, skip_special_tokens=True)[0])


def parse_json_object(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()
    match = re.search(r"\{.*\}", clean, flags=re.S)
    if match:
        clean = match.group(0)
    try:
        obj = json.loads(clean)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="step5_experiments/server_bundle")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default="step5_experiments/results/qwen_sanity_outputs.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    manifest_path = Path(args.manifest) if args.manifest else bundle / "sample_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    if args.start > 0:
        manifest = manifest[args.start :]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("id"):
                    done_ids.add(row["id"])

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for idx, item in enumerate(manifest, 1):
            if item["id"] in done_ids:
                continue
            started = time.time()
            image_path = bundle / item["image_path"]
            if not image_path.exists() or image_path.stat().st_size <= 0:
                row = {
                    "id": item["id"],
                    "img_id": item["img_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "complexity": item["complexity"],
                    "question_class": item["question_class"],
                    "error": f"missing_image:{image_path}",
                    "elapsed_sec": 0,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{idx}/{len(manifest)}] {item['id']} missing image", flush=True)
                continue
            question = item["question"]
            direct_prompt = (
                "Answer the question from the image. Use a concise dataset-style answer. "
                "Return only the answer text, no explanation.\n"
                f"Question: {question}"
            )
            evidence_prompt = (
                "Question: {question}\n"
                "Return exactly one JSON object with these string keys: "
                "lesion_presence, lesion_type, lesion_count, location, instrument_presence, "
                "text_overlay_presence, abnormality_presence, uncertainty, evidence_sentence. "
                "Use yes/no/unknown when appropriate. Do not add markdown."
            ).format(question=question)

            try:
                direct_raw = generate(model, processor, image_path, DIRECT_SYSTEM, direct_prompt, 64)
                evidence_raw = generate(model, processor, image_path, EVIDENCE_SYSTEM, evidence_prompt, 192)
                parsed_evidence = parse_json_object(evidence_raw)
                error = None
            except Exception as exc:
                direct_raw = ""
                evidence_raw = ""
                parsed_evidence = None
                error = repr(exc)
            row = {
                "id": item["id"],
                "img_id": item["img_id"],
                "question": question,
                "answer": item["answer"],
                "complexity": item["complexity"],
                "question_class": item["question_class"],
                "qwen_direct": direct_raw,
                "qwen_evidence_raw": evidence_raw,
                "qwen_evidence": parsed_evidence,
                "error": error,
                "elapsed_sec": round(time.time() - started, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if error:
                print(f"[{idx}/{len(manifest)}] {item['id']} ERROR {error}", flush=True)
            else:
                print(f"[{idx}/{len(manifest)}] {item['id']} {row['elapsed_sec']}s direct={direct_raw[:80]!r}", flush=True)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
