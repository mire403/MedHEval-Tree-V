#!/usr/bin/env python3
"""Run stronger Qwen prompt baselines for Kvasir-VQA-x1.

Outputs one JSONL row per sample with three prompt variants:
- constrained_answer: image + question, concise answer only.
- class_constrained_answer: image + question + dataset metadata.
- evidence_then_answer: image + question + compact JSON evidence/final answer.
"""

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def final_from_json(text: str) -> str:
    obj = parse_json_object(text)
    if not obj:
        return strip_thinking(text)
    for key in ["final_answer", "answer", "final"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return strip_thinking(text)


def build_inputs(processor, image_path: Path, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
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


def generate(model, processor, image_path: Path, prompt: str, max_new_tokens: int) -> str:
    inputs = build_inputs(processor, image_path, prompt).to(model.device)
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


def concise_prompt(question: str) -> str:
    return (
        "You are answering a gastrointestinal endoscopy visual question. "
        "Return only the final answer, with no reasoning and no full sentence unless the answer requires it. "
        "Use concise benchmark style: yes/no when appropriate, a short count when asked how many, "
        "a short anatomical location when asked where, or a short lesion/finding name when asked what. "
        "Do not mention uncertainty unless the image is truly unclear.\n\n"
        f"Question: {question}\nAnswer:"
    )


def class_prompt(question: str, complexity: Any, question_class: Any) -> str:
    return (
        "You are answering a gastrointestinal endoscopy VQA benchmark question. "
        "Return only the final answer text. Match the expected answer type implied by the metadata. "
        "If the class is presence/status, answer with a concise yes/no/status phrase. "
        "If the class is count, answer with a number or short count phrase. "
        "If the class is location/color/type, answer with a concise location/color/type phrase. "
        "Avoid explanations and avoid adding unsupported details.\n\n"
        f"Question complexity: {complexity}\n"
        f"Question class list: {question_class}\n"
        f"Question: {question}\nAnswer:"
    )


def evidence_answer_prompt(question: str, complexity: Any, question_class: Any) -> str:
    return (
        "Inspect the endoscopic image and answer the question. "
        "Return exactly one compact JSON object and no markdown. "
        "The JSON keys must be: lesion_presence, lesion_type, lesion_count, location, "
        "instrument_presence, text_overlay_presence, abnormality_presence, uncertainty, final_answer. "
        "The final_answer must be concise and must directly answer the question in benchmark style. "
        "Use unknown when a visual field cannot be determined.\n\n"
        f"Question complexity: {complexity}\n"
        f"Question class list: {question_class}\n"
        f"Question: {question}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="step5_experiments/server_bundle_full")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="step5_experiments/results/qwen_strong_baselines.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modes", default="constrained,class,evidence")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    manifest_path = Path(args.manifest) if args.manifest else bundle / "test_full_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    if args.start > 0:
        manifest = manifest[args.start :]
    if args.limit > 0:
        manifest = manifest[: args.limit]

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
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
            row: dict[str, Any] = {
                "id": item["id"],
                "img_id": item["img_id"],
                "question": item["question"],
                "answer": item["answer"],
                "complexity": item["complexity"],
                "question_class": item["question_class"],
                "error": None,
            }
            try:
                if not image_path.exists() or image_path.stat().st_size <= 0:
                    raise FileNotFoundError(str(image_path))
                if "constrained" in modes:
                    raw = generate(model, processor, image_path, concise_prompt(item["question"]), 64)
                    row["qwen_constrained_raw"] = raw
                    row["qwen_constrained_answer"] = raw
                if "class" in modes:
                    raw = generate(
                        model,
                        processor,
                        image_path,
                        class_prompt(item["question"], item["complexity"], item["question_class"]),
                        64,
                    )
                    row["qwen_class_constrained_raw"] = raw
                    row["qwen_class_constrained_answer"] = raw
                if "evidence" in modes:
                    raw = generate(
                        model,
                        processor,
                        image_path,
                        evidence_answer_prompt(item["question"], item["complexity"], item["question_class"]),
                        192,
                    )
                    row["qwen_evidence_then_answer_raw"] = raw
                    row["qwen_evidence_then_answer_json"] = parse_json_object(raw)
                    row["qwen_evidence_then_answer"] = final_from_json(raw)
            except Exception as exc:
                row["error"] = repr(exc)
            row["elapsed_sec"] = round(time.time() - started, 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = "ERROR" if row.get("error") else "OK"
            preview = row.get("qwen_constrained_answer") or row.get("qwen_class_constrained_answer") or ""
            print(f"[{idx}/{len(manifest)}] {item['id']} {status} {row['elapsed_sec']}s {preview[:80]!r}", flush=True)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
