#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "models/Qwen3.5-9B")


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    image_path = Path(os.environ.get("SMOKE_IMAGE_PATH", "outputs/red_smoke.jpg"))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color="red").save(image_path)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": "What is the dominant color in this image? Reply with one lowercase word only.",
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
        )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(strip_thinking(response))


if __name__ == "__main__":
    main()
