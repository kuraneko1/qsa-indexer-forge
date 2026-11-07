#!/usr/bin/env python3
"""Build an eight-image collage from the deterministic MM training cases."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qsa.mm_data import GENERATOR_VERSION, make_specs  # noqa: E402

OUT = ROOT / "docs" / "assets"
SELECTION = [
    ("code-ocr", 0), ("code-ocr", 1), ("code-ocr", 2),
    ("table-cell", 0), ("table-cell", 1), ("table-cell", 2),
    ("shape-count", 0), ("shape-count", 5),
]


def main() -> None:
    margin, gap, header, caption = 24, 20, 120, 64
    image_w, image_h = 640, 448
    width = margin * 2 + image_w * 4 + gap * 3
    height = header + margin + (image_h + caption) * 2 + gap + margin
    canvas = Image.new("RGB", (width, height), "#f1f4f8")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=32)
    text = ImageFont.load_default(size=20)
    draw.text((margin, 26), "Multimodal distillation: 8 synthetic samples", fill="#213047", font=title)
    draw.text((margin, 76), f"{GENERATOR_VERSION} | deterministic seeds | 3 OCR + 3 tables + 2 shape counts",
              fill="#69788c", font=text)

    manifest = []
    for slot, (kind, seed) in enumerate(SELECTION):
        spec = next(s for s in make_specs(seed) if s["id"] == kind)
        col, row = slot % 4, slot // 4
        x = margin + col * (image_w + gap)
        y = header + row * (image_h + caption + gap)
        canvas.paste(spec["image"], (x, y))
        draw.rectangle((x, y + image_h, x + image_w, y + image_h + caption), fill="#ffffff")
        draw.text((x + 14, y + image_h + 19), f"{kind}  |  seed={seed}", fill="#213047", font=text)
        manifest.append({
            "slot": slot + 1, "id": kind, "seed": seed,
            "question": spec["question"], "answer": spec["answer"],
            "answer_content": spec["meta"], "png_sha256": spec["png_sha256"],
            "image_size": spec["image_size"],
        })

    image_path = OUT / "multimodal-training-samples.png"
    json_path = OUT / "multimodal-training-samples.json"
    canvas.save(image_path)
    json_path.write_text(json.dumps({
        "generator": GENERATOR_VERSION, "samples": manifest,
        "canvas": list(canvas.size),
        "note": "Synthetic deterministic training images; no external image dataset or personal data.",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"write {image_path} ({canvas.width}x{canvas.height})")
    print(f"write {json_path} ({len(manifest)} samples)")


if __name__ == "__main__":
    main()
