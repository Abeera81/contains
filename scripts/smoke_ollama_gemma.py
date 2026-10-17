"""M0 smoke test — OLLAMA GEMMA.

Sends one text prompt and one image prompt to gemma3:4b via the Ollama HTTP API.
The image is generated locally with PIL and contains the line:
    INGREDIENTS: wheat flour, whey powder, soy lecithin
Prints Gemma's transcription. If the model is still downloading or unreachable,
prints FAIL and exits non-zero WITHOUT blocking (per M0 instructions).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import load_env

MODEL = "gemma3:4b"
LABEL_TEXT = "INGREDIENTS: wheat flour, whey powder, soy lecithin"


def make_label_png() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 760, 200
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([8, 8, W - 8, H - 8], outline="black", width=3)
    d.text((28, 70), LABEL_TEXT, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    load_env()
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    # 1) text prompt
    try:
        r = httpx.post(
            f"{host}/api/generate",
            json={
                "model": MODEL,
                "prompt": "In one short sentence, what is a food allergen?",
                "stream": False,
                "options": {"num_predict": 40, "temperature": 0.0},
            },
            timeout=120.0,
        )
        r.raise_for_status()
        text_out = r.json().get("response", "").strip()
        print(f"[text] gemma3:4b -> {text_out!r}")
    except Exception as e:
        print(f"FAIL: OLLAMA GEMMA — text prompt failed ({type(e).__name__}: {e}); model may still be downloading")
        return 1

    # 2) image prompt (transcription)
    try:
        png = make_label_png()
        b64 = base64.b64encode(png).decode("ascii")
        t0 = time.time()
        r = httpx.post(
            f"{host}/api/generate",
            json={
                "model": MODEL,
                "prompt": "Transcribe the ingredient text in this image exactly. Output only the text.",
                "images": [b64],
                "stream": False,
                "options": {"num_predict": 80, "temperature": 0.0},
            },
            timeout=180.0,
        )
        r.raise_for_status()
        dt = time.time() - t0
        transcription = r.json().get("response", "").strip()
        print(f"[image] gemma3:4b transcription ({dt:.1f}s): {transcription!r}")
    except Exception as e:
        print(f"FAIL: OLLAMA GEMMA — image prompt failed ({type(e).__name__}: {e})")
        return 1

    low = transcription.lower()
    hits = [w for w in ("wheat", "whey", "soy", "lecithin") if w in low]
    if len(hits) >= 2:
        print(f"PASS: OLLAMA GEMMA — transcription recovered {hits} from the label image")
        return 0
    print(f"FAIL: OLLAMA GEMMA — transcription did not clearly recover the label (matched {hits})")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1200:])
        print(f"FAIL: OLLAMA GEMMA — {type(e).__name__}: {e}")
        sys.exit(1)
