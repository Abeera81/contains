"""Photo -> ingredient text with Gemma 3 4B running locally in Ollama (nothing leaves the machine)."""

from __future__ import annotations

import base64
import io
import os

PROMPT = ("Transcribe the ingredient list printed in this photo of a food label. Output ONLY the ingredient "
          "text, exactly as printed, on one line. If there is no ingredient list, output NONE.")
MAX_BYTES = 8 * 1024 * 1024


class OCRError(RuntimeError):
    pass


def extract_text(image_bytes: bytes) -> str:
    if not image_bytes or len(image_bytes) > MAX_BYTES:
        raise OCRError("image missing or larger than 8 MB")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    except Exception as e:  # noqa: BLE001
        raise OCRError("not a readable image") from e
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        r = httpx.post(f"{host}/api/generate", json={
            "model": os.environ.get("OLLAMA_OCR_MODEL", "gemma3:4b"), "prompt": PROMPT, "stream": False,
            "images": [base64.b64encode(buf.getvalue()).decode()],
            "options": {"num_predict": 400, "temperature": 0.0}}, timeout=240.0)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise OCRError("local OCR (Ollama gemma3:4b) is unavailable") from e
    text = (r.json().get("response") or "").strip()
    if not text or text.upper().startswith("NONE"):
        raise OCRError("no ingredient list found in the photo")
    return text
