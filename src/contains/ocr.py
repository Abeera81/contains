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


class OCRUnavailable(OCRError):
    """Ollama (or the gemma model) is not reachable: photo reading is local-only."""


README_URL = "https://github.com/Abeera81/contains#photo-reading-local-only"
LOCAL_ONLY_MESSAGE = ("Photo reading is local-only: it runs Gemma 3 4B on your own computer, which this hosted "
                      "version doesn't have. Paste the ingredient text instead, or run Contains locally — "
                      f"see the README: {README_URL}")
_avail = {"t": 0.0, "ok": False}


def available(ttl: float = 30.0) -> bool:
    """True if Ollama answers and has the OCR model. Cached; fast (1.5 s connect budget)."""
    import time
    if time.monotonic() - _avail["t"] < ttl:
        return _avail["ok"]
    ok = False
    if os.environ.get("PHOTO_ENABLED", "").strip() != "0":
        try:
            import httpx
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
            r = httpx.get(f"{host}/api/tags", timeout=1.5)
            want = os.environ.get("OLLAMA_OCR_MODEL", "gemma3:4b")
            ok = r.status_code == 200 and any(m.get("name", "").startswith(want.split(":")[0])
                                              for m in r.json().get("models", []))
        except Exception:  # noqa: BLE001
            ok = False
    _avail["t"], _avail["ok"] = time.monotonic(), ok
    return ok


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
    if not available():
        raise OCRUnavailable(LOCAL_ONLY_MESSAGE)
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        r = httpx.post(f"{host}/api/generate", json={
            "model": os.environ.get("OLLAMA_OCR_MODEL", "gemma3:4b"), "prompt": PROMPT, "stream": False,
            "images": [base64.b64encode(buf.getvalue()).decode()],
            "options": {"num_predict": 400, "temperature": 0.0}}, timeout=240.0)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise OCRUnavailable(LOCAL_ONLY_MESSAGE) from e
    except Exception as e:  # noqa: BLE001
        raise OCRError("local OCR failed or timed out") from e
    text = (r.json().get("response") or "").strip()
    if not text or text.upper().startswith("NONE"):
        raise OCRError("no ingredient list found in the photo")
    return text
