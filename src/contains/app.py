"""FastAPI app: one page + /api/label (text) + /api/photo (image -> local Gemma OCR -> same pipeline).

    uv run uvicorn contains.app:app --host 127.0.0.1 --port 8000

Never returns an error page from the API: failures degrade to rules-only with a banner.
Ingredient text is never logged.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from contains import ocr
from contains.pipeline import STATS, analyze
from contains.sampler import get_sampler

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("contains")


def _tokenizer_offline() -> None:
    """With CONTAINS_HF_OFFLINE=1 the wake path must not touch the Hugging Face hub: the tokenizer was
    baked into HF_HOME at build time (scripts/prefetch_tokenizer.py). If the cache is missing, say so loudly
    and use the network instead of breaking the model layer. Must run before transformers is imported."""
    if os.environ.get("CONTAINS_HF_OFFLINE") != "1":
        return
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    model = (os.environ.get("TINKER_BASE_MODEL") or "Qwen/Qwen3.5-4B").replace("/", "--")
    if (home / "hub" / f"models--{model}").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        log.info("tokenizer: baked cache found, Hugging Face hub disabled for the wake path")
    else:
        log.warning("tokenizer: baked cache MISSING at %s; falling back to the network", home)


_tokenizer_offline()

app = FastAPI(title="Contains", docs_url=None, redoc_url=None)
app.state.sampler = get_sampler()


@app.on_event("startup")
def _warm():
    """Build the Tinker client in the background so the first user request is not a cold start."""
    import threading
    w = getattr(app.state.sampler, "warm", None)
    if w:
        threading.Thread(target=lambda: _try(w), daemon=True).start()


def _try(fn):
    t0 = time.monotonic()
    try:
        fn()
        log.info("warm-up done in %.1fs", time.monotonic() - t0)
    except Exception as e:  # noqa: BLE001 - best effort; the first request retries and falls back with a banner
        log.warning("warm-up FAILED after %.1fs: %s: %s", time.monotonic() - t0, type(e).__name__, str(e).replace(chr(10), ' ')[:400])


def _safe(text: str) -> dict:
    try:
        return analyze(text, app.state.sampler)
    except Exception:  # noqa: BLE001 - last resort: still answer, rules-only
        from contains.sampler import RulesOnly
        res = analyze(text, RulesOnly())
        res["banner"] = "Something went wrong in the model layer; this output is rules-only. Confirm with your supplier."
        return res


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz():
    smp = app.state.sampler
    state = getattr(smp, "state", None)  # cold | warming | ready | failed for Tinker; None for the others
    return {"ok": True, "sampler": getattr(smp, "name", "?"), "model": state or "n/a",
            "photo": ocr.available(), "unsure_downgraded": STATS["unsure_downgraded"]}


@app.post("/api/label")
async def label(request: Request):
    try:
        body = await request.json()
        text = str(body.get("text", ""))
    except Exception:  # noqa: BLE001
        text = ""
    return JSONResponse(await run_in_threadpool(_safe, text))  # a cold first call may wait; never block the event loop


@app.post("/api/photo")
async def photo(request: Request):
    """Raw image bytes in the body (no multipart dependency)."""
    data = await request.body()
    try:
        text = await run_in_threadpool(ocr.extract_text, data)
    except ocr.OCRUnavailable:
        return JSONResponse({"empty": True, "local_only": True, "ocr_error": ocr.LOCAL_ONLY_MESSAGE,
                             "banner": None, "contains_line": "", "verify_line": "", "notes": [], "allergens": []})
    except ocr.OCRError as e:
        return JSONResponse({"empty": True, "ocr_error": f"Couldn't read the photo: {e}. Type or paste the list instead.",
                             "banner": None, "contains_line": "", "verify_line": "", "notes": [], "allergens": []})
    res = await run_in_threadpool(_safe, text)
    res["ocr_text"] = text
    return JSONResponse(res)


@app.exception_handler(Exception)
async def _any(request: Request, exc: Exception):  # never an error page
    return JSONResponse({"empty": True, "banner": "Temporary problem — please try again.", "contains_line": "",
                         "verify_line": "", "notes": [], "allergens": []}, status_code=200)
