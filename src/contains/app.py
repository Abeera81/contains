"""FastAPI app: one page + /api/label (text) + /api/photo (image -> local Gemma OCR -> same pipeline).

    uv run uvicorn contains.app:app --host 127.0.0.1 --port 8000

Never returns an error page from the API: failures degrade to rules-only with a banner.
Ingredient text is never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from contains import ocr
from contains.pipeline import STATS, analyze
from contains.sampler import get_sampler

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

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
    try:
        fn()
    except Exception:  # noqa: BLE001 - warm-up is best effort; requests fall back with a banner
        pass


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
    return {"ok": True, "sampler": getattr(app.state.sampler, "name", "?"),
            "photo": ocr.available(), "unsure_downgraded": STATS["unsure_downgraded"]}


@app.post("/api/label")
async def label(request: Request):
    try:
        body = await request.json()
        text = str(body.get("text", ""))
    except Exception:  # noqa: BLE001
        text = ""
    return JSONResponse(_safe(text))


@app.post("/api/photo")
async def photo(request: Request):
    """Raw image bytes in the body (no multipart dependency)."""
    data = await request.body()
    try:
        text = ocr.extract_text(data)
    except ocr.OCRUnavailable:
        return JSONResponse({"empty": True, "local_only": True, "ocr_error": ocr.LOCAL_ONLY_MESSAGE,
                             "banner": None, "contains_line": "", "verify_line": "", "notes": [], "allergens": []})
    except ocr.OCRError as e:
        return JSONResponse({"empty": True, "ocr_error": f"Couldn't read the photo: {e}. Type or paste the list instead.",
                             "banner": None, "contains_line": "", "verify_line": "", "notes": [], "allergens": []})
    res = _safe(text)
    res["ocr_text"] = text
    return JSONResponse(res)


@app.exception_handler(Exception)
async def _any(request: Request, exc: Exception):  # never an error page
    return JSONResponse({"empty": True, "banner": "Temporary problem — please try again.", "contains_line": "",
                         "verify_line": "", "notes": [], "allergens": []}, status_code=200)
