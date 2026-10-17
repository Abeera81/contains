"""Shared helpers for M0 smoke tests. Never prints secrets.

These helpers are deliberately defensive: the Tinker SDK surface (method names,
response field names) is discovered at runtime by introspection so a small naming
difference between the docs and the installed version does not fail the smoke test
for the wrong reason.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Load .env from repo root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass


def require_api_key() -> str:
    key = os.environ.get("TINKER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TINKER_API_KEY is not set (check .env)")
    return key  # caller must never print this


def service_client():
    import tinker

    return tinker.ServiceClient()


def list_supported_models(sc) -> tuple[list[str], Any]:
    """Return (model_name_list, raw_response) via the capabilities method."""
    resp = sc.get_server_capabilities()

    # Find the list of models inside the response object, robust to field naming.
    models: list[str] = []
    candidates = []
    for attr in ("supported_models", "models", "model_names"):
        if hasattr(resp, attr):
            candidates = getattr(resp, attr)
            break
    for item in candidates or []:
        if isinstance(item, str):
            models.append(item)
        else:
            for attr in ("model_name", "name", "id", "model"):
                if hasattr(item, attr) and isinstance(getattr(item, attr), str):
                    models.append(getattr(item, attr))
                    break
    return models, resp


def find_model(names: list[str], *must_contain: str) -> str | None:
    """First model name containing all substrings (case-insensitive)."""
    needles = [s.lower() for s in must_contain]
    for n in names:
        low = n.lower()
        if all(s in low for s in needles):
            return n
    return None
