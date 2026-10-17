"""M1-E — human review CLI for allergen labels.

Shows one row at a time (ingredient text + proposed labels) and accepts
keystrokes to confirm / toggle each allergen / skip. Verified rows are APPENDED
to data/verified/<name>.jsonl. Resumable: on restart it reads the output file and
skips rows whose id is already verified. It NEVER rewrites or auto-edits the
output file — it only appends confirmed rows.

Usage:
    uv run python scripts/review_cli.py data/test_gold_draft.jsonl
    uv run python scripts/review_cli.py data/traps_draft.jsonl --out traps

Keys per row:
    1..9   toggle an allergen (1=milk .. 9=sesame, FALCPA order)
    c      confirm & save the current label set
    n      none: clear all labels, then confirm & save
    s      skip (do not save; revisit next run)
    u      undo last toggle
    q      quit (progress already saved for confirmed rows)
    ?      show the allergen legend again
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.labels import ALLERGENS, DISPLAY_NAMES, contains_line

ROOT = Path(__file__).resolve().parent.parent
VERIFIED_DIR = ROOT / "data" / "verified"


def legend() -> str:
    parts = [f"{i}={DISPLAY_NAMES[a]}" for i, a in enumerate(ALLERGENS, 1)]
    return "  ".join(parts)


def row_id(row: dict) -> str:
    return str(row.get("id") or row.get("code") or row.get("text", ""))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            done.add(str(json.loads(ln)["id"]))
    return done


def proposed_labels(row: dict) -> set[str]:
    return set(row.get("proposed") or row.get("expected") or row.get("labels") or [])


def review_one(row: dict) -> tuple[str, set[str]]:
    """Return (action, labels). action in {'save','skip','quit'}."""
    labels = proposed_labels(row)
    history: list[str] = []
    text = row.get("text") or row.get("ingredients_text", "")
    print("\n" + "=" * 72)
    print(f"id={row_id(row)}  brand={row.get('brands', '')!r}")
    print(f"TEXT: {text}")
    if row.get("rationale"):
        print(f"(draft rationale: {row['rationale']})")
    while True:
        marks = " ".join(
            f"[{'x' if a in labels else ' '}]{i}{DISPLAY_NAMES[a][:4]}"
            for i, a in enumerate(ALLERGENS, 1)
        )
        print(f"\n  {marks}")
        print(f"  => {contains_line(labels) or 'Contains: (none)'}")
        try:
            key = input("  [1-9 toggle | c confirm | n none | s skip | u undo | q quit | ?]: ").strip().lower()
        except EOFError:
            return "quit", labels
        if key in {str(i) for i in range(1, 10)}:
            a = ALLERGENS[int(key) - 1]
            labels.symmetric_difference_update({a})
            history.append(a)
        elif key == "c":
            return "save", labels
        elif key == "n":
            labels = set()
            return "save", labels
        elif key == "s":
            return "skip", labels
        elif key == "u" and history:
            labels.symmetric_difference_update({history.pop()})
        elif key == "q":
            return "quit", labels
        elif key == "?":
            print("  legend:", legend())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="draft jsonl (e.g. data/test_gold_draft.jsonl)")
    ap.add_argument("--out", help="output name under data/verified/ (default: input stem minus _draft)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not in_path.exists():
        print(f"error: {in_path} not found")
        return 1

    name = args.out or in_path.stem.replace("_draft", "")
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VERIFIED_DIR / f"{name}.jsonl"

    rows = load_rows(in_path)
    done = already_done(out_path)
    todo = [r for r in rows if row_id(r) not in done]

    print(f"Review: {in_path.name} -> {out_path} (append-only, resumable)")
    print(f"{len(done)} already verified, {len(todo)} remaining of {len(rows)}.")
    print("Legend:", legend())

    saved = 0
    for i, row in enumerate(todo, 1):
        print(f"\n--- {i}/{len(todo)} (remaining this session) ---")
        action, labels = review_one(row)
        if action == "quit":
            print(f"\nQuit. Saved {saved} this session; progress is in {out_path}.")
            return 0
        if action == "skip":
            continue
        # APPEND only — never rewrite existing verified rows.
        ordered = [a for a in ALLERGENS if a in labels]
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "id": row_id(row),
                "text": row.get("text") or row.get("ingredients_text", ""),
                "labels": ordered,
                "verified": True,
                "source": in_path.name,
            }, ensure_ascii=False) + "\n")
        saved += 1

    print(f"\nDone. Verified {saved} new rows this session. Total in {out_path}: {len(done) + saved}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
