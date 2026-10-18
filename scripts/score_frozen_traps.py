"""SINGLE scored run on the 62 FROZEN traps (data/verified/traps.jsonl). Run once; never again.

    uv run python -I scripts/score_frozen_traps.py

Systems: (1) shipped = LoRA + rules_product through contains.pipeline.analyze (incl. unsure
downgrade), (2) rules_naive alone, (3) Qwen3.5-4B zero-shot. Per allergen: caught (yes) /
CONFIDENT MISS (said no) / abstained (unsure) on gold positives; false alarms and abstentions on
gold negatives. Abstentions are never merged into recall. Refuses to run twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import load_env
from contains.labels import ALLERGENS
from contains.pipeline import analyze
from contains.rules import detect_naive
from contains.sampler import TinkerSampler
from score_m3 import sample_all

TRAPS = ROOT / "data" / "verified" / "traps.jsonl"
OUT = ROOT / "data" / "eval"
SENTINEL = OUT / "frozen_traps_SCORED.json"


def tally(rows, preds):
    per = {a: dict(pos=0, c=0, m=0, ab=0, fa=0, na=0) for a in ALLERGENS}
    miss, abst, fas = [], [], []
    for r, p in zip(rows, preds):
        for a in ALLERGENS:
            v, g = p[a], a in r["labels"]
            d = per[a]
            if g:
                d["pos"] += 1
                if v == "yes": d["c"] += 1
                elif v == "no": d["m"] += 1; miss.append((r["id"], a))
                else: d["ab"] += 1; abst.append((r["id"], a))
            else:
                if v == "yes": d["fa"] += 1; fas.append((r["id"], a))
                elif v == "unsure": d["na"] += 1
    return per, miss, abst, fas


def table(title, rows, preds):
    per, miss, abst, fas = tally(rows, preds)
    T = {k: sum(per[a][k] for a in ALLERGENS) for k in ("pos", "c", "m", "ab", "fa", "na")}
    f = lambda x, n: f"{x/n:.2f}" if n else "n/a"
    L = [f"\n**{title}** (n={len(rows)} rows, {T['pos']} gold positives)", "",
         "| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |", "|---|---|---|---|---|---|---|"]
    for a in ALLERGENS:
        d = per[a]
        L.append(f"| {a} | {d['pos']} | {d['c']} | {d['m']} | {d['ab']} | {d['fa']} | {d['na']} |")
    L.append(f"| **all** | {T['pos']} | {T['c']} ({f(T['c'], T['pos'])}) | **{T['m']}** ({f(T['m'], T['pos'])}) | {T['ab']} ({f(T['ab'], T['pos'])}) | {T['fa']} | {T['na']} |")
    L.append(f"\nconfident misses (id, allergen): {miss}\nabstained on gold positives: {abst}\nfalse alarms: {fas}")
    return "\n".join(L), T


def main() -> int:
    if SENTINEL.exists():
        print("REFUSING: frozen traps were already scored (see", SENTINEL, ")")
        return 1
    load_env()
    h = hashlib.sha256(TRAPS.read_bytes()).hexdigest()
    assert h.startswith("e62378cc"), f"traps hash changed: {h[:8]}"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    rules_sha = hashlib.sha256((ROOT / "src/contains/rules.py").read_bytes()).hexdigest()[:12]
    rows = [json.loads(ln) for ln in TRAPS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lora_path = os.environ["LORA_SAMPLER_PATH"]
    base = os.environ.get("TINKER_BASE_MODEL") or "Qwen/Qwen3.5-4B"
    out = [f"traps sha256 {h[:8]} · HEAD {head} · rules.py sha256 {rules_sha} · n={len(rows)}"]

    # 1) shipped: LoRA + rules_product, through the real pipeline (one sampler call per row)
    smp = TinkerSampler("tinker_lora", model_path=lora_path, base_model=base)
    smp.warm()
    ship, banners, lora_raw = [], 0, []
    for i, r in enumerate(rows, 1):
        res = analyze(r["text"], smp, timeout=120)
        banners += bool(res["banner"])
        ship.append({a["key"]: {"contains": "yes", "absent": "no", "unsure": "unsure"}[a["status"]] for a in res["allergens"]})
        if i % 20 == 0: print(f"  shipped {i}/{len(rows)}", flush=True)
    assert banners == 0, f"{banners} rows fell back to rules-only; model layer failed -> NOT a valid run"
    # 2) rules_naive
    naive = [{a: ("yes" if a in set(detect_naive(r["text"])) else "no") for a in ALLERGENS} for r in rows]
    # 3) Qwen zero-shot (same prompt/parser as baselines)
    qwen, i1, o1, bad = sample_all([r["text"] for r in rows], None)

    for name, preds in (("shipped_lora_rules", ship), ("rules_naive", naive), ("qwen_zeroshot", qwen)):
        (OUT / f"frozen_traps_{name}.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "gold": r["labels"], "pred": p})
                                                                  for r, p in zip(rows, preds)), encoding="utf-8")
    t1, _ = table("1. SHIPPED: LoRA + rules_product", rows, ship); out.append(t1)
    # footnote: with / without trap-021
    keep = [i for i, r in enumerate(rows) if r["id"] != "trap-021"]
    t1b, _ = table("1b. SHIPPED without trap-021 (post-hoc plural fix row)", [rows[i] for i in keep], [ship[i] for i in keep]); out.append(t1b)
    j = next(i for i, r in enumerate(rows) if r["id"] == "trap-021")
    out.append(f"\ntrap-021 ('{rows[j]['text']}') shipped verdicts: tree_nuts={ship[j]['tree_nuts']}, milk={ship[j]['milk']}")
    t2, _ = table("2. rules_naive alone (BASELINE)", rows, naive); out.append(t2)
    t3, _ = table(f"3. Qwen3.5-4B zero-shot alone (unparseable {bad})", rows, qwen); out.append(t3)
    cost = (i1 * 0.33 + o1 * 1.005) / 1e6
    out.append(f"\nTinker tokens: qwen zero-shot in {i1:,} out {o1:,}; shipped = 62 single sampler calls (~{62*300:,} in tok). est. total ≈ ${cost*2 + 0.01:.3f}")
    txt = "\n".join(out)
    print(txt)
    (OUT / "frozen_traps_report.md").write_text(txt, encoding="utf-8")
    SENTINEL.write_text(json.dumps({"scored": True, "head": head, "rules_sha": rules_sha}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
