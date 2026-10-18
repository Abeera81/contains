# RESULTS

Every number here comes from a script run in this repo. If something wasn't run it says NOT RUN.

## M0 — go/no-go smoke tests (2026-10-17)

Environment: Windows 11, Python 3.12 (uv), tinker SDK 0.34.0. Tinker IDs discovered via
`ServiceClient.get_server_capabilities()` (37 models returned).

| Model | Display name | Tinker ID passed | trainable | sampleable | ctx |
|---|---|---|---|---|---|
| LoRA base | Qwen/Qwen3.5-4B | `Qwen/Qwen3.5-4B` | yes | yes | 65536 |
| Big baseline | gpt-oss-120b | `openai/gpt-oss-120b` | yes | yes | 32768 |

### TINKER TRAIN — PASS
`uv run python scripts/smoke_tinker_train.py`
- Renderer: tokenizer `apply_chat_template` fallback (tinker-cookbook not installable on Windows; see note).
- Toy batch: 2 examples, 172 tokens (20 completion tokens).
- 1× forward_backward + optim_step OK in ~17 s.
- Loss (`metrics['loss:sum']` over completion tokens): 53.3774 → mean per completion token: **2.6689**.
- Cost: Tinker returns no per-call $ field; estimate = 172 tok × $0.737/1M = **$0.000127**.

### TINKER SAMPLE — PASS
`uv run python scripts/smoke_tinker_sample.py` (max_tokens=32, temperature=0)
- Qwen3.5-4B base: 37 prompt + 32 completion tokens, latency ~21.6 s (cold).
- gpt-oss-120b: 92 prompt + 32 completion tokens, latency ~2.06 s.
- Both returned non-empty output.
- Cost: 2 × ~32 completion tokens; negligible (<$0.001). No per-call $ field returned.

### OLLAMA GEMMA — PASS
`uv run python scripts/smoke_ollama_gemma.py` (gemma3:4b via Ollama)
- Text prompt: coherent 1-sentence answer.
- Image prompt (PIL-generated label): transcription = `INGREDIENTS: wheat flour, whey powder, soy lecithin`
  (exact), ~50 s. Recovered wheat/whey/soy/lecithin.

### OPEN FOOD FACTS — PARTIAL (readability gate PASS; filtered enumeration DEFERRED to M1)
`uv run python scripts/smoke_off_parquet.py`
- Source: `https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet`
  (duckdb httpfs range requests; single ~8 GB file).
- Columns: **111**. Confirmed filter columns: `ingredients_text STRUCT(lang,text)[]`,
  `allergens_tags VARCHAR[]`, `countries_tags VARCHAR[]`.
- Total rows: **4,815,797** (DESCRIBE ~28 s, COUNT ~2.4 s).
- Filtered count + 3 samples: **HTTP 429** (HF unauthenticated range-request rate limit on the
  heavy multi-column scan). NOT RUN to completion.
- Resolution: M1 builds `data/raw/off_working.parquet` from the HF `refs/convert/parquet` shards
  (needed columns + filtered rows); the smoke script already prefers that local file.

### Tinker spend so far
Only the train + sample smoke tests billed. Estimated total **< $0.01** (training est. $0.000127 +
two 32-token samples). No per-call cost field is exposed by the SDK; estimates use the published
$0.737 / 1M training-token rate.

### Note — tinker-cookbook on Windows
`tinker-cookbook` pins `tml-renderers==0.2.0`, which has no Windows wheel (Linux + macOS only), so it
is an optional `[train]` extra installed on Render/Linux and for real training. On Windows the smoke
test builds supervised Datums with the tokenizer chat template (what the cookbook renderer wraps).

## M1 — data (2026-10-17)

> **SUPERSEDED NUMBERS.** The figures below are as first built. CURRENT data (after de-doubling,
> the language filter and the tightened declaration detector, see M2): working set **35,193 rows**;
> splits train 28,154 / dev 3,520 / test 3,519 (zero brand overlap); rows with >=1 label 30,591 (86.9%).
> Current class balance: milk 15,922 (45.2%) · soy 13,905 (39.5%) · wheat 11,785 (33.5%) · egg 4,335 (12.3%)
> · tree_nuts 4,310 (12.2%) · peanuts 2,403 (6.8%) · sesame 1,626 (4.6%) · fish 938 (2.7%)
> · crustacean_shellfish 332 (0.9%). Gold = 100 reviewed rows (95 after the language filter), not 200.

Source: single HF file `openfoodfacts/product-database@refs/convert/parquet/default/food/0000.parquet`
(~7.95 GB, 4,815,797 rows, 4703 row groups). No small file-shards exist on either
revision, so sampling is at the ROW-GROUP level via authenticated range reads.

Command: `uv run python -I scripts/build_working_set.py` then `scripts/clean_working_set.py`
- Probe: 49 groups spread every 97 -> pass-rate 8.231%
- Plan: stride 7 -> 672 groups spread across all 4703 (indices 0,7,14 .. 4697)
- Collected 40,880 filtered rows; dedupe dropped 0 by code, 3,709 by normalized text
- HTML clean pass: stripped markup from 28,710 rows; dropped 35 (length band) + 968 (new dupes)
- **Final working set: 36,168 rows, 4.4 MB -> data/raw/off_working.parquet**

Country distribution (target countries, rows may count in >1):
  US 28,861 (77.0%) · UK 4,769 (12.7%) · CA 1,865 (5.0%) · AU 1,288 (3.4%) · IE 713 (1.9%)
  -> US-dominant (77%) but below the 85% skew-warning threshold; not flagged as badly skewed.

Class balance (per-allergen positive counts over 36,168 rows), `scripts/make_splits.py`:
  milk 16,346 (45.2%) · soy 14,144 (39.1%) · wheat 11,814 (32.7%) · tree_nuts 4,432 (12.3%)
  · egg 4,416 (12.2%) · peanuts 2,469 (6.8%) · sesame 1,676 (4.6%) · fish 995 (2.8%)
  · crustacean_shellfish 341 (0.9%)
  rows with >=1 label: 31,354 (86.7%); rows with no label: 4,814 (13.3%)

Brand-disjoint splits (seed 1337, 80/10/10 by row count):
  train 28,928 (80.0%, 5,776 brands) · dev 3,623 (10.0%, 1,150 brands) · test 3,617 (10.0%, 1,282 brands)
  Brand-overlap check: PASS (0 shared brands; 9,827 empty-brand rows all in train).

Traps: data/traps_draft.jsonl — 62 rows (50 hidden-allergen, 12 clean negatives), all verify:false.
Gold: data/test_gold_draft.jsonl — 200 rows sampled (seed 20261017) from the test split.
## M2 — zero-shot baselines on DEV (2026-10-17)

Working-set repair (before M2): collapsed 34,832 doubled ingredient lines (duplicate
STRUCT entries) and added a has_declaration flag (36,026 rows).
Language filter (scripts/filter_language.py): 833 non-English rows (2.3%; fr 421, de 136,
es 65, ro 38, it 32 ...) removed. Causes: OFF's `en` slot sometimes holds foreign text /
en_text falls back to main|xx entries, and the country filter is "any-of" (8.1% of rows
also list a non-target country). langdetect on lowercased text + English-vocab guard.
Some short true-English rows are dropped too (conservative). FINAL working set 35,193 rows;
splits train 28,154 / dev 3,520 / test 3,519, zero brand overlap. ALL baselines below were
re-run on this set (earlier runs on the 36,026 set were discarded).

**Declaration leakage** (ingredient text embeds an explicit allergen statement — "may contain",
"allergy advice", "allergens:", facility statements, and "contains: <allergen>" where the clause names
an allergen and is not a composition line such as "contains: cocoa solids 30% min"). CORRECTED rates
with the tightened detector (`uv run python -I scripts/refresh_declaration_flag.py`):
  working set **12.2%** (4,281/35,193) · train 12.2% (3,427) · dev 13.5% (474) · test 10.8% (380) ·
  300-subset 10.0% (30 declared, **270 undeclared**) · traps 0/105 (frozen 62 + traps_dev 43) clean.
The earlier figure (11.4-11.7%) was NOT an overstatement on net: tightening removed 98 composition false
positives ("Milk chocolate contains: cocoa solids...") but ADDED 271 real declarations the old regex
missed (contains sulphites/mustard/celery/dairy/..., 'Barley Flour (contains Gluten)'). Known residual:
garbled OCR-style lines with a '%' in the clause can still be missed. The earlier per-backend
UNDECLARED blocks and four-way table used the old flag and were replaced by the tables below.

Every baseline is reported BOTH ways: on ALL rows, and on UNDECLARED-TEXT rows only.
The undeclared number is the honest one — it is the case the product exists for (a
baker typing an ingredient list with no allergen statement to copy from).

CAVEAT (next to the numbers, not a footnote): dev/test gold is OFF-tag-derived, and
rules_naive shares ingredient vocabulary with those tags, so rules_naive's dev F1 is
inflated relative to real-world text. The frozen trap set (scored once at end of M3)
is the real test. rules_product is NOT a baseline (it shares authorship with the traps).


#### cmd: `uv run python scripts/eval_baselines.py rules --full`
### rules_naive (BASELINE) [dev-full-3520] ALL ROWS  (n=3520)
```
allergen                supp   prec    rec     f1
milk                    1512  0.940  0.978  0.958
egg                      356  0.846  0.955  0.897
fish                     100  0.781  0.890  0.832
crustacean_shellfish      43  0.784  0.930  0.851
tree_nuts                440  0.846  0.236  0.369
peanuts                  285  0.824  0.968  0.890
wheat                   1099  0.802  0.995  0.888
soy                     1257  0.900  0.986  0.941
sesame                   202  0.854  0.985  0.915
MACRO                         0.842  0.881  0.838

exact_match=0.749  unsure_rate=0.000
p50_latency_ms=0.2
wall_s=0.9
in_tok=0
out_tok=0
usd_per_1k=n/a (free)
```

#### cmd: `uv run python scripts/eval_baselines.py rules --subset`
### rules_naive (BASELINE) [dev-subset-300] ALL ROWS  (n=300)
```
allergen                supp   prec    rec     f1
milk                     118  0.921  0.983  0.951
egg                       30  0.800  0.933  0.862
fish                       6  0.750  1.000  0.857
crustacean_shellfish       5  0.800  0.800  0.800
tree_nuts                 39  0.857  0.308  0.453
peanuts                   18  0.818  1.000  0.900
wheat                     98  0.829  0.990  0.902
soy                      107  0.922  1.000  0.960
sesame                    24  0.821  0.958  0.885
MACRO                         0.835  0.886  0.841

exact_match=0.760  unsure_rate=0.000
p50_latency_ms=0.2
wall_s=0.1
in_tok=0
out_tok=0
usd_per_1k=n/a (free)
```

#### cmd: `uv run python scripts/eval_baselines.py gemma --subset`
### gemma [dev-subset-300] ALL ROWS  (n=300)
```
allergen                supp   prec    rec     f1
milk                     118  0.786  0.966  0.867
egg                       30  0.492  0.967  0.652
fish                       6  0.071  1.000  0.132
crustacean_shellfish       5  0.444  0.800  0.571
tree_nuts                 39  0.351  1.000  0.520
peanuts                   18  0.360  1.000  0.529
wheat                     98  0.681  1.000  0.810
soy                      107  0.883  0.991  0.934
sesame                    24  0.622  0.958  0.754
MACRO                         0.521  0.965  0.641

exact_match=0.397  unsure_rate=0.003
p50_latency_ms=13607.4
wall_s=4217.9
in_tok=79929
out_tok=13904
usd_per_1k=n/a (free)
```

> NOTE: first Qwen/gpt-oss attempts scored 100% unparseable (Qwen3.5 thinking mode and gpt-oss
> analysis channel consumed the 64-token budget); discarded as harness errors, not model results.
> Fixed: Qwen enable_thinking=False; gpt-oss reasoning_effort=low, max_tokens=512, final channel parsed.

#### cmd: `uv run python scripts/eval_baselines.py qwen --subset`
### qwen [dev-subset-300] ALL ROWS  (n=300)
```
allergen                supp   prec    rec     f1
milk                     118  0.935  0.975  0.954
egg                       30  0.806  0.833  0.820
fish                       6  0.714  0.833  0.769
crustacean_shellfish       5  1.000  0.800  0.889
tree_nuts                 39  0.972  0.897  0.933
peanuts                   18  1.000  1.000  1.000
wheat                     98  0.980  0.990  0.985
soy                      107  0.938  0.981  0.959
sesame                    24  0.880  0.917  0.898
MACRO                         0.914  0.914  0.912

exact_match=0.873  unsure_rate=0.005
p50_latency_ms=2302.4
wall_s=734.7
in_tok=82569
out_tok=14400
usd_per_1k=$0.139  (total $0.0417)
```

#### cmd: `uv run python scripts/eval_baselines.py gptoss --subset`
### gptoss [dev-subset-300] ALL ROWS  (n=300)
```
allergen                supp   prec    rec     f1
milk                     118  0.944  0.992  0.967
egg                       30  0.800  0.800  0.800
fish                       6  0.625  0.833  0.714
crustacean_shellfish       5  1.000  0.800  0.889
tree_nuts                 39  0.973  0.923  0.947
peanuts                   18  1.000  1.000  1.000
wheat                     98  0.960  0.990  0.975
soy                      107  0.929  0.981  0.955
sesame                    24  0.880  0.917  0.898
MACRO                         0.901  0.915  0.905

exact_match=0.873  unsure_rate=0.077
p50_latency_ms=3274.8
wall_s=886.3
in_tok=95585
out_tok=51742
usd_per_1k=$0.250  (total $0.0750)
```


### FOUR-WAY TABLE — dev subset (300, seed 300), CORRECTED declaration flag
cmd: `uv run python -I scripts/summarize_baselines.py` (reads data/eval/*_dev-subset-300.jsonl, flags from data/splits/dev.jsonl)
Per-allergen RECALL stays visible; support = gold positives in that row set. fish (3-6) and crustacean_shellfish (2-5)
are far too thin for a reliable per-class number (crustacean is also thin in test: 20 rows).
CAVEAT: dev gold is OFF-tag-derived and rules_naive shares its vocabulary, so rules_naive is flattered here.
Latency p50 / $ per 1k labels: rules 0.2 ms free · gemma3:4b 13,607 ms local free · qwen3.5-4b 2,302 ms $0.139 ·
gpt-oss-120b 3,275 ms $0.250 (cmds: eval_baselines.py <backend> --subset). Tinker spend $0.117.

### ALL ROWS (n=300)
| model | macro P | macro R | macro F1 | exact | unsure |
|---|---|---|---|---|---|
| rules_naive | 0.835 | 0.886 | 0.841 | 0.760 | 0.000 |
| gemma3:4b | 0.521 | 0.965 | 0.641 | 0.397 | 0.003 |
| qwen3.5-4b | 0.914 | 0.914 | 0.912 | 0.873 | 0.005 |
| gpt-oss-120b | 0.901 | 0.915 | 0.905 | 0.873 | 0.077 |

per-allergen RECALL / precision (support = gold positives)
| allergen | support | rules_naive | gemma3:4b | qwen3.5-4b | gpt-oss-120b |
|---|---|---|---|---|---|
| milk | 118 | R 0.98 / P 0.92 | R 0.97 / P 0.79 | R 0.97 / P 0.93 | R 0.99 / P 0.94 |
| egg | 30 | R 0.93 / P 0.80 | R 0.97 / P 0.49 | R 0.83 / P 0.81 | R 0.80 / P 0.80 |
| fish | 6 | R 1.00 / P 0.75 | R 1.00 / P 0.07 | R 0.83 / P 0.71 | R 0.83 / P 0.62 |
| crustacean_shellfish | 5 | R 0.80 / P 0.80 | R 0.80 / P 0.44 | R 0.80 / P 1.00 | R 0.80 / P 1.00 |
| tree_nuts | 39 | R 0.31 / P 0.86 | R 1.00 / P 0.35 | R 0.90 / P 0.97 | R 0.92 / P 0.97 |
| peanuts | 18 | R 1.00 / P 0.82 | R 1.00 / P 0.36 | R 1.00 / P 1.00 | R 1.00 / P 1.00 |
| wheat | 98 | R 0.99 / P 0.83 | R 1.00 / P 0.68 | R 0.99 / P 0.98 | R 0.99 / P 0.96 |
| soy | 107 | R 1.00 / P 0.92 | R 0.99 / P 0.88 | R 0.98 / P 0.94 | R 0.98 / P 0.93 |
| sesame | 24 | R 0.96 / P 0.82 | R 0.96 / P 0.62 | R 0.92 / P 0.88 | R 0.92 / P 0.88 |

### UNDECLARED ONLY (honest) (n=270)
| model | macro P | macro R | macro F1 | exact | unsure |
|---|---|---|---|---|---|
| rules_naive | 0.851 | 0.898 | 0.844 | 0.778 | 0.000 |
| gemma3:4b | 0.515 | 0.985 | 0.633 | 0.415 | 0.002 |
| qwen3.5-4b | 0.904 | 0.976 | 0.934 | 0.893 | 0.000 |
| gpt-oss-120b | 0.894 | 0.982 | 0.928 | 0.896 | 0.080 |

per-allergen RECALL / precision (support = gold positives)
| allergen | support | rules_naive | gemma3:4b | qwen3.5-4b | gpt-oss-120b |
|---|---|---|---|---|---|
| milk | 105 | R 0.98 / P 0.93 | R 0.96 / P 0.81 | R 0.98 / P 0.94 | R 1.00 / P 0.95 |
| egg | 22 | R 0.91 / P 0.87 | R 0.95 / P 0.48 | R 0.91 / P 0.83 | R 0.91 / P 0.83 |
| fish | 3 | R 1.00 / P 0.60 | R 1.00 / P 0.04 | R 1.00 / P 0.60 | R 1.00 / P 0.50 |
| crustacean_shellfish | 2 | R 1.00 / P 0.67 | R 1.00 / P 0.33 | R 1.00 / P 1.00 | R 1.00 / P 1.00 |
| tree_nuts | 32 | R 0.25 / P 0.89 | R 1.00 / P 0.33 | R 0.94 / P 0.97 | R 0.97 / P 0.97 |
| peanuts | 17 | R 1.00 / P 1.00 | R 1.00 / P 0.40 | R 1.00 / P 1.00 | R 1.00 / P 1.00 |
| wheat | 83 | R 0.99 / P 0.83 | R 1.00 / P 0.67 | R 1.00 / P 0.99 | R 1.00 / P 0.97 |
| soy | 92 | R 1.00 / P 0.92 | R 0.99 / P 0.88 | R 1.00 / P 0.93 | R 1.00 / P 0.92 |
| sesame | 23 | R 0.96 / P 0.96 | R 0.96 / P 0.69 | R 0.96 / P 0.88 | R 0.96 / P 0.92 |

### rules_naive, FULL dev (cmd: eval_baselines.py rules --full)
ALL ROWS n=3520: macro P 0.842 R 0.881 F1 0.838 exact 0.749
  recall (support): milk 0.98 (1512), egg 0.96 (356), fish 0.89 (100), crustacean_shellfish 0.93 (43), tree_nuts 0.24 (440), peanuts 0.97 (285), wheat 1.00 (1099), soy 0.99 (1257), sesame 0.99 (202)
UNDECLARED ONLY (honest) n=3046: macro P 0.885 R 0.877 F1 0.857 exact 0.766
  recall (support): milk 0.97 (1275), egg 0.95 (297), fish 0.86 (77), crustacean_shellfish 0.97 (32), tree_nuts 0.22 (372), peanuts 0.96 (245), wheat 1.00 (903), soy 0.98 (1085), sesame 0.98 (169)

### traps_dev headroom check (n=40 scored: 30 hidden + 10 clean; 3 sanity rows unscored)
cmd: `uv run python -I scripts/build_traps_dev.py && uv run python -I scripts/score_traps_dev.py`
DEV trap set (different compounds from the frozen 62; peanuts has 0 positives here). Per-allergen RECALL (support):

| allergen | support | rules_naive | qwen3.5-4b | gpt-oss-120b |
|---|---|---|---|---|
| milk | 11 | 0.45 (P 1.00) | 0.73 (P 1.00) | 1.00 (P 0.92) |
| egg | 8 | 0.25 (P 1.00) | 1.00 (P 1.00) | 1.00 (P 1.00) |
| fish | 3 | 0.33 (P 1.00) | 1.00 (P 0.60) | 1.00 (P 1.00) |
| crustacean_shellfish | 3 | 0.67 (P 1.00) | 0.67 (P 0.67) | 1.00 (P 1.00) |
| tree_nuts | 8 | 0.25 (P 1.00) | 0.50 (P 1.00) | 0.75 (P 1.00) |
| peanuts | 0 | 0.00 (P 0.00) | 0.00 (P 0.00) | 0.00 (P 0.00) |
| wheat | 11 | 0.09 (P 1.00) | 0.45 (P 1.00) | 1.00 (P 1.00) |
| soy | 3 | 0.67 (P 1.00) | 0.67 (P 0.67) | 1.00 (P 1.00) |
| sesame | 5 | 0.40 (P 1.00) | 1.00 (P 0.83) | 0.80 (P 1.00) |

| overall | recall (micro) | macro F1 | exact-match | unsure rate | missed (said no) | abstained | false alarms |
|---|---|---|---|---|---|---|---|
| rules_naive | 0.33 (17/52) | 0.474 | 0.400 | 0.000 | 35 | 0 | 0 |
| qwen3.5-4b | 0.71 (37/52) | 0.681 | 0.575 | 0.000 | 15 | 0 | 5 |
| gpt-oss-120b | 0.94 (49/52) | 0.856 | 0.925 | 0.033 | 0 | 3 | 1 |

rules_naive missed (id, allergen): [('tdev-001', 'crustacean_shellfish'), ('tdev-002', 'tree_nuts'), ('tdev-002', 'sesame'), ('tdev-003', 'wheat'), ('tdev-004', 'milk'), ('tdev-004', 'egg'), ('tdev-005', 'sesame'), ('tdev-006', 'milk'), ('tdev-006', 'egg'), ('tdev-007', 'milk'), ('tdev-008', 'milk'), ('tdev-010', 'milk'), ('tdev-010', 'egg'), ('tdev-010', 'wheat'), ('tdev-011', 'tree_nuts'), ('tdev-011', 'wheat'), ('tdev-012', 'wheat'), ('tdev-013', 'fish'), ('tdev-013', 'wheat'), ('tdev-014', 'wheat'), ('tdev-015', 'wheat'), ('tdev-016', 'fish'), ('tdev-019', 'wheat'), ('tdev-019', 'sesame'), ('tdev-020', 'tree_nuts'), ('tdev-021', 'tree_nuts'), ('tdev-022', 'tree_nuts'), ('tdev-025', 'milk'), ('tdev-025', 'egg'), ('tdev-026', 'egg'), ('tdev-027', 'soy'), ('tdev-028', 'egg'), ('tdev-028', 'wheat'), ('tdev-029', 'tree_nuts'), ('tdev-029', 'wheat')]
qwen3.5-4b missed (id, allergen): [('tdev-001', 'crustacean_shellfish'), ('tdev-002', 'tree_nuts'), ('tdev-006', 'milk'), ('tdev-008', 'milk'), ('tdev-010', 'wheat'), ('tdev-011', 'wheat'), ('tdev-012', 'wheat'), ('tdev-013', 'wheat'), ('tdev-014', 'wheat'), ('tdev-018', 'milk'), ('tdev-020', 'tree_nuts'), ('tdev-021', 'tree_nuts'), ('tdev-022', 'tree_nuts'), ('tdev-027', 'soy'), ('tdev-029', 'wheat')]
gpt-oss-120b missed (id, allergen): []


### M3 prep — training data (cmds beside numbers)
- Declarations STRIPPED from train inputs (labels kept): `uv run python -I scripts/strip_train_declarations.py --write`
  -> 3,427 of 28,154 train rows affected (12.2%); 0 still flagged after stripping.
- DROPPED from training: 768 rows = 757 where a label has no rule-visible evidence left after stripping (we destroyed
  the evidence ourselves; NOT converted to 'unsure') + 30 under 15 chars (overlap counted once). Clean train = 27,386 rows
  (data/splits/train_clean.jsonl).
- Deliberate ABSTENTION slice (`uv run python -I scripts/build_abstention_slice.py`, seed 4242) -> data/splits/train_abstain.jsonl:
  805 rows = 600 abstain (formulation-dependent: unnamed lecithin, natural flavoring, unspecified vegetable/cooking oil,
  vanaspati/banaspati, plain chocolate, nut spreads, bbq/stir-fry sauces, salad dressing, curry paste, bouillon/seasoning mix;
  1,310 'unsure' cells) + 205 contrast rows (named sources: sunflower/soy lecithin, canola/soybean/peanut oil, milk chocolate,
  named flavors; definite answers) so the model learns unnamed->unsure, named->decide. Built from TRAIN lines only.

## M3 LoRA v1 (dev + traps_dev only; test and frozen traps NOT RUN)
cmd: `uv run python -I scripts/build_m3_train.py && uv run python -I scripts/train_m3.py probe && uv run python -I scripts/train_m3.py probe --low && uv run python -I scripts/train_m3.py train --lr 3e-5 --epochs 1 && uv run python -I scripts/score_m3.py`
Train: 5,755 rows, 359 steps, rank 16, LR 3e-5, 1.88M tokens, ~$1.46 est (probe ~$0.53, scoring ~$0.50 est). Dev loss 0.0201 -> 0.0199.

### M3 LoRA (Qwen3.5-4B r16) — DEV, 3520 rows, unparseable 0; wall 152s

**DEV full, ALL rows** (n=3520)

| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |
|---|---|---|---|---|---|---|---|---|---|
| milk | 1512 | 1489 | 21 | 2 | 0.98 | 0.01 | 0.00 | 62 | 16 |
| egg | 356 | 332 | 24 | 0 | 0.93 | 0.07 | 0.00 | 51 | 4 |
| fish | 100 | 98 | 2 | 0 | 0.98 | 0.02 | 0.00 | 24 | 1 |
| crustacean_shellfish | 43 | 42 | 1 | 0 | 0.98 | 0.02 | 0.00 | 14 | 1 |
| tree_nuts | 440 | 412 | 28 | 0 | 0.94 | 0.06 | 0.00 | 42 | 3 |
| peanuts | 285 | 272 | 13 | 0 | 0.95 | 0.05 | 0.00 | 44 | 2 |
| wheat | 1099 | 1094 | 5 | 0 | 1.00 | 0.00 | 0.00 | 41 | 4 |
| soy | 1257 | 1240 | 16 | 1 | 0.99 | 0.01 | 0.00 | 125 | 14 |
| sesame | 202 | 196 | 6 | 0 | 0.97 | 0.03 | 0.00 | 24 | 3 |
| **all** | 5294 | 5175 | **116** | 3 | 0.98 | 0.02 | 0.00 | 427 | 48 |

macro-F1 (unsure=not flagged) 0.920 · exact-match 0.891 · unsure-cell rate 0.0016

**DEV full, UNDECLARED only (honest)** (n=3046)

| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |
|---|---|---|---|---|---|---|---|---|---|
| milk | 1275 | 1255 | 18 | 2 | 0.98 | 0.01 | 0.00 | 35 | 8 |
| egg | 297 | 279 | 18 | 0 | 0.94 | 0.06 | 0.00 | 27 | 0 |
| fish | 77 | 75 | 2 | 0 | 0.97 | 0.03 | 0.00 | 18 | 0 |
| crustacean_shellfish | 32 | 31 | 1 | 0 | 0.97 | 0.03 | 0.00 | 9 | 0 |
| tree_nuts | 372 | 354 | 18 | 0 | 0.95 | 0.05 | 0.00 | 24 | 0 |
| peanuts | 245 | 236 | 9 | 0 | 0.96 | 0.04 | 0.00 | 18 | 1 |
| wheat | 903 | 898 | 5 | 0 | 0.99 | 0.01 | 0.00 | 32 | 4 |
| soy | 1085 | 1072 | 12 | 1 | 0.99 | 0.01 | 0.00 | 81 | 6 |
| sesame | 169 | 166 | 3 | 0 | 0.98 | 0.02 | 0.00 | 8 | 0 |
| **all** | 4455 | 4366 | **86** | 3 | 0.98 | 0.02 | 0.00 | 252 | 19 |

macro-F1 (unsure=not flagged) 0.938 · exact-match 0.910 · unsure-cell rate 0.0008

**DEV subset-300 (same rows as the baseline table), UNDECLARED only** (n=270)

| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |
|---|---|---|---|---|---|---|---|---|---|
| milk | 105 | 105 | 0 | 0 | 1.00 | 0.00 | 0.00 | 6 | 1 |
| egg | 22 | 20 | 2 | 0 | 0.91 | 0.09 | 0.00 | 4 | 0 |
| fish | 3 | 3 | 0 | 0 | 1.00 | 0.00 | 0.00 | 3 | 0 |
| crustacean_shellfish | 2 | 2 | 0 | 0 | 1.00 | 0.00 | 0.00 | 1 | 0 |
| tree_nuts | 32 | 31 | 1 | 0 | 0.97 | 0.03 | 0.00 | 2 | 0 |
| peanuts | 17 | 17 | 0 | 0 | 1.00 | 0.00 | 0.00 | 0 | 0 |
| wheat | 83 | 83 | 0 | 0 | 1.00 | 0.00 | 0.00 | 1 | 1 |
| soy | 92 | 92 | 0 | 0 | 1.00 | 0.00 | 0.00 | 8 | 1 |
| sesame | 23 | 22 | 1 | 0 | 0.96 | 0.04 | 0.00 | 1 | 0 |
| **all** | 379 | 375 | **4** | 0 | 0.99 | 0.01 | 0.00 | 26 | 3 |

macro-F1 (unsure=not flagged) 0.908 · exact-match 0.900 · unsure-cell rate 0.0012

**REFERENCE Qwen3.5-4B zero-shot, DEV subset-300 UNDECLARED** (n=268)

| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |
|---|---|---|---|---|---|---|---|---|---|
| milk | 103 | 101 | 2 | 0 | 0.98 | 0.02 | 0.00 | 7 | 0 |
| egg | 22 | 20 | 2 | 0 | 0.91 | 0.09 | 0.00 | 4 | 0 |
| fish | 3 | 3 | 0 | 0 | 1.00 | 0.00 | 0.00 | 2 | 0 |
| crustacean_shellfish | 3 | 3 | 0 | 0 | 1.00 | 0.00 | 0.00 | 0 | 0 |
| tree_nuts | 32 | 30 | 2 | 0 | 0.94 | 0.06 | 0.00 | 1 | 0 |
| peanuts | 17 | 17 | 0 | 0 | 1.00 | 0.00 | 0.00 | 0 | 0 |
| wheat | 81 | 81 | 0 | 0 | 1.00 | 0.00 | 0.00 | 1 | 0 |
| soy | 90 | 90 | 0 | 0 | 1.00 | 0.00 | 0.00 | 7 | 0 |
| sesame | 23 | 22 | 1 | 0 | 0.96 | 0.04 | 0.00 | 2 | 0 |
| **all** | 374 | 367 | **7** | 0 | 0.98 | 0.02 | 0.00 | 24 | 0 |

macro-F1 (unsure=not flagged) 0.936 · exact-match 0.896 · unsure-cell rate 0.0000

**traps_dev (unparseable 0)** (n=40)

| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |
|---|---|---|---|---|---|---|---|---|---|
| milk | 11 | 8 | 0 | 3 | 0.73 | 0.00 | 0.27 | 0 | 3 |
| egg | 8 | 4 | 3 | 1 | 0.50 | 0.38 | 0.12 | 0 | 0 |
| fish | 3 | 2 | 0 | 1 | 0.67 | 0.00 | 0.33 | 0 | 1 |
| crustacean_shellfish | 3 | 2 | 0 | 1 | 0.67 | 0.00 | 0.33 | 0 | 0 |
| tree_nuts | 8 | 4 | 4 | 0 | 0.50 | 0.50 | 0.00 | 0 | 0 |
| peanuts | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 | 0 |
| wheat | 11 | 2 | 7 | 2 | 0.18 | 0.64 | 0.18 | 0 | 1 |
| soy | 3 | 1 | 2 | 0 | 0.33 | 0.67 | 0.00 | 0 | 8 |
| sesame | 5 | 2 | 1 | 2 | 0.40 | 0.20 | 0.40 | 0 | 1 |
| **all** | 52 | 25 | **17** | 10 | 0.48 | 0.33 | 0.19 | 0 | 14 |

macro-F1 (unsure=not flagged) 0.573 · exact-match 0.425 · unsure-cell rate 0.0667

traps_dev misses (id, allergen): [('tdev-002', 'tree_nuts'), ('tdev-005', 'sesame'), ('tdev-010', 'egg'), ('tdev-010', 'wheat'), ('tdev-011', 'wheat'), ('tdev-012', 'wheat'), ('tdev-013', 'wheat'), ('tdev-014', 'wheat'), ('tdev-020', 'tree_nuts'), ('tdev-021', 'tree_nuts'), ('tdev-022', 'tree_nuts'), ('tdev-024', 'wheat'), ('tdev-025', 'egg'), ('tdev-027', 'soy'), ('tdev-028', 'egg'), ('tdev-029', 'wheat'), ('tdev-030', 'soy')]
abstained: [('tdev-001', 'crustacean_shellfish'), ('tdev-002', 'sesame'), ('tdev-003', 'wheat'), ('tdev-006', 'milk'), ('tdev-007', 'milk'), ('tdev-008', 'milk'), ('tdev-013', 'fish'), ('tdev-019', 'wheat'), ('tdev-019', 'sesame'), ('tdev-026', 'egg')]
false alarms: []

scoring tokens in 996,401 out 170,880 (~$0.501 at base-Qwen sampling prices)
### M3 SUMMARY — NEGATIVE RESULT: LoRA v1 did not close the compound gap
All numbers from `data/eval/m3_dev_report.md`, produced by `uv run python -I scripts/score_m3.py` (preds cached in `data/eval/lora_m3_*.jsonl`). Training: `uv run python -I scripts/train_m3.py train --lr 3e-5 --epochs 1` (5,755 rows, 359 steps, rank 16, ~$1.46 est). LR probe: `scripts/train_m3.py probe` / `probe --low`. Qwen zero-shot traps_dev figures: `uv run python -I scripts/score_traps_dev.py` (M2 section above).

| metric | Qwen3.5-4B zero-shot | LoRA v1 | source / command |
|---|---|---|---|
| dev undeclared recall (full, n=4,455 gold+) | ~0.98 (subset-300: 367/374) | 0.98 (4,366 caught, 86 missed, 3 abstained) | score_m3.py |
| dev subset-300 undeclared, missed | 7 | 4 | score_m3.py |
| traps_dev recall (52 gold+) | 0.71 | **0.48 (25/52)** | score_m3.py / score_traps_dev.py |
| traps_dev confident misses (said no) | 15 | **17** | same |
| traps_dev abstentions on gold positives | 0 | 10 | same |
| traps_dev confident false alarms | 5 | 0 | same |
| abstention over-firing | none (never abstains) | 3 gold-milk positives on traps_dev; 8 clean-row soy cells on traps_dev; 19 dev gold negatives (undeclared view; 48 in all-rows view) | score_m3.py |

Reading: LoRA v1 holds dev (0.98), introduces abstention the base model never produces, and has 0 confident false alarms on traps_dev, but it catches fewer traps, has more confident misses, and its abstention over-fires. Worst on traps_dev: wheat 2/11 caught (7 missed), soy 1/3, tree_nuts 4/8. No training row taught named-compound -> allergen; LR 3e-5 may also be low (probe was 20 steps). Not a win.
**Decision (user, 2026-10-18): stop training. Shipped configuration = LoRA + rules_product** (rules supply the compound layer). Test and frozen traps: NOT RUN for any system. LoRA + rules_product on traps_dev: NOT RUN.
Spend: probe ~$0.53 + train ~$1.46 + scoring ~$0.50 (est. at base sampling prices); ~$2.31 remaining, reserved for frozen-trap scoring + demo/first-user usage.

### NOTE 2026-10-18 — rules_product was edited after seeing first-user recipes, BEFORE frozen-trap scoring
- Sequence: 7881bab (RESULTS.md tracked; M3 negative result; frozen traps NOT RUN) -> 7616720 (M4). In M4 the user's cake recipes (`data/user_session/cake_inputs.txt`) were run through the pipeline; neither layer found wheat in "all-purpose flour" / "roasted kunafa", so `rules_product` was edited in 7616720: (a) named wheat flours/doughs added: all-purpose, all purpose, plain, bread, cake, pastry, self-raising, self-rising, strong flour; maida, atta, kunafa, kunafeh, knafeh, kadayif, kataifi, phyllo, filo; (b) simple plural matching (-s/-es) for ALL product terms. The frozen traps (sha e62378cc…) were not scored before or after; scored: NOT RUN.
- Overlap with the 62 frozen trap rows (free check, no API calls; regex over data/verified/traps.jsonl): the added wheat terms appear in **0** rows. Plural matching creates a new surface match in **5** rows (trap-006 "almonds", trap-016 "almonds", trap-017 "hazelnuts", trap-021 "pine nuts", trap-033 "fish balls"); it changes a product verdict in **1** (trap-021, tree_nuts, a gold positive: a miss becomes a hit, flattering rules_product on that row).
- `rules_naive` (the baseline) was NOT touched: the diff 7616720 vs e2e5809 changes no `_NAIVE_TERMS`/`_NAIVE` line, `_compile` defaults to plural=False, and re-running it reproduces all 300 cached M2 predictions (0 of 300 differ).
- Caveat: rules_product shares authorship with the trap set and is only ever reported as "LoRA + rules_product", never as a competitor.


## FROZEN TRAPS — single scored run, 2026-10-18 (first and only time scored)
- Commands: `uv run python -I scripts/score_frozen_traps.py` (sampled + saved predictions to `data/eval/frozen_traps_*.jsonl`; crashed on a console-encoding `print` AFTER sampling and saving), then `uv run python -I scripts/report_frozen_traps.py` (re-reads the saved predictions, makes NO API calls; produced every table below). No system was re-sampled.
- Freeze: traps sha256 e62378cc (verified immediately before scoring); `rules_product` frozen at HEAD 189392c (`src/contains/rules.py` sha256 7b966af7daf1) — no rules edits after this point, for any reason; later misses are findings, not patches.
- Systems: (1) SHIPPED = LoRA + rules_product through `contains.pipeline.analyze` (merge rule + unsure downgrade; 0 of 62 rows fell back to rules-only); (2) `rules_naive` alone; (3) Qwen3.5-4B zero-shot (same prompt/parser as the M2 baselines). Verdict mapping: contains->yes, absent->no, unsure->unsure. Abstentions are NEVER counted as recall; "CONFIDENT MISS" = said no.
- Rule-of-thumb read: recall = caught / gold+, confident-miss rate = said-no / gold+.

### Headline (85 gold positives over 62 rows; 49 hidden + 13 clean negatives)
| system | caught | CONFIDENT MISS | abstained | false alarms |
|---|---|---|---|---|
| 1. SHIPPED LoRA + rules_product | 73 (0.86) | 11 (0.13) | 1 | 0 |
| 1b. SHIPPED without trap-021 (83 gold+) | 71 (0.86) | 11 (0.13) | 1 | 0 |
| 2. rules_naive (baseline) | 48 (0.56) | 37 (0.44) | 0 | 4 |
| 3. Qwen3.5-4B zero-shot | 66 (0.78) | 19 (0.22) | 0 | 4 |
(source: tables below; abstentions on gold NEGATIVES: shipped 5 [soy x4, milk x1], others 0.)

**Footnote — trap-021** (`tomato, mozzarella, basil, balsamic glaze, pine nuts`): the shipped tree_nuts verdict is `yes`, and `rules_product` only matches "pine nuts" because of the plural fix made after seeing first-user recipes (commit 7616720; see the 2026-10-18 note above). Shipped number with the row: 73/85 caught, 11 confident misses; without it: 71/83, 11 confident misses (same ratios, tree_nuts 8/8 vs 7/7). The LoRA's own verdict on that row was not recorded separately, so whether the model alone would have caught it is NOT KNOWN; worst case without the fix is one more caught->missed (72/85).

### Read (no patches made)
- Shipped config beats both alternatives on recall and confident misses and has 0 false alarms, but it still confidently says "no" on 11 of 85 gold positives (13%) — those are the dangerous errors. Abstention caught only 1 trap (trap-003 milk, couverture chocolate): abstention is NOT what carries the safety here; rules do.
- Shipped confident misses: wheat x6 (bare "flour" in trap-026/034/045 — product rules drop bare "flour" by design and the LoRA does not say wheat; "imitation crab" trap-027; béchamel trap-025; caesar/croutons trap-007), milk x2 (pesto trap-002/018), egg x2 (tempura batter trap-013; frangipane trap-026), fish x1 (caesar dressing trap-007). Possible vocabulary gap: "béchamel" with an accent is not matched by the "bechamel" rule. All recorded as findings only.
- rules_naive false alarms are all wheat (4), Qwen false alarms are sesame x3 + egg x1.
- The earlier traps_dev result (LoRA alone 0.48 recall) is a different, dev-only set and system; the shipped LoRA+rules number here is the headline for the product.
- Credit: this run = 62 single sampler calls (shipped) + 62 Qwen zero-shot calls, ~124 short calls, est. ~$0.02 (token counts from the crashed print were not saved; Tinker returns no $ field).

traps sha256 e62378cc - n=62 - rules.py sha256 7b966af7daf1

**1. SHIPPED: LoRA + rules_product** (n=62 rows, 85 gold positives)

| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |
|---|---|---|---|---|---|---|
| milk | 19 | 16 | 2 | 1 | 0 | 1 |
| egg | 9 | 7 | 2 | 0 | 0 | 0 |
| fish | 8 | 7 | 1 | 0 | 0 | 0 |
| crustacean_shellfish | 4 | 4 | 0 | 0 | 0 | 0 |
| tree_nuts | 8 | 8 | 0 | 0 | 0 | 0 |
| peanuts | 1 | 1 | 0 | 0 | 0 | 0 |
| wheat | 21 | 15 | 6 | 0 | 0 | 0 |
| soy | 8 | 8 | 0 | 0 | 0 | 4 |
| sesame | 7 | 7 | 0 | 0 | 0 | 0 |
| **all** | 85 | 73 (0.86) | **11** (0.13) | 1 (0.01) | 0 | 5 |

confident misses (id, allergen): [('trap-002', 'milk'), ('trap-007', 'fish'), ('trap-007', 'wheat'), ('trap-013', 'egg'), ('trap-018', 'milk'), ('trap-025', 'wheat'), ('trap-026', 'egg'), ('trap-026', 'wheat'), ('trap-027', 'wheat'), ('trap-034', 'wheat'), ('trap-045', 'wheat')]
abstained on gold positives: [('trap-003', 'milk')]
false alarms: []

**1b. SHIPPED without trap-021** (n=61 rows, 83 gold positives)

| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |
|---|---|---|---|---|---|---|
| milk | 18 | 15 | 2 | 1 | 0 | 1 |
| egg | 9 | 7 | 2 | 0 | 0 | 0 |
| fish | 8 | 7 | 1 | 0 | 0 | 0 |
| crustacean_shellfish | 4 | 4 | 0 | 0 | 0 | 0 |
| tree_nuts | 7 | 7 | 0 | 0 | 0 | 0 |
| peanuts | 1 | 1 | 0 | 0 | 0 | 0 |
| wheat | 21 | 15 | 6 | 0 | 0 | 0 |
| soy | 8 | 8 | 0 | 0 | 0 | 4 |
| sesame | 7 | 7 | 0 | 0 | 0 | 0 |
| **all** | 83 | 71 (0.86) | **11** (0.13) | 1 (0.01) | 0 | 5 |

confident misses (id, allergen): [('trap-002', 'milk'), ('trap-007', 'fish'), ('trap-007', 'wheat'), ('trap-013', 'egg'), ('trap-018', 'milk'), ('trap-025', 'wheat'), ('trap-026', 'egg'), ('trap-026', 'wheat'), ('trap-027', 'wheat'), ('trap-034', 'wheat'), ('trap-045', 'wheat')]
abstained on gold positives: [('trap-003', 'milk')]
false alarms: []

trap-021 ('tomato, mozzarella, basil, balsamic glaze, pine nuts') shipped: tree_nuts=yes milk=yes

**2. rules_naive alone (BASELINE)** (n=62 rows, 85 gold positives)

| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |
|---|---|---|---|---|---|---|
| milk | 19 | 12 | 7 | 0 | 0 | 0 |
| egg | 9 | 4 | 5 | 0 | 0 | 0 |
| fish | 8 | 3 | 5 | 0 | 0 | 0 |
| crustacean_shellfish | 4 | 4 | 0 | 0 | 0 | 0 |
| tree_nuts | 8 | 1 | 7 | 0 | 0 | 0 |
| peanuts | 1 | 1 | 0 | 0 | 0 | 0 |
| wheat | 21 | 11 | 10 | 0 | 4 | 0 |
| soy | 8 | 5 | 3 | 0 | 0 | 0 |
| sesame | 7 | 7 | 0 | 0 | 0 | 0 |
| **all** | 85 | 48 (0.56) | **37** (0.44) | 0 (0.00) | 4 | 0 |

confident misses (id, allergen): [('trap-001', 'fish'), ('trap-002', 'milk'), ('trap-002', 'tree_nuts'), ('trap-003', 'milk'), ('trap-004', 'fish'), ('trap-006', 'tree_nuts'), ('trap-007', 'milk'), ('trap-007', 'egg'), ('trap-007', 'fish'), ('trap-007', 'wheat'), ('trap-013', 'egg'), ('trap-013', 'wheat'), ('trap-016', 'tree_nuts'), ('trap-017', 'tree_nuts'), ('trap-018', 'milk'), ('trap-018', 'tree_nuts'), ('trap-019', 'egg'), ('trap-021', 'milk'), ('trap-021', 'tree_nuts'), ('trap-022', 'wheat'), ('trap-023', 'wheat'), ('trap-023', 'soy'), ('trap-024', 'wheat'), ('trap-024', 'soy'), ('trap-025', 'wheat'), ('trap-026', 'egg'), ('trap-026', 'tree_nuts'), ('trap-027', 'fish'), ('trap-027', 'wheat'), ('trap-030', 'milk'), ('trap-030', 'wheat'), ('trap-038', 'wheat'), ('trap-044', 'fish'), ('trap-046', 'milk'), ('trap-046', 'egg'), ('trap-046', 'wheat'), ('trap-048', 'soy')]
abstained on gold positives: []
false alarms: [('trap-040', 'wheat'), ('trap-052', 'wheat'), ('trap-056', 'wheat'), ('trap-061', 'wheat')]

**3. Qwen3.5-4B zero-shot alone** (n=62 rows, 85 gold positives)

| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |
|---|---|---|---|---|---|---|
| milk | 19 | 15 | 4 | 0 | 0 | 0 |
| egg | 9 | 9 | 0 | 0 | 1 | 0 |
| fish | 8 | 6 | 2 | 0 | 0 | 0 |
| crustacean_shellfish | 4 | 3 | 1 | 0 | 0 | 0 |
| tree_nuts | 8 | 5 | 3 | 0 | 0 | 0 |
| peanuts | 1 | 1 | 0 | 0 | 0 | 0 |
| wheat | 21 | 14 | 7 | 0 | 0 | 0 |
| soy | 8 | 6 | 2 | 0 | 0 | 0 |
| sesame | 7 | 7 | 0 | 0 | 3 | 0 |
| **all** | 85 | 66 (0.78) | **19** (0.22) | 0 (0.00) | 4 | 0 |

confident misses (id, allergen): [('trap-002', 'milk'), ('trap-002', 'tree_nuts'), ('trap-004', 'fish'), ('trap-007', 'fish'), ('trap-008', 'milk'), ('trap-011', 'wheat'), ('trap-018', 'milk'), ('trap-018', 'tree_nuts'), ('trap-023', 'wheat'), ('trap-023', 'soy'), ('trap-024', 'wheat'), ('trap-026', 'tree_nuts'), ('trap-027', 'crustacean_shellfish'), ('trap-027', 'wheat'), ('trap-030', 'milk'), ('trap-034', 'wheat'), ('trap-038', 'wheat'), ('trap-045', 'wheat'), ('trap-048', 'soy')]
abstained on gold positives: []
false alarms: [('trap-007', 'sesame'), ('trap-018', 'sesame'), ('trap-019', 'sesame'), ('trap-025', 'egg')]


### FROZEN TRAPS — ABLATION: LoRA only (2026-10-18; ran AFTER the main scoring; frozen traps now CLOSED)
- Command: `uv run python -I scripts/score_frozen_lora_only.py` (62 single sampler calls; same prompt/parser as `score_m3.py`; raw model verdicts: no rules_product, no unsure-downgrade). Ran after the main scoring with **no intervening changes**: HEAD 4df9f8b, traps sha256 e62378cc, `rules.py` sha256 7b966af7daf1 (unchanged), working tree clean apart from the new script. Nothing tuned. The traps are not scored again by any system.
- Per-layer contribution on the 62 frozen rows (85 gold positives; caught / CONFIDENT MISS / abstained / false alarms):

| system | caught | CONFIDENT MISS | abstained | false alarms | abstained on gold negatives |
|---|---|---|---|---|---|
| rules_naive (baseline) | 48 | 37 | 0 | 4 | 0 |
| Qwen zero-shot | 66 | 19 | 0 | 4 | 0 |
| **LoRA only** | 56 (0.66) | 16 (0.19) | 13 (0.15) | 0 | 16 |
| **Shipped LoRA + rules_product** | 73 (0.86) | 11 (0.13) | 1 | 0 | 5 |

- Reading: adding rules_product to the LoRA turns 17 gold positives the LoRA did not catch into catches (56 -> 73). The LoRA alone is NOT better than Qwen zero-shot on recall (56 vs 66) here; its distinguishing behaviour is 0 false alarms plus abstention (13 on gold positives, 16 on gold negatives).
- **Cost of the no-trigger downgrade:** of the shipped config's 11 confident misses, 5 were LoRA-only ABSTENTIONS that the pipeline's "unsure needs an unnamed-source term" rule downgraded to absent: trap-002 milk and trap-018 milk (pesto), trap-007 wheat (caesar/croutons), trap-013 egg (tempura batter), trap-026 egg (frangipane). The downgrade rule reduced noisy unsure verdicts but converted those abstentions into confident "no" on real allergens. Recorded as a finding; not patched.
- LoRA-only tables (abstentions kept separate from misses):

**4. LoRA ONLY (no rules; unparseable 0)** (n=62 rows, 85 gold positives)

| allergen | gold+ | caught | CONFIDENT MISS (said no) | abstained | false alarms | abstained on neg |
|---|---|---|---|---|---|---|
| milk | 19 | 16 | 0 | 3 | 0 | 3 |
| egg | 9 | 7 | 0 | 2 | 0 | 3 |
| fish | 8 | 4 | 3 | 1 | 0 | 0 |
| crustacean_shellfish | 4 | 3 | 1 | 0 | 0 | 0 |
| tree_nuts | 8 | 6 | 2 | 0 | 0 | 0 |
| peanuts | 1 | 1 | 0 | 0 | 0 | 0 |
| wheat | 21 | 8 | 10 | 3 | 0 | 1 |
| soy | 8 | 4 | 0 | 4 | 0 | 8 |
| sesame | 7 | 7 | 0 | 0 | 0 | 1 |
| **all** | 85 | 56 (0.66) | **16** (0.19) | 13 (0.15) | 0 | 16 |

confident misses (id, allergen): [('trap-001', 'fish'), ('trap-002', 'tree_nuts'), ('trap-004', 'fish'), ('trap-007', 'fish'), ('trap-018', 'tree_nuts'), ('trap-022', 'wheat'), ('trap-023', 'wheat'), ('trap-025', 'wheat'), ('trap-026', 'wheat'), ('trap-027', 'crustacean_shellfish'), ('trap-027', 'wheat'), ('trap-030', 'wheat'), ('trap-034', 'wheat'), ('trap-038', 'wheat'), ('trap-041', 'wheat'), ('trap-045', 'wheat')]
abstained on gold positives: [('trap-002', 'milk'), ('trap-003', 'milk'), ('trap-007', 'wheat'), ('trap-013', 'egg'), ('trap-013', 'wheat'), ('trap-018', 'milk'), ('trap-023', 'soy'), ('trap-024', 'wheat'), ('trap-024', 'soy'), ('trap-026', 'egg'), ('trap-044', 'fish'), ('trap-044', 'soy'), ('trap-048', 'soy')]
false alarms: []

tokens in 13,355 out 2,976
