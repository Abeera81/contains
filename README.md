# Contains

A ~$2 fine-tuned open model (Qwen3.5-4B + Tinker LoRA) plus a rules layer that finds the
allergens hidden inside compound ingredients and prints a US-style **"Contains:"** line for
the 9 US major allergens. Runs offline on the baker's laptop, so the recipe never leaves it.

Write-up: [Contains: I built an allergen checker for home bakers that says "I'm not sure" instead of guessing](https://dev.to/abeera_lodhi/contains-i-built-an-allergen-checker-for-home-bakers-that-says-im-not-sure-instead-of-guessing-29g8) (DEV).

> **Draft, not certification.** Output is a draft a human must verify with their suppliers.
> Scope is the 9 US major allergens (FALCPA + FASTER Act). Not EU rules, not cross-contact,
> not nutrition, not legal certification.

## Status

Milestone M0 (setup + go/no-go smoke tests). See `PLAN.md`.

## Quickstart

```bash
make setup      # uv venv + deps (Windows: ./make.ps1 setup)
make test       # unit tests
make smoke      # M0 go/no-go smoke tests (needs TINKER_API_KEY + Ollama)
```

Copy `.env.example` to `.env` and fill `TINKER_API_KEY`. The same key is used for LoRA
training, LoRA sampling, and the `gpt-oss-120b` big-model baseline — there is no second
inference account.

## Run the app

```bash
uv run uvicorn contains.app:app --host 127.0.0.1 --port 8000     # or: make app / ./make.ps1 app
```

Config via `.env` / environment: `SAMPLER=tinker_lora|tinker_base|ollama|rules_only` (swap the model
layer with no code change), `LORA_SAMPLER_PATH`, `TINKER_API_KEY`. If the model layer is unreachable
the app answers rules-only and shows a banner.

### Photo reading (local-only)

Reading a label photo runs **Gemma 3 4B in Ollama on your own machine**; the hosted demo has no
Ollama (and not the RAM for it), so there the page says photo reading is local-only and you paste
the text instead. To use photos: install [Ollama](https://ollama.com), `ollama pull gemma3:4b`, run
the app locally as above. Expect ~10-60 s per photo on CPU.

### Hosted demo: cold starts and latency

The hosted demo runs on Render's free tier. A free instance **spins down after about 15 minutes of
inactivity, and the first request after that can take 50 s or more** (Render's documented behaviour)
while the container restarts and the Tinker client warms up.

Measured on the live site (warm instance, 2026-10-18):

| request | time |
|---|---|
| `GET /healthz` | 0.5-1.2 s |
| first model sample after an idle stretch | ~8.4 s |
| later new inputs (model sample) | 2-3 s |
| repeat of an input already seen (in-memory cache) | 0.57 s |

**No cold-start figure is reported.** I tried to measure one after 17+ minutes idle, but the instance
had not been spun down (the in-memory counters were unchanged, so the process never restarted), and I
chose not to trigger a redeploy just to time it. Treat the 50 s+ figure as Render's, not ours.

## Architecture (short)

- **remote** backend: Tinker sampling, Qwen3.5-4B + our LoRA (hosted demo on Render free tier)
- **local** backend: llama.cpp, Qwen3.5-4B Q4 GGUF + LoRA (fully offline)
- **baselines**: keyword rules; Qwen3.5-4B zero-shot; Gemma 3 4B zero-shot (Ollama);
  gpt-oss-120b (Tinker sampling)
- **photo → text**: Gemma 3 4B vision via Ollama reads supplier label photos (local)

## Credits

- **[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)** — supervised
  training loop and renderers. Apache-2.0.
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — local GGUF inference and LoRA
  conversion. MIT.
- **[Open Food Facts](https://world.openfoodfacts.org/)** — training/eval data
  (`openfoodfacts/product-database`). Open Database License (ODbL).
- **Gemma** (Google) — label-photo transcription and a zero-shot baseline, via Ollama,
  under the Gemma Terms of Use.
- **Qwen3.5-4B** (Alibaba) and **gpt-oss-120b** (OpenAI) — base / baseline models, served
  through [Tinker](https://thinkingmachines.ai/).

Commits may continue after the hackathon deadline; see the DEV post for what was done by the
deadline.
