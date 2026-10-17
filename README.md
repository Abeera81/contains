# Contains

A ~$2 fine-tuned open model (Qwen3.5-4B + Tinker LoRA) plus a rules layer that finds the
allergens hidden inside compound ingredients and prints a US-style **"Contains:"** line for
the 9 US major allergens. Runs offline on the baker's laptop, so the recipe never leaves it.

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
