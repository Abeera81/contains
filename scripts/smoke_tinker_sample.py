"""M0 smoke test — TINKER SAMPLE (budget check).

Samples one short completion from base Qwen3.5-4B and one from gpt-oss-120b via
the Tinker sampling API (same TINKER_API_KEY). Tokens kept tiny on purpose.
Prints outputs, latency and token counts, then one PASS/FAIL line.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import find_model, list_supported_models, load_env, service_client

MAX_TOKENS = 32
PROMPT = (
    "INGREDIENTS: wheat flour, whey powder, soy lecithin\n"
    "Name the US major allergens present, comma-separated:"
)


def sample_one(sc, model_id, tokenizer_prompt_fn):
    import tinker

    sampling = sc.create_sampling_client(base_model=model_id)
    tok = sampling.get_tokenizer()
    prompt_ids = tokenizer_prompt_fn(tok)
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    t0 = time.time()
    fut = sampling.sample(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=params,
    )
    resp = fut.result() if hasattr(fut, "result") else fut
    dt = time.time() - t0
    seq = resp.sequences[0]
    out_tokens = list(seq.tokens)
    text = tok.decode(out_tokens)
    return text.strip(), len(prompt_ids), len(out_tokens), dt


def prompt_ids_for(tok):
    # Plain chat prompt; both model families accept apply_chat_template.
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tok.encode(text, add_special_tokens=False)
    except Exception:
        return tok.encode(PROMPT)


def main() -> int:
    load_env()
    sc = service_client()
    names, _ = list_supported_models(sc)
    qwen = find_model(names, "qwen3.5", "4b")
    gptoss = find_model(names, "gpt-oss-120b")
    # avoid the :peft: variants for a plain base sample
    gptoss = find_model([n for n in names if ":" not in n], "gpt-oss-120b") or gptoss
    qwen = find_model([n for n in names if ":" not in n], "qwen3.5", "4b") or qwen

    if not qwen or not gptoss:
        print(f"FAIL: TINKER SAMPLE — missing model id (qwen={qwen!r}, gpt-oss={gptoss!r})")
        return 1

    ok = True
    for label, mid in (("Qwen3.5-4B base", qwen), ("gpt-oss-120b", gptoss)):
        try:
            text, n_in, n_out, dt = sample_one(sc, mid, prompt_ids_for)
            print(f"[{label}] id={mid!r}")
            print(f"  output: {text!r}")
            print(f"  prompt_tokens={n_in} completion_tokens={n_out} latency={dt:.2f}s")
        except Exception as e:
            ok = False
            print(f"[{label}] id={mid!r} ERROR {type(e).__name__}: {e}")

    if ok:
        print(f"PASS: TINKER SAMPLE — sampled {MAX_TOKENS}-token max from Qwen3.5-4B and gpt-oss-120b")
        return 0
    print("FAIL: TINKER SAMPLE — at least one model failed to sample")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:])
        print(f"FAIL: TINKER SAMPLE — {type(e).__name__}: {e}")
        sys.exit(1)
