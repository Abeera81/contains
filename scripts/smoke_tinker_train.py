"""M0 smoke test — TINKER TRAIN.

Creates a LoRA training client for Qwen3.5-4B, builds a 2-example toy batch with
the tinker-cookbook renderer (falling back to the tokenizer chat template if the
cookbook is not installed on this platform), and runs ONE forward_backward +
optim_step. Prints the exact Tinker model ID, loss, token count and cost info.

Prints exactly one PASS/FAIL line at the end. Never prints secrets.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import find_model, list_supported_models, load_env, service_client

BASE_MODEL_WANT = ("qwen3.5", "4b")
RANK = 16
LR = 1e-4
PRICE_PER_1M_TRAIN_TOK = 0.737  # Tinker pricing page, 2026-10-17

SYSTEM = "You label US major food allergens. Output exactly one line."
TOY = [
    (
        "INGREDIENTS: wheat flour, whey powder, soy lecithin\n"
        "Which of [milk, egg, fish, crustacean_shellfish, tree_nuts, peanuts, wheat, soy, sesame] does this contain?",
        "CONTAINS: milk, wheat, soy",
    ),
    (
        "INGREDIENTS: water, sugar, anchovy extract, tamarind\n"
        "Which of [milk, egg, fish, crustacean_shellfish, tree_nuts, peanuts, wheat, soy, sesame] does this contain?",
        "CONTAINS: fish",
    ),
]


def build_datum_cookbook(renderer, user, assistant):
    """Build a Datum using a tinker-cookbook renderer. Returns (datum, n_full, n_completion)."""
    import tinker

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    # tinker-cookbook renderers expose a supervised-example builder; its exact
    # name has varied, so try the known ones.
    for meth in ("build_supervised_example", "build_supervised_datum", "render_supervised"):
        fn = getattr(renderer, meth, None)
        if fn is None:
            continue
        out = fn(msgs)
        if isinstance(out, tinker.Datum):
            return out, 0, 0  # counts unknown from an opaque Datum
        tokens, weights = out  # (tokens, weights)
        return _datum_from_tokens(tokens, weights)
    raise RuntimeError("cookbook renderer has no supervised-example builder")


def _datum_from_tokens(tokens, weights):
    import tinker

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )
    n_completion = int(sum(1 for w in weights[1:] if w > 0))
    return datum, len(tokens), n_completion


def build_datum_fallback(tokenizer, user, assistant):
    """Build a Datum via the tokenizer chat template (what the renderer wraps)."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    completion_tokens = full_tokens[len(prompt_tokens):]
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return _datum_from_tokens(tokens, weights)


def extract_loss(fb_out, n_completion_tokens: int) -> tuple[float | None, float | None]:
    """Return (loss_sum, mean_loss_per_completion_token) from a ForwardBackwardOutput."""
    import numpy as np

    loss_sum = None
    metrics = getattr(fb_out, "metrics", None)
    if isinstance(metrics, dict):
        for k in ("loss:sum", "loss", "train_loss"):
            if k in metrics:
                loss_sum = float(metrics[k])
                break

    # Fall back to summing per-token elementwise loss across datums.
    if loss_sum is None:
        lfo = getattr(fb_out, "loss_fn_outputs", None) or []
        total = 0.0
        found = False
        for d in lfo:
            ew = d.get("elementwise_loss") if isinstance(d, dict) else None
            if ew is not None:
                arr = np.asarray(getattr(ew, "data", ew), dtype=float)
                total += float(arr.sum())
                found = True
        loss_sum = total if found else None

    mean = (loss_sum / n_completion_tokens) if (loss_sum is not None and n_completion_tokens) else None
    return loss_sum, mean


def main() -> int:
    load_env()
    import tinker

    sc = service_client()
    names, _ = list_supported_models(sc)
    model_id = find_model(names, *BASE_MODEL_WANT)
    if model_id is None:
        print(f"FAIL: TINKER TRAIN — no Qwen3.5-4B model in server capabilities ({len(names)} models)")
        return 1
    print(f"Model discovery: get_server_capabilities() returned {len(names)} models.")
    print("  Display name: Qwen/Qwen3.5-4B")
    print(f"  Tinker ID passed to create_lora_training_client(base_model=...): {model_id!r}")

    tc = sc.create_lora_training_client(base_model=model_id, rank=RANK)
    tokenizer = tc.get_tokenizer()

    # Renderer: prefer tinker-cookbook; fall back to the tokenizer chat template.
    renderer = None
    renderer_path = "tokenizer chat-template fallback (tinker-cookbook not importable on this platform)"
    try:
        from tinker_cookbook import renderers

        reg = list(renderers.get_registered_renderer_names())
        rname = next((r for r in reg if "qwen3" in r.lower()), reg[0] if reg else None)
        if rname:
            renderer = renderers.get_renderer(rname, tokenizer)
            renderer_path = f"tinker-cookbook renderer {rname!r}"
    except Exception as e:
        renderer_path = f"tokenizer chat-template fallback (cookbook import failed: {type(e).__name__})"

    data = []
    total_tokens = 0
    total_completion = 0
    for user, assistant in TOY:
        if renderer is not None:
            try:
                d, n, nc = build_datum_cookbook(renderer, user, assistant)
            except Exception:
                d, n, nc = build_datum_fallback(tokenizer, user, assistant)
                renderer_path += " [+ fallback for datum build]"
        else:
            d, n, nc = build_datum_fallback(tokenizer, user, assistant)
        data.append(d)
        total_tokens += n
        total_completion += nc
    print(f"Renderer: {renderer_path}")
    print(f"Toy batch: {len(data)} examples, {total_tokens} tokens total ({total_completion} completion tokens).")

    t0 = time.time()
    fb_future = tc.forward_backward(data, "cross_entropy")
    fb_out = fb_future.result()
    opt_future = tc.optim_step(adam_params=tinker.AdamParams(learning_rate=LR))
    opt_future.result()
    dt = time.time() - t0

    loss_sum, mean_loss = extract_loss(fb_out, total_completion)
    sum_str = f"{loss_sum:.4f}" if loss_sum is not None else "n/a"
    mean_str = f"{mean_loss:.4f}" if mean_loss is not None else "n/a"
    est_cost = total_tokens / 1_000_000 * PRICE_PER_1M_TRAIN_TOK

    print(f"forward_backward + optim_step OK in {dt:.1f}s")
    print(f"Loss (metrics['loss:sum'] over completion tokens): {sum_str}")
    print(f"Mean loss per completion token: {mean_str}  (= loss_sum / {total_completion})")
    print(
        f"Reported cost: Tinker returns no per-call $ field; "
        f"estimate from tokens = {total_tokens} tok x ${PRICE_PER_1M_TRAIN_TOK}/1M = ${est_cost:.6f}"
    )
    print(f"PASS: TINKER TRAIN — 1 step on {model_id}, mean_loss={mean_str}, ~${est_cost:.6f} est spend")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:])
        print(f"FAIL: TINKER TRAIN — {type(e).__name__}: {e}")
        sys.exit(1)
