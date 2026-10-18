"""Pluggable model-layer samplers, chosen by config (no code change to swap).

    SAMPLER=tinker_lora   Qwen3.5-4B + our LoRA on Tinker   (needs LORA_SAMPLER_PATH)
    SAMPLER=tinker_base   Qwen3.5-4B zero-shot on Tinker    (TINKER_BASE_MODEL)
    SAMPLER=ollama        local gemma3:4b via Ollama        (OLLAMA_HOST, OLLAMA_TEXT_MODEL)
    SAMPLER=rules_only    no model layer, zero API calls

All use the SAME prompt and strict parser as the baselines. Every failure raises SamplerError;
the pipeline turns that into a rules-only result with a banner.
"""

from __future__ import annotations

import os

from contains.prompt import build_prompt, parse_response


class SamplerError(RuntimeError):
    pass


class RulesOnly:
    name = "rules_only"

    def verdicts(self, text: str):  # pragma: no cover - never called
        return None


class _Misconfigured:
    """Configured sampler cannot be built -> every call fails -> banner (never a crash)."""

    def __init__(self, name: str, why: str):
        self.name, self.why = name, why

    def verdicts(self, text: str):
        raise SamplerError(self.why)


class TinkerSampler:
    def __init__(self, name: str, model_path: str | None = None, base_model: str | None = None):
        import threading
        self.name, self._path, self._base = name, model_path, base_model
        self._client = self._tok = None
        self._lock = threading.Lock()  # the warm-up thread and the first request share ONE client build
        self.state = "cold"            # cold -> warming -> ready | failed

    @property
    def ready(self) -> bool:
        return self._client is not None and self._tok is not None

    def _ensure(self):
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            self.state = "warming"
            try:
                from dotenv import load_dotenv
                load_dotenv()
                import tinker
                sc = tinker.ServiceClient()
                client = sc.create_sampling_client(model_path=self._path, base_model=self._base)
                self._tok = client.get_tokenizer()
                self._client = client
                self.state = "ready"
            except Exception:
                self.state = "failed"
                raise

    def warm(self):
        """Create the sampling client now (no sample call, no token cost)."""
        self._ensure()

    def verdicts(self, text: str) -> dict[str, str]:
        import tinker
        self._ensure()
        p = self._tok.apply_chat_template([{"role": "user", "content": build_prompt(text)}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        ids = self._tok.encode(p, add_special_tokens=False)
        out = self._client.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                                  sampling_params=tinker.SamplingParams(max_tokens=64, temperature=0.0)).result()
        pred, ok = parse_response(self._tok.decode(list(out.sequences[0].tokens)))
        if not ok:
            raise SamplerError("unparseable model output")
        return pred


class OllamaSampler:
    name = "ollama"

    def verdicts(self, text: str) -> dict[str, str]:
        import httpx
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        r = httpx.post(f"{host}/api/generate", json={
            "model": os.environ.get("OLLAMA_TEXT_MODEL", "gemma3:4b"), "prompt": build_prompt(text),
            "stream": False, "options": {"num_predict": 64, "temperature": 0.0}}, timeout=120.0)
        r.raise_for_status()
        pred, ok = parse_response(r.json().get("response", ""))
        if not ok:
            raise SamplerError("unparseable model output")
        return pred


def get_sampler(env=None):
    env = os.environ if env is None else env
    kind = (env.get("SAMPLER") or "tinker_lora").strip().lower()
    if kind == "rules_only":
        return RulesOnly()
    if kind == "tinker_lora":
        path = (env.get("LORA_SAMPLER_PATH") or "").strip()
        if not path:
            return _Misconfigured("tinker_lora", "LORA_SAMPLER_PATH not set")
        return TinkerSampler("tinker_lora", model_path=path, base_model=env.get("TINKER_BASE_MODEL") or "Qwen/Qwen3.5-4B")
    if kind == "tinker_base":
        return TinkerSampler("tinker_base", base_model=env.get("TINKER_BASE_MODEL") or "Qwen/Qwen3.5-4B")
    if kind == "ollama":
        return OllamaSampler()
    return _Misconfigured(kind, f"unknown SAMPLER {kind!r}")
