"""Bake the base-model tokenizer into HF_HOME at BUILD time so the wake path never calls the Hugging Face hub.

    HF_HOME=<dir> uv run python -I scripts/prefetch_tokenizer.py

Tinker's SamplingClient.get_tokenizer() does AutoTokenizer.from_pretrained(<base model id>, fast=True), so we do
exactly that here, then prove the cache is complete by loading it again with local_files_only=True. Exits non-zero
(failing the build) if either step fails. Tokenizer files only; no model weights are downloaded.
"""
import os
import sys
import time

from transformers.models.auto.tokenization_auto import AutoTokenizer

model = (os.environ.get("TINKER_BASE_MODEL") or "Qwen/Qwen3.5-4B").split(":")[0]
t0 = time.time()
AutoTokenizer.from_pretrained(model, fast=True)
print(f"downloaded tokenizer for {model} in {time.time() - t0:.1f}s into HF_HOME={os.environ.get('HF_HOME', '(default)')}")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model, fast=True, local_files_only=True)
ids = tok.encode("wheat flour, butter", add_special_tokens=False)
print(f"offline reload OK in {time.time() - t0:.1f}s ({len(ids)} tokens) - cache is complete")
sys.exit(0)
