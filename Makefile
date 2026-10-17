# Contains — canonical task runner (Linux/Render). On Windows use ./make.ps1 <target>.
# Keep target names in sync with AGENTS.md and make.ps1.

.PHONY: setup data baselines train eval app test smoke

PY := uv run

setup:  ## uv venv + install deps (incl. dev)
	uv sync --extra dev

data:  ## build OFF splits (train/dev/test by brand)
	@echo "data: NOT IMPLEMENTED (milestone M1)"; exit 1

baselines:  ## run baseline systems, write docs/RESULTS.md
	@echo "baselines: NOT IMPLEMENTED (milestone M2)"; exit 1

train:  ## Tinker LoRA training
	@echo "train: NOT IMPLEMENTED (milestone M3)"; exit 1

eval:  ## full evaluation -> docs/RESULTS.md + charts
	@echo "eval: NOT IMPLEMENTED (milestone M3)"; exit 1

app:  ## run FastAPI app at http://127.0.0.1:8000
	@echo "app: NOT IMPLEMENTED (milestone M4)"; exit 1

test:  ## unit tests
	$(PY) pytest

smoke:  ## run all M0 go/no-go smoke tests
	$(PY) python scripts/smoke_tinker_train.py
	$(PY) python scripts/smoke_tinker_sample.py
	$(PY) python scripts/smoke_ollama_gemma.py
	$(PY) python scripts/smoke_off_parquet.py
