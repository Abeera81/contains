#!/usr/bin/env pwsh
# Windows convenience shim mirroring the canonical Makefile targets.
# The Makefile remains canonical (Render builds on Linux); this only saves
# installing GNU make locally. Usage: ./make.ps1 <target>
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

function Invoke-Step($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "setup"     { Invoke-Step "uv sync --extra dev" }
    "test"      { Invoke-Step "uv run pytest" }
    "smoke" {
        Invoke-Step "uv run python scripts/smoke_tinker_train.py"
        Invoke-Step "uv run python scripts/smoke_tinker_sample.py"
        Invoke-Step "uv run python scripts/smoke_ollama_gemma.py"
        Invoke-Step "uv run python scripts/smoke_off_parquet.py"
    }
    "data"      { Write-Host "data: NOT IMPLEMENTED (milestone M1)"; exit 1 }
    "baselines" { Write-Host "baselines: NOT IMPLEMENTED (milestone M2)"; exit 1 }
    "train"     { Write-Host "train: NOT IMPLEMENTED (milestone M3)"; exit 1 }
    "eval"      { Write-Host "eval: NOT IMPLEMENTED (milestone M3)"; exit 1 }
    "app"       { Write-Host "app: NOT IMPLEMENTED (milestone M4)"; exit 1 }
    default     { Write-Host "Targets: setup, test, smoke, data, baselines, train, eval, app" }
}
