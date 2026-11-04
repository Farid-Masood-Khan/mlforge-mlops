# MLForge PowerShell Automation Script
param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$VenvPython = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }

switch ($Command.ToLower()) {
    "setup" {
        Write-Host "Setting up virtual environment..." -ForegroundColor Cyan
        if (-not (Test-Path ".venv")) {
            python -m venv .venv
        }
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt
    }
    "lint" {
        Write-Host "Running ruff linter..." -ForegroundColor Cyan
        & $VenvPython -m ruff check src tests
    }
    "format" {
        Write-Host "Checking formatting with ruff..." -ForegroundColor Cyan
        & $VenvPython -m ruff format --check src tests
    }
    "typecheck" {
        Write-Host "Running mypy type checks..." -ForegroundColor Cyan
        & $VenvPython -m mypy src
    }
    "test" {
        Write-Host "Running pytest test suite..." -ForegroundColor Cyan
        & $VenvPython -m pytest tests
    }
    "data" {
        Write-Host "Downloading and validating data..." -ForegroundColor Cyan
        & $VenvPython -m src.data.download
        & $VenvPython -m src.data.validate
    }
    "train" {
        Write-Host "Executing training pipeline..." -ForegroundColor Cyan
        & $VenvPython -m src.models.train
    }
    "serve" {
        Write-Host "Starting FastAPI serving service on port 8000..." -ForegroundColor Cyan
        & $VenvPython -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
    }
    "drift" {
        Write-Host "Running traffic simulation and generating drift report..." -ForegroundColor Cyan
        & $VenvPython -m src.monitoring.simulator --count 500 --perturb monthly_charges
        & $VenvPython -m src.monitoring.report
    }
    "demo" {
        Write-Host "Starting Streamlit UI demo..." -ForegroundColor Cyan
        & $VenvPython -m streamlit run demo/app.py
    }
    "reproduce" {
        Write-Host "Verifying pipeline end-to-end reproducibility..." -ForegroundColor Cyan
        & $VenvPython -m src.reproduce
    }
    "ci" {
        Write-Host "Running full local CI verification..." -ForegroundColor Green
        & $VenvPython -m ruff check src tests demo
        & $VenvPython -m ruff format --check src tests demo
        & $VenvPython -m mypy src
        & $VenvPython -m pytest tests
    }
    default {
        Write-Host "MLForge Command Runner" -ForegroundColor Yellow
        Write-Host "Usage: .\run.ps1 [setup|lint|format|typecheck|test|data|train|serve|drift|demo|reproduce|ci]"
    }
}

