# ============================================================
# AI Interview Simulation System — one-click launcher
# Save this file in the same folder as app.py, then just double-click
# it (or right-click -> Run with PowerShell) instead of typing the
# activation + run commands by hand each time.
# ============================================================

# Move to the folder this script lives in, so it works regardless of
# where you double-click it from.
Set-Location -Path $PSScriptRoot

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host "Checking Ollama models..." -ForegroundColor Cyan
$models = & ollama list 2>$null
if ($models -notmatch "phi3") {
    Write-Host "WARNING: 'phi3' model not found in Ollama. Run: ollama pull phi3" -ForegroundColor Yellow
}
if ($models -notmatch "moondream") {
    Write-Host "WARNING: 'moondream' model not found in Ollama. Run: ollama pull moondream" -ForegroundColor Yellow
}

Write-Host "Starting AI Interview Simulation System..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" -m streamlit run app.py