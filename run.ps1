# Dr Halim - Quick Launch (venv must exist, run setup_and_run.ps1 first)

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BackendDir = Join-Path $ScriptDir "backend"
$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found." -ForegroundColor Red
    Write-Host "Run setup_and_run.ps1 first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "  Dr Halim starting on http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Open frontend\index.html in your browser"   -ForegroundColor Green
Write-Host "  Ctrl+C to stop"                             -ForegroundColor DarkGray
Write-Host ""

Set-Location $BackendDir
& $VenvPython -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
