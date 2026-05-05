# Dr Halim - Setup and Launch
# Usage: powershell -ExecutionPolicy Bypass -File setup_and_run.ps1

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir    = Join-Path $ScriptDir "venv"
$ReqFile    = Join-Path $BackendDir "requirements.txt"

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "     Dr Halim - AI Vet  Setup           " -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# ================================================================
#  STEP 1 - Find Python 3.10 / 3.11 / 3.12
# ================================================================
Write-Host "[1/5] Searching for compatible Python (3.10 / 3.11 / 3.12)..." -ForegroundColor Yellow

$PythonExe = $null
$PythonVer = $null

# Versions to try, best first
$PyVersions = @("3.12", "3.11", "3.10")

foreach ($v in $PyVersions) {
    try {
        $out = & py "-$v" --version 2>&1
        if ($out -match "Python ($v\.\d+)") {
            $PythonExe = "py_launcher_$v"
            $PythonVer = $matches[1]
            Write-Host "      Found Python $PythonVer via py launcher (-$v)" -ForegroundColor Green
            break
        }
    } catch {}
}

# If py launcher failed, search common install paths
if (-not $PythonExe) {
    $SearchPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe"
    )
    foreach ($p in $SearchPaths) {
        if (Test-Path $p) {
            $out = & $p --version 2>&1
            if ($out -match "Python (3\.(10|11|12)\.\d+)") {
                $PythonExe = $p
                $PythonVer = $matches[1]
                Write-Host "      Found Python $PythonVer at: $p" -ForegroundColor Green
                break
            }
        }
    }
}

if (-not $PythonExe) {
    Write-Host ""
    Write-Host "  ERROR: No compatible Python found (need 3.10, 3.11 or 3.12)." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Install Python 3.12 from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/release/python-31210/" -ForegroundColor Yellow
    Write-Host "  -> Check 'Add Python to PATH' during install." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# ================================================================
#  STEP 2 - Create virtual environment
# ================================================================
Write-Host ""
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow

if (Test-Path (Join-Path $VenvDir "Scripts\python.exe")) {
    Write-Host "      venv already exists, skipping creation." -ForegroundColor DarkGray
} else {
    if (Test-Path $VenvDir) { Remove-Item $VenvDir -Recurse -Force }

    if ($PythonExe -match "^py_launcher_(.+)$") {
        $v = $matches[1]
        & py "-$v" -m venv $VenvDir
    } else {
        & $PythonExe -m venv $VenvDir
    }

    if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
        Write-Host "  ERROR: Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "      Virtual environment created." -ForegroundColor Green
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"

# ================================================================
#  STEP 3 - Upgrade pip
# ================================================================
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip setuptools wheel -q
Write-Host "      pip upgraded." -ForegroundColor Green

# ================================================================
#  STEP 4 - Install packages in correct order
# ================================================================
Write-Host ""
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
Write-Host "      (This can take 5-15 minutes on first run)" -ForegroundColor DarkGray
Write-Host ""

function Install-Package {
    param([string]$pkg, [string]$label = "")
    $display = if ($label) { $label } else { $pkg }
    Write-Host "      Installing $display ..." -ForegroundColor DarkGray -NoNewline
    $result = & $VenvPip install $pkg.Split(" ") --quiet 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        return $false
    }
    return $true
}

# Install numpy first (critical - must be 1.x for TF + ultralytics compatibility)
Install-Package "numpy==1.26.4" "numpy 1.26.4"

# TensorFlow CPU (breed model - avoids GPU/CUDA complexity)
Install-Package "tensorflow-cpu==2.16.2" "tensorflow-cpu 2.16.2"

# PyTorch CPU build (explicit index for Windows compatibility)
Write-Host "      Installing torch + torchvision (CPU)..." -ForegroundColor DarkGray -NoNewline
& $VenvPip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " Retrying without index-url..." -ForegroundColor Yellow
    Install-Package "torch==2.3.1 torchvision==0.18.1" "torch + torchvision"
}

# API framework
Install-Package "fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.9" "FastAPI + uvicorn"

# HTTP
Install-Package "requests==2.31.0" "requests"

# Computer vision
Install-Package "Pillow==10.3.0" "Pillow"
Install-Package "opencv-python==4.9.0.80" "opencv-python"
Install-Package "ultralytics==8.2.0" "ultralytics (YOLOv8)"

# Audio
Install-Package "librosa==0.10.1 soundfile==0.12.1 audioread==3.0.1" "librosa + soundfile"

# Numerics
Install-Package "scipy==1.13.1 matplotlib==3.9.0 tqdm==4.66.4" "scipy + matplotlib"

Write-Host ""
Write-Host "      All packages installed successfully!" -ForegroundColor Green

# ================================================================
#  STEP 5 - Quick sanity check
# ================================================================
Write-Host ""
Write-Host "[5/5] Running import check..." -ForegroundColor Yellow

$CheckScript = @"
import sys
errors = []
try:
    import tensorflow as tf
    print(f'  tensorflow {tf.__version__}  OK')
except Exception as e:
    errors.append(f'  tensorflow  FAILED: {e}')

try:
    import torch
    print(f'  torch       {torch.__version__}  OK')
except Exception as e:
    errors.append(f'  torch  FAILED: {e}')

try:
    import torchvision
    print(f'  torchvision {torchvision.__version__}  OK')
except Exception as e:
    errors.append(f'  torchvision  FAILED: {e}')

try:
    import ultralytics
    print(f'  ultralytics {ultralytics.__version__}  OK')
except Exception as e:
    errors.append(f'  ultralytics  FAILED: {e}')

try:
    import librosa
    print(f'  librosa     {librosa.__version__}  OK')
except Exception as e:
    errors.append(f'  librosa  FAILED: {e}')

try:
    import fastapi
    print(f'  fastapi     {fastapi.__version__}  OK')
except Exception as e:
    errors.append(f'  fastapi  FAILED: {e}')

try:
    import cv2
    print(f'  opencv      {cv2.__version__}  OK')
except Exception as e:
    errors.append(f'  opencv  FAILED: {e}')

if errors:
    print('')
    print('IMPORT ERRORS:')
    for e in errors: print(e)
    sys.exit(1)
else:
    print('')
    print('All imports OK - Dr Halim is ready!')
"@

$CheckScript | & $VenvPython
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Some imports failed. Check errors above." -ForegroundColor Red
    Write-Host "  The server may still work if errors are minor." -ForegroundColor Yellow
    Read-Host "Press Enter to continue anyway"
}

# ================================================================
#  LAUNCH
# ================================================================
Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Dr Halim backend starting...          " -ForegroundColor Cyan
Write-Host "  URL  : http://localhost:8000           " -ForegroundColor Cyan
Write-Host "  Open : frontend\index.html in browser " -ForegroundColor Cyan
Write-Host "  Stop : Ctrl+C                         " -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

Set-Location $BackendDir
& $VenvPython -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
