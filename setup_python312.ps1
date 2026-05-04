$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"
$PythonExe = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        $version = & py -3.12 --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python 3\.12\.") {
            $PythonExe = "py -3.12"
        }
    } catch {}
}

if (-not $PythonExe) {
    $candidate = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($candidate) {
        $version = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python 3\.12\.") {
            $PythonExe = "`"$candidate`""
        }
    }
}

if (-not $PythonExe) {
    Write-Host "Python 3.12 was not found. Install it, then run this script again." -ForegroundColor Red
    exit 1
}

if (Test-Path $VenvDir) {
    $cfg = Join-Path $VenvDir "pyvenv.cfg"
    if ((Test-Path $cfg) -and (Select-String -Path $cfg -Pattern "version = 3\.12\." -Quiet)) {
        Write-Host "Existing .venv already uses Python 3.12." -ForegroundColor Green
    } else {
        Write-Host "Removing existing .venv because it is not Python 3.12..." -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating .venv with Python 3.12..." -ForegroundColor Cyan
    Invoke-Expression "$PythonExe -m venv `"$VenvDir`""
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
