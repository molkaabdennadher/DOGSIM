@echo off
title Dr Halim — Installation des packages
echo.
echo  =============================================
echo       Dr Halim — Correctif Python 3.13
echo       Installation des packages compatibles
echo  =============================================
echo.

set PIP="%~dp0venv\Scripts\pip.exe"
set PYTHON="%~dp0venv\Scripts\python.exe"

:: Verifier que le venv existe
if not exist "%~dp0venv\Scripts\pip.exe" (
    echo  [ERREUR] venv introuvable. Lance d'abord setup_and_run.ps1
    pause
    exit /b 1
)

:: Detecter la version Python
for /f "tokens=2 delims= " %%v in ('%PYTHON% --version 2^>^&1') do set PYVER=%%v
echo  Version Python detectee : %PYVER%
echo.

:: Extraire le numero de version majeur.mineur (ex: 3.13)
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% GEQ 3 if %MINOR% GEQ 13 (
    echo  [INFO] Python 3.13+ detecte — installation des versions compatibles...
    echo.
    %PIP% install "numpy==2.2.0"
    %PIP% install "tensorflow-cpu==2.20.0"
    %PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    %PIP% install "opencv-python>=4.10.0" --force-reinstall
    %PIP% install "ultralytics" --upgrade
    %PIP% install "setuptools==68.0.0" --force-reinstall
    %PIP% install fastapi uvicorn python-multipart requests Pillow soundfile scipy
) else (
    echo  [INFO] Python 3.10/3.11/3.12 detecte — installation des versions originales...
    echo.
    %PIP% install "numpy==1.26.4"
    %PIP% install "tensorflow-cpu==2.16.2"
    %PIP% install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
    %PIP% install "opencv-python==4.9.0.80"
    %PIP% install "ultralytics==8.2.0"
    %PIP% install "setuptools==68.0.0" --force-reinstall
    %PIP% install fastapi uvicorn python-multipart requests Pillow soundfile scipy
)

echo.
echo  =============================================
echo   Installation terminee !
echo   Tu peux maintenant lancer LANCER_DEMO.bat
echo  =============================================
echo.
pause
