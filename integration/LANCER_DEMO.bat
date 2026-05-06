@echo off
title DOGSIM-TN — Lancement Demo
echo.
echo  =============================================
echo       DOGSIM-TN  x  Happy Paws
echo       Lancement de la demo...
echo  =============================================
echo.

:: 1. DOGSIM-TN (Flask, port 8080)
echo  [1/2] Demarrage DOGSIM-TN sur http://localhost:8080 ...
start "DOGSIM-TN Server" cmd /k "cd /d "%~dp0" && python server.py"

:: 2. Happy Paws (uvicorn, port 8000) — Dr Halim demarre automatiquement
echo  [2/2] Demarrage Happy Paws sur http://localhost:8000 ...
start "Happy Paws Server" cmd /k "cd /d "%~dp0recommendation\adoption system\pet-advisor" && .venv\Scripts\uvicorn backend.main:app --port 8000"

:: Attendre que les serveurs demarrent
echo.
echo  Attente du demarrage des serveurs (8 secondes)...
timeout /t 8 /nobreak > nul

:: Ouvrir le navigateur sur DOGSIM-TN
echo  Ouverture du navigateur...
start http://localhost:8080

echo.
echo  =============================================
echo   Tout est lance !
echo   DOGSIM-TN  : http://localhost:8080
echo   Happy Paws : http://localhost:8000
echo   Dr. Halim  : demarre automatiquement
echo.
echo   Pour arreter : ferme les 2 fenetres noires
echo  =============================================
echo.
pause
