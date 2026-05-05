# ======================================================================
# start_dr_halim_with_ollama.ps1
# Démarrage complet: Ollama + Backend Dr Halim en 2 terminaux
# ======================================================================

param(
    [switch]$NoOllama = $false,  # Skip Ollama, ne lancer que le backend
    [switch]$NoBackend = $false  # Skip backend, ne lancer qu'Ollama
)

# ======================================================================
#  CONFIGURATION
# ======================================================================
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND_DIR = Join-Path $SCRIPT_DIR "backend"
$PYTHON_SCRIPT = Join-Path $BACKEND_DIR "main.py"

# ======================================================================
#  COULEURS & FONCTIONS
# ======================================================================
function Write-Banner {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Dr Halim - AI Veterinary System avec Ollama3 llama3      ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success { Write-Host $args -ForegroundColor Green -BackgroundColor Black }
function Write-Error-Msg { Write-Host $args -ForegroundColor Red -BackgroundColor Black }
function Write-Info { Write-Host $args -ForegroundColor Cyan -BackgroundColor Black }
function Write-Step { Write-Host "`n$args" -ForegroundColor Yellow -BackgroundColor Black }

# ======================================================================
#  VÉRIFICATIONS
# ======================================================================
Write-Banner

# Vérifier Python
Write-Step "1️⃣  Vérification de Python..."
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error-Msg "❌ Python n'est pas installé"
    exit 1
}
$python_version = python --version 2>&1
Write-Success "✅ Python trouvé: $python_version"

# Vérifier le fichier main.py
if (!(Test-Path $PYTHON_SCRIPT)) {
    Write-Error-Msg "❌ Fichier non trouvé: $PYTHON_SCRIPT"
    exit 1
}
Write-Success "✅ Backend trouvé: $PYTHON_SCRIPT"

# Vérifier les dépendances
Write-Step "2️⃣  Vérification des dépendances..."
try {
    python -c "import fastapi; import torch; import tensorflow; import requests" 2>&1 | Out-Null
    Write-Success "✅ Dépendances principales présentes"
} catch {
    Write-Error-Msg "❌ Dépendances manquantes. Exécutez:"
    Write-Host "   cd adoption_system/dr_halim"
    Write-Host "   pip install -r requirements.txt"
    exit 1
}

# ======================================================================
#  CONFIGURATION OLLAMA
# ======================================================================
if (!$NoOllama) {
    Write-Step "3️⃣  Vérification d'Ollama..."
    
    if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Error-Msg "❌ Ollama n'est pas installé"
        Write-Host ""
        Write-Host "Solution:"
        Write-Host "  1. Téléchargez https://ollama.ai"
        Write-Host "  2. Installez l'application"
        Write-Host "  3. Redémarrez"
        Write-Host "  4. Réexécutez ce script"
        exit 1
    }
    Write-Success "✅ Ollama installé"
    
    # Vérifier llama3
    $models_raw = ollama list 2>&1
    if ($models_raw -like "*llama3*") {
        Write-Success "✅ Modèle llama3 trouvé"
    } else {
        Write-Info "⚠️  llama3 n'est pas encore téléchargé"
        Write-Host "   Téléchargement en arrière-plan (peut prendre 5-15 min)..."
        ollama pull llama3
    }
}

# ======================================================================
#  LANCEMENT DES SERVICES
# ======================================================================
Write-Step "4️⃣  Démarrage des services..."

if (!$NoOllama) {
    Write-Info "   🔄 Lancement d'Ollama dans un nouveau terminal..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve; Write-Host 'Appuyez sur une touche pour fermer...'; `$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
    Start-Sleep -Seconds 2
}

if (!$NoBackend) {
    Write-Info "   🔄 Lancement du backend Dr Halim dans un nouveau terminal..."
    $backend_cmd = "cd '$SCRIPT_DIR'; python backend/main.py; Write-Host 'Appuyez sur une touche pour fermer...'; `$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backend_cmd
    Start-Sleep -Seconds 2
}

# ======================================================================
#  AFFICHER LES URLS
# ======================================================================
Write-Step "5️⃣  Services actifs:"
Write-Host ""

if (!$NoOllama) {
    Write-Host "   🔷 Ollama API:" -ForegroundColor Cyan
    Write-Host "      http://127.0.0.1:11434" -ForegroundColor Yellow
    Write-Host "      Modèle: llama3" -ForegroundColor Yellow
    Write-Host ""
}

if (!$NoBackend) {
    Write-Host "   🟦 Dr Halim Backend:" -ForegroundColor Cyan
    Write-Host "      http://localhost:8000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Endpoints:" -ForegroundColor Cyan
    Write-Host "      POST /predict-breed    → Détection de race + recommandation" -ForegroundColor Yellow
    Write-Host "      POST /predict-skin     → Analyse de peau" -ForegroundColor Yellow
    Write-Host "      POST /predict-behavior → Analyse de comportement" -ForegroundColor Yellow
    Write-Host "      POST /predict-fecal    → Analyse des selles" -ForegroundColor Yellow
    Write-Host "      GET  /               → Health check" -ForegroundColor Yellow
    Write-Host ""
}

# ======================================================================
#  INSTRUCTIONS
# ======================================================================
Write-Step "🚀 Prochaines étapes:"
Write-Host ""
Write-Host "   1️⃣  Attendez que les deux terminaux affichent 'Application startup complete'" -ForegroundColor White
Write-Host ""
Write-Host "   2️⃣  Testez l'API dans un 3e terminal:" -ForegroundColor White
Write-Host "       python test_breed_detection.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   3️⃣  Ou avec curl:" -ForegroundColor White
Write-Host "       curl -X POST http://localhost:8000/predict-breed -F file=@dog.jpg" -ForegroundColor Gray
Write-Host ""
Write-Host "   4️⃣  Fermez un terminal en appuyant sur Ctrl+C (puis Enter)" -ForegroundColor White
Write-Host ""

# ======================================================================
#  ATTENDRE LA FIN
# ======================================================================
Write-Host "Appuyez sur une touche pour fermer cette fenêtre..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
