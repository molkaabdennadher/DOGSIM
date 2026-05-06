# ======================================================================
# setup_ollama_llama3.ps1
# Télécharge et configure Ollama3 + llama3 pour Dr Halim
# ======================================================================

param(
    [switch]$SkipDownload = $false,
    [switch]$JustServe = $false
)

# ======================================================================
#  COULEURS
# ======================================================================
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Error-Msg { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

# ======================================================================
#  VÉRIFIER OLLAMA
# ======================================================================
Write-Info "🔍 Vérification d'Ollama..."

if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error-Msg "❌ Ollama n'est pas installé ou non accessible."
    Write-Host ""
    Write-Host "Solution:"
    Write-Host "  1. Téléchargez https://ollama.ai"
    Write-Host "  2. Installez l'application"
    Write-Host "  3. Redémarrez votre PC"
    Write-Host "  4. Réexécutez ce script"
    exit 1
}

$ollama_version = ollama --version
Write-Success "✅ Ollama trouvé: $ollama_version"

# ======================================================================
#  TÉLÉCHARGER LE MODÈLE LLAMA3
# ======================================================================
if ($JustServe) {
    Write-Info "`n⏭️  Mode 'JustServe' - passer le téléchargement"
} else {
    Write-Info "`n📥 Téléchargement de llama3 (peut prendre 5-15 minutes)..."
    Write-Host "   Taille: ~4.7 GB"
    Write-Host ""
    
    try {
        ollama pull llama3
        Write-Success "✅ llama3 téléchargé avec succès"
    } catch {
        Write-Error-Msg "❌ Erreur lors du téléchargement de llama3"
        Write-Error-Msg $_
        exit 1
    }
}

# ======================================================================
#  VÉRIFIER LES MODÈLES DISPONIBLES
# ======================================================================
Write-Info "`n📦 Modèles disponibles:"
ollama list

# ======================================================================
#  TESTER LA CONNEXION API
# ======================================================================
Write-Info "`n🌐 Test de connexion API..."

$max_retries = 5
$retry_count = 0
$api_ready = $false

while ($retry_count -lt $max_retries) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" `
                                      -TimeoutSec 5 `
                                      -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $api_ready = $true
            break
        }
    } catch {
        # API pas encore prête
    }
    $retry_count++
    if ($retry_count -lt $max_retries) {
        Write-Host "   ⏳ Tentative $retry_count/$max_retries - attente de l'API..."
        Start-Sleep -Seconds 2
    }
}

if ($api_ready) {
    Write-Success "✅ API Ollama accessible sur http://127.0.0.1:11434"
} else {
    Write-Error-Msg "❌ API Ollama non accessible sur http://127.0.0.1:11434"
    Write-Host ""
    Write-Host "Vérifiez que:"
    Write-Host "  1. Ollama est en cours d'exécution (ollama serve)"
    Write-Host "  2. Aucun autre service n'utilise le port 11434"
}

# ======================================================================
#  AFFICHER LES PARAMÈTRES DE CONFIGURATION
# ======================================================================
Write-Info "`n⚙️  Configuration pour Dr Halim backend/main.py:"
Write-Host "   OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'"
Write-Host "   Modèle utilisé: llama3"
Write-Host "   Timeout: 120 secondes (peut être augmenté si lent)"

# ======================================================================
#  INSTRUCTIONS DE DÉMARRAGE
# ======================================================================
Write-Info "`n🚀 Prochaines étapes:"
Write-Host ""
Write-Host "1️⃣  Démarrer Ollama en arrière-plan (PowerShell Administrateur):"
Write-Host "   ollama serve"
Write-Host ""
Write-Host "2️⃣  Dans un autre terminal, démarrer le backend Dr Halim:"
Write-Host "   cd adoption_system/dr_halim"
Write-Host "   python backend/main.py"
Write-Host ""
Write-Host "3️⃣  Tester l'API (dans un 3e terminal):"
Write-Host "   \`curl -X POST 'http://localhost:8000/predict-breed' -F 'file=@dog.jpg'\`"
Write-Host ""

# ======================================================================
#  OPTION: DÉMARRER OLLAMA SERVE
# ======================================================================
Write-Host ""
$start_serve = Read-Host "Démarrer 'ollama serve' maintenant ? (y/n) [n]"

if ($start_serve -eq "y" -or $start_serve -eq "Y") {
    Write-Info "`n🔄 Démarrage de 'ollama serve'..."
    Write-Host "   ✋ Appuyez sur Ctrl+C pour arrêter plus tard"
    Write-Host ""
    Write-Host "   API disponible sur: http://127.0.0.1:11434"
    Write-Host ""
    ollama serve
} else {
    Write-Host ""
    Write-Success "✅ Configuration terminée!"
    Write-Host "   Exécutez 'ollama serve' manuellement pour démarrer l'API"
}
