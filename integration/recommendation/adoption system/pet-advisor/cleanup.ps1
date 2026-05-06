# cleanup.ps1 — Supprime les fichiers et dossiers inutiles de pet-advisor
# Exécuter depuis PowerShell : .\cleanup.ps1

$base = "C:\Users\Adham Ferchichi\Documents\recommendation\adoption system\pet-advisor"

function Remove-IfExists($path) {
    $full = Join-Path $base $path
    if (Test-Path $full) {
        Remove-Item -Recurse -Force $full
        Write-Host "  Supprime : $path" -ForegroundColor Yellow
    } else {
        Write-Host "  Absent   : $path" -ForegroundColor DarkGray
    }
}

Write-Host "`n=== Nettoyage pet-advisor ===" -ForegroundColor Cyan

# ── Backend legacy (remplacés par backend/maya/) ──────────────────────────────
Remove-IfExists "backend\agent.py"
Remove-IfExists "backend\feedback.py"
Remove-IfExists "backend\schemas.py"
Remove-IfExists "backend\session.py"
Remove-IfExists "backend\tools"

# ── Module shelter-assignment standalone (remplacé par Ali dans backend) ──────
Remove-IfExists "shelter-assignment"

# ── Artifacts ReID (fonctionnalité supprimée) ─────────────────────────────────
Remove-IfExists "artifacts\reID"

# ── Artifacts racine obsolètes (migrés vers sous-dossiers) ───────────────────
Remove-IfExists "artifacts\aggression_incidents.db"
Remove-IfExists "artifacts\feedback.db"
Remove-IfExists "artifacts\sessions.db"
Remove-IfExists "artifacts\aggression_outbox"

# ── Journal SQLite corrompu ───────────────────────────────────────────────────
Remove-IfExists "artifacts\aliAdvisor\shelter_assignment.db-journal"

# ── Modèles YOLO en double (garder yolov8l-pose.pt seulement) ────────────────
Remove-IfExists "artifacts\agressionDetection\yolov8l.pt"
Remove-IfExists "artifacts\agressionDetection\yolov8n.pt"

# ── Dossiers de données brutes (non utilisés par le serveur) ─────────────────
Remove-IfExists "datastes"
Remove-IfExists "datasets"

# ── Dossier uploads temporaire vide ──────────────────────────────────────────
Remove-IfExists "uploads"

# ── Tous les __pycache__ ──────────────────────────────────────────────────────
Write-Host "`n  Nettoyage __pycache__..." -ForegroundColor Cyan
Get-ChildItem -Path $base -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName
        Write-Host "  Supprime : $($_.FullName.Replace($base, ''))" -ForegroundColor Yellow
    }

# ── Fichiers .pyc isolés ──────────────────────────────────────────────────────
Get-ChildItem -Path $base -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-Item -Force $_.FullName
        Write-Host "  Supprime : $($_.FullName.Replace($base, ''))" -ForegroundColor Yellow
    }

Write-Host "`nNettoyage termine." -ForegroundColor Green
