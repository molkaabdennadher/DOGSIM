@echo off
cd /d "%~dp0"
echo Nettoyage pet-advisor...

rmdir /s /q "backend\__pycache__"
rmdir /s /q "backend\aggression\__pycache__"
rmdir /s /q "backend\ali\__pycache__"
rmdir /s /q "backend\maya\__pycache__"
rmdir /s /q "backend\maya\tools\__pycache__"
rmdir /s /q "backend\tools\__pycache__"
del /f /q "backend\agent.py"
del /f /q "backend\feedback.py"
del /f /q "backend\schemas.py"
del /f /q "backend\session.py"
rmdir /s /q "backend\tools"
rmdir /s /q "shelter-assignment"
rmdir /s /q "artifacts\reID"
rmdir /s /q "artifacts\aggression_outbox"
del /f /q "artifacts\aggression_incidents.db"
del /f /q "artifacts\feedback.db"
del /f /q "artifacts\sessions.db"
del /f /q "artifacts\aliAdvisor\shelter_assignment.db-journal"
del /f /q "artifacts\agressionDetection\yolov8l.pt"
del /f /q "artifacts\agressionDetection\yolov8n.pt"
rmdir /s /q "datastes"
rmdir /s /q "uploads"

echo Fait.
pause