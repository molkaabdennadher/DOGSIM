# ============================================================
#  CV MODULE — Configuration centrale
#  Stray Dogs Tunisia | ESPRIT 3A IA
# ============================================================

# --- Modèle YOLO ---
YOLO_MODEL = "yolov8n.pt"          # yolov8n (nano) = rapide, bon pour prototype
                                    # Remplacer par yolov8m.pt pour + de précision
YOLO_CONFIDENCE = 0.45             # Seuil de confiance minimum
YOLO_IOU = 0.45                    # Seuil IoU pour NMS

# Classes COCO utilisées
CLASS_DOG    = 16                  # index COCO : "dog"
CLASS_PERSON = 0                   # index COCO : "person"

# --- Buffer vidéo ---
BUFFER_SECONDS = 60               # On garde les 60 dernières secondes en mémoire
                                   # Ajuster selon la RAM disponible

# --- Détection de stationnarité ---
STATIONARITY_THRESHOLD_PX  = 60   # Rayon (pixels) dans lequel le chien doit rester
STATIONARITY_MIN_SECONDS   = 8    # Durée minimale de stagnation pour déclencher l'alerte
                                   # (8s = compromis faux positifs / vrais positifs)

# --- Backtrack ---
BACKTRACK_WINDOW_SECONDS = 45     # On remonte jusqu'à 45s avant la stagnation
BACKTRACK_SEARCH_RADIUS_PX = 200  # Rayon de recherche de l'humain autour du chien

# --- Tracking ---
MAX_LOST_FRAMES = 15              # Nombre de frames avant de considérer un objet perdu
IOU_MATCH_THRESHOLD = 0.3         # Seuil IoU pour associer détections aux tracks

# --- Rapport d'infraction ---
OUTPUT_DIR = "cv_module/output"
SAVE_ANNOTATED_FRAME = True        # Sauvegarder la frame annotée
BLUR_FACES = False                 # Ne PAS flouter : le visage est la preuve de l'infraction
SAVE_ACTION_SEQUENCE = True        # Sauvegarder les N frames autour du moment d'action

# --- Caméra ---
# Pour ce module : on part du principe que la caméra est HORS feeding zone
# => toute détection de nourrissage = infraction
CAMERA_IN_FEEDING_ZONE = False
