# ============================================================
#  DETECTOR — Détection YOLO (chiens + humains)
# ============================================================

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    """Résultat d'une détection YOLO."""
    class_id:    int
    label:       str              # "dog" ou "person"
    confidence:  float
    bbox:        tuple            # (x1, y1, x2, y2) en pixels
    centroid:    tuple            # (cx, cy)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height


class YOLODetector:
    """
    Détecteur YOLO pour chiens et humains.
    Utilise ultralytics YOLOv8.
    """

    DOG_CLASS    = 16
    PERSON_CLASS = 0

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.45,
                 iou: float = 0.45):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics non installé. Lance : pip install ultralytics"
            )

        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou        = iou
        print(f"[Detector] Modèle YOLO chargé : {model_path}")

    def detect(self, frame: np.ndarray,
               detect_dogs: bool = True,
               detect_persons: bool = True) -> list[Detection]:
        """
        Lancer la détection sur une frame.

        Args:
            frame           : image BGR (numpy array)
            detect_dogs     : inclure les chiens
            detect_persons  : inclure les humains

        Returns:
            Liste de Detection
        """
        target_classes = []
        if detect_dogs:
            target_classes.append(self.DOG_CLASS)
        if detect_persons:
            target_classes.append(self.PERSON_CLASS)

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=target_classes,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                label = "dog" if cls_id == self.DOG_CLASS else "person"

                detections.append(Detection(
                    class_id   = cls_id,
                    label      = label,
                    confidence = conf,
                    bbox       = (x1, y1, x2, y2),
                    centroid   = (cx, cy)
                ))

        return detections

    def detect_dogs(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame, detect_dogs=True, detect_persons=False)

    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame, detect_dogs=False, detect_persons=True)
