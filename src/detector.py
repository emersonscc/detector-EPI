"""Detector de EPI (capacete e óculos) baseado em YOLO."""
from dataclasses import dataclass

from ultralytics import YOLO

# classes que indicam ausência de EPI (violação)
VIOLATIONS = {"no_helmet", "no_goggles"}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2)

    @property
    def is_violation(self):
        return self.label in VIOLATIONS


class EPIDetector:
    def __init__(self, weights="models/best.pt", conf=0.4):
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        """Roda o modelo em um frame BGR e devolve a lista de detecções."""
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = result.names[int(box.cls[0])]
            detections.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return detections
