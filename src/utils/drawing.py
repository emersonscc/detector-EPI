"""Desenho das detecções e do painel de status sobre o frame."""
import cv2

GREEN = (0, 180, 0)
RED = (0, 0, 220)

LABELS_PT = {
    "helmet": "Capacete",
    "no_helmet": "Sem capacete",
    "goggles": "Oculos",
    "no_goggles": "Sem oculos",
}


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = RED if d.is_violation else GREEN
        text = f"{LABELS_PT.get(d.label, d.label)} {d.confidence:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def draw_status(frame, detections):
    """Faixa no topo: 'EPI OK' ou quantas violações foram encontradas."""
    n = sum(d.is_violation for d in detections)
    text, color = ("EPI OK", GREEN) if n == 0 else (f"ATENCAO: {n} item(ns) de EPI ausente(s)", RED)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), color, -1)
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame
