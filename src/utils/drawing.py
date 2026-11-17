"""Desenho das detecções e do painel de status sobre o frame.

Espessura das linhas e tamanho do texto são proporcionais à resolução da imagem,
para ficarem legíveis tanto em 640x480 quanto em fotos de celular de 4000 px.
"""
import cv2

GREEN = (0, 180, 0)
RED = (0, 0, 220)
GRAY = (110, 110, 110)

LABELS_PT = {
    "helmet": "Capacete",
    "no_helmet": "Sem capacete",
    "goggles": "Oculos",
    "no_goggles": "Sem oculos",
}


def _scale(frame):
    """Fator de escala do desenho em relação a uma imagem de ~1000 px."""
    return max(frame.shape[:2]) / 1000.0


def draw_detections(frame, detections, show_conf=True):
    s = _scale(frame)
    font_scale = max(0.45, 0.6 * s)
    thick = max(1, round(2 * s))
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = RED if d.is_violation else GREEN
        text = LABELS_PT.get(d.label, d.label) + (f" {d.confidence:.0%}" if show_conf else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick // 2))
        pad = max(3, round(4 * s))
        # rótulo acima da caixa; se não couber (caixa no topo), vai para dentro dela
        ty = y1 - pad if y1 - th - 2 * pad >= 0 else y1 + th + pad
        cv2.rectangle(frame, (x1, ty - th - pad), (x1 + tw + 2 * pad, ty + pad), color, -1)
        cv2.putText(frame, text, (x1 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
    return frame


def status_of(detections):
    """('ok' | 'violacao' | 'vazio', número de violações)."""
    if not detections:
        return "vazio", 0
    n = sum(d.is_violation for d in detections)
    return ("violacao" if n else "ok"), n


def draw_status(frame, detections):
    """Faixa no topo: 'EPI OK', quantas violações foram encontradas, ou nenhuma pessoa."""
    s = _scale(frame)
    state, n = status_of(detections)
    text, color = {
        "ok": ("EPI OK", GREEN),
        "violacao": (f"ATENCAO: {n} item(ns) de EPI ausente(s)", RED),
        "vazio": ("Nenhuma pessoa detectada", GRAY),
    }[state]
    h = max(32, round(40 * s))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), color, -1)
    cv2.putText(frame, text, (round(10 * s) + 4, round(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, 0.8 * s), (255, 255, 255), max(2, round(2 * s)), cv2.LINE_AA)
    return frame
