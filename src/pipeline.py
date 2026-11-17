"""Pipeline de processamento de um frame: pré-processamento -> detecção -> desenho.

Usado pela interface gráfica e pelo processamento em lote. O pré-processamento é
uma lista de funções frame -> frame (ex.: CLAHE, redução de ruído), aplicadas em
ordem antes da detecção; a imagem exibida/salva é a pré-processada, para que o
efeito das técnicas fique visível.
"""
import time
from dataclasses import dataclass, field

from src.utils.drawing import draw_detections, draw_status, status_of


@dataclass
class FrameResult:
    original: object          # frame BGR de entrada
    processed: object         # frame BGR após o pré-processamento
    annotated: object         # frame processado com as caixas desenhadas
    detections: list
    infer_ms: float
    status: str = "vazio"     # 'ok' | 'violacao' | 'vazio'
    violations: int = 0
    counts: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, detector):
        self.detector = detector
        self.preprocessors = []   # lista de funções frame -> frame
        self.show_conf = True

    @property
    def conf(self):
        return self.detector.conf

    @conf.setter
    def conf(self, value):
        self.detector.conf = float(value)

    def preprocess(self, frame):
        for fn in list(self.preprocessors):
            frame = fn(frame)
        return frame

    def process(self, frame):
        processed = self.preprocess(frame)
        t0 = time.perf_counter()
        detections = self.detector.detect(processed)
        infer_ms = (time.perf_counter() - t0) * 1000
        annotated = draw_detections(processed.copy(), detections, self.show_conf)
        status, n = status_of(detections)
        counts = {name: 0 for name in ("helmet", "no_helmet", "goggles", "no_goggles")}
        for d in detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return FrameResult(frame, processed, annotated, detections, infer_ms, status, n, counts)

    @staticmethod
    def with_banner(result):
        """Cópia anotada com a faixa de status no topo (usada ao salvar)."""
        return draw_status(result.annotated.copy(), result.detections)
