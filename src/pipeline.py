"""Pipeline de processamento de um frame: pré-processamento -> detecção -> desenho.

Usado pela interface gráfica e pelo processamento em lote. O pré-processamento
(src/preprocess.py: balanço de branco, gama, CLAHE, filtro bilateral, nitidez) é
aplicado antes da detecção; a imagem exibida/salva é a pré-processada, para que o
efeito das técnicas fique visível.
"""
import time
from dataclasses import dataclass, field

from src.preprocess import Preprocessor
from src.utils.drawing import draw_detections, draw_status, status_of


@dataclass
class FrameResult:
    original: object          # frame BGR de entrada
    processed: object         # frame BGR após o pré-processamento
    annotated: object         # frame processado com as caixas desenhadas
    detections: list
    infer_ms: float
    prep_ms: float = 0.0      # tempo do pré-processamento
    status: str = "vazio"     # 'ok' | 'violacao' | 'vazio'
    violations: int = 0
    counts: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, detector):
        self.detector = detector
        self.preprocessor = Preprocessor()
        self.preprocessors = [self.preprocessor]   # funções frame -> frame, aplicadas em ordem
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
        t_prep = time.perf_counter()
        processed = self.preprocess(frame)
        prep_ms = (time.perf_counter() - t_prep) * 1000
        t0 = time.perf_counter()
        detections = self.detector.detect(processed)
        infer_ms = (time.perf_counter() - t0) * 1000
        annotated = draw_detections(processed.copy(), detections, self.show_conf)
        status, n = status_of(detections)
        counts = {name: 0 for name in ("helmet", "no_helmet", "goggles", "no_goggles")}
        for d in detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return FrameResult(frame, processed, annotated, detections, infer_ms, prep_ms, status, n, counts)

    @staticmethod
    def with_banner(result):
        """Cópia anotada com a faixa de status no topo (usada ao salvar)."""
        return draw_status(result.annotated.copy(), result.detections)
