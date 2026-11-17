"""Tarefas demoradas executadas fora da thread da interface (QThread).

- ModelLoader: carrega o YOLO e faz uma inferência de aquecimento.
- StreamWorker: lê vídeo ou webcam quadro a quadro, com pausa e gravação.
- BatchWorker: processa uma pasta inteira, salva as imagens anotadas e um CSV.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.pipeline import Pipeline
from src.utils.io import list_images, read_image, write_image


def to_qimage(bgr):
    """Converte um frame BGR do OpenCV em QImage (cópia independente do buffer NumPy)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ModelLoader(QThread):
    loaded = Signal(object)   # Pipeline
    failed = Signal(str)

    def __init__(self, weights, conf):
        super().__init__()
        self.weights, self.conf = weights, conf

    def run(self):
        try:
            from src.detector import EPIDetector  # importa o Ultralytics/PyTorch aqui (lento)
            detector = EPIDetector(self.weights, self.conf)
            detector.detect(np.zeros((480, 640, 3), np.uint8))  # aquecimento
            self.loaded.emit(Pipeline(detector))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class StreamWorker(QThread):
    frame_ready = Signal(QImage, object, dict)   # imagem, FrameResult, estatísticas
    message = Signal(str)
    finished_stream = Signal(str)                # motivo do fim

    def __init__(self, pipeline, source, is_webcam):
        super().__init__()
        self.pipeline = pipeline
        self.source = source
        self.is_webcam = is_webcam
        self._running = True
        self._paused = False
        self._record_path = None
        self._writer = None
        self._snapshot_path = None
        self.last_result = None

    # ---- controle (chamado pela interface) ----
    def stop(self):
        self._running = False

    def set_paused(self, paused):
        self._paused = paused

    def start_recording(self, path):
        self._record_path = str(path)

    def stop_recording(self):
        self._record_path = None

    @property
    def recording(self):
        return self._record_path is not None

    # ---- laço principal ----
    def _open(self):
        if self.is_webcam:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(str(self.source))
        return cap

    def run(self):
        cap = self._open()
        if not cap.isOpened():
            what = f"a webcam {self.source}" if self.is_webcam else f"o vídeo {Path(str(self.source)).name}"
            self.finished_stream.emit(f"Não foi possível abrir {what}.")
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else 0
        idx, fps, last = 0, 0.0, time.perf_counter()
        reason = "Transmissão encerrada."

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok:
                reason = "Fim do vídeo." if not self.is_webcam else "A webcam parou de enviar imagens."
                break
            idx += 1
            result = self.pipeline.process(frame)
            self.last_result = result
            self._handle_recording(result, fps_src)

            now = time.perf_counter()
            dt = now - last
            last = now
            fps = 0.9 * fps + 0.1 * (1 / dt) if fps else 1 / max(dt, 1e-6)
            stats = {"fps": fps, "infer_ms": result.infer_ms, "frame": idx, "total": total,
                     "size": (frame.shape[1], frame.shape[0]), "recording": self.recording}
            self.frame_ready.emit(to_qimage(result.annotated), result, stats)

            if not self.is_webcam:  # não passa da velocidade original do vídeo
                wait = 1 / fps_src - (time.perf_counter() - now)
                if wait > 0:
                    time.sleep(wait)

        cap.release()
        self._close_writer()
        self.finished_stream.emit(reason)

    def _handle_recording(self, result, fps_src):
        if self._record_path and self._writer is None:
            h, w = result.annotated.shape[:2]
            self._writer = cv2.VideoWriter(self._record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
            self.message.emit(f"Gravando em {self._record_path}")
        elif not self._record_path and self._writer is not None:
            self._close_writer()
        if self._writer is not None:
            self._writer.write(Pipeline.with_banner(result))

    def _close_writer(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.message.emit("Gravação salva.")


class BatchWorker(QThread):
    progress = Signal(int, int, str)          # atual, total, nome do arquivo
    preview = Signal(QImage, object)          # imagem anotada, FrameResult
    done = Signal(str, dict)                  # caminho do CSV, resumo

    def __init__(self, pipeline, folder, out_dir):
        super().__init__()
        self.pipeline, self.folder, self.out_dir = pipeline, Path(folder), Path(out_dir)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        files = list_images(self.folder)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.out_dir / "relatorio_epi.csv"
        summary = {"imagens": 0, "ok": 0, "violacao": 0, "vazio": 0, "erros": 0}
        names = {"ok": "EPI OK", "violacao": "EPI AUSENTE", "vazio": "SEM PESSOAS"}

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel abre com acentos
            w = csv.writer(f, delimiter=";")
            w.writerow(["arquivo", "status", "capacete", "sem_capacete", "oculos", "sem_oculos",
                        "violacoes", "confianca_minima"])
            for i, path in enumerate(files, 1):
                if not self._running:
                    break
                self.progress.emit(i, len(files), path.name)
                try:
                    result = self.pipeline.process(read_image(path))
                except Exception:  # noqa: BLE001 - arquivo corrompido não interrompe o lote
                    summary["erros"] += 1
                    w.writerow([path.name, "ERRO DE LEITURA", "", "", "", "", "", ""])
                    continue
                write_image(self.out_dir / f"{path.stem}_epi.jpg", Pipeline.with_banner(result))
                c = result.counts
                w.writerow([path.name, names[result.status], c["helmet"], c["no_helmet"], c["goggles"],
                            c["no_goggles"], result.violations, f"{self.pipeline.conf:.2f}".replace(".", ",")])
                summary["imagens"] += 1
                summary[result.status] += 1
                self.preview.emit(to_qimage(result.annotated), result)
        self.done.emit(str(csv_path), summary)
