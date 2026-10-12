"""Ponto de entrada: detecta capacete e óculos de proteção em imagem, vídeo ou webcam."""
import argparse
from pathlib import Path

import cv2

from src.detector import EPIDetector
from src.utils.drawing import draw_detections, draw_status


def annotate(frame, detector):
    dets = detector.detect(frame)
    draw_detections(frame, dets)
    draw_status(frame, dets)
    return frame


def run_image(path, detector, save):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Não foi possível abrir a imagem: {path}")
    frame = annotate(frame, detector)
    if save:
        out = Path("results") / f"{Path(path).stem}_epi.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"Resultado salvo em {out}")
    cv2.imshow("EPI Detector", frame)
    cv2.waitKey(0)


def run_stream(source, detector):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("EPI Detector (q para sair)", annotate(frame, detector))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


def main():
    p = argparse.ArgumentParser(description="Detector de capacete e óculos de proteção")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="caminho de uma imagem")
    src.add_argument("--video", help="caminho de um vídeo")
    src.add_argument("--webcam", type=int, nargs="?", const=0, help="índice da webcam (padrão 0)")
    p.add_argument("--weights", default="models/best.pt", help="pesos do modelo treinado")
    p.add_argument("--conf", type=float, default=0.4, help="confiança mínima (0-1)")
    p.add_argument("--save", action="store_true", help="salva a imagem anotada em results/")
    args = p.parse_args()

    detector = EPIDetector(args.weights, args.conf)
    if args.image:
        run_image(args.image, detector, args.save)
    else:
        run_stream(args.video if args.video else args.webcam, detector)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
