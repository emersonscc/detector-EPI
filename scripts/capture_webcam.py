"""Captura imagens pela webcam para montar o dataset próprio.

Dois modos:
  sintese  -> você SEM EPI. Com a gravação ligada, salva automaticamente um frame a cada
              intervalo, descartando frames quase iguais ao anterior. Essas imagens são a
              base para o gerador sintético (scripts/synth_dataset.py).
  teste    -> você COM ou SEM EPI real. Cada ESPAÇO salva uma foto. Essas fotos formam o
              conjunto de TESTE REAL e devem ser anotadas à mão (Roboflow).

Teclas:  ESPAÇO = liga/desliga gravação (sintese) ou tira foto (teste)   Q = sair

O nome de cada arquivo leva a pessoa e a sessão (ex.: emerson_s2_0031.jpg). Isso permite
dividir treino/validação POR PESSOA, evitando que frames quase iguais caiam nos dois lados.

Uso:
    python scripts/capture_webcam.py --mode sintese --pessoa emerson
    python scripts/capture_webcam.py --mode teste --pessoa emerson
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def too_similar(a, b, threshold):
    """Compara versões reduzidas em cinza: diferença média abaixo do limiar = frame repetido."""
    if b is None:
        return False
    small = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)
    return float(np.mean(np.abs(small(a) - small(b)))) < threshold


def next_session(out_dir, pessoa):
    existing = [p.stem.split("_")[1] for p in out_dir.glob(f"{pessoa}_s*_*.jpg")]
    nums = [int(s[1:]) for s in existing if s[1:].isdigit()]
    return max(nums, default=0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sintese", "teste"], required=True)
    ap.add_argument("--pessoa", required=True, help="nome curto, sem espaços (ex.: emerson)")
    ap.add_argument("--out", default="data/raw/webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--intervalo", type=float, default=0.4, help="segundos entre frames no modo sintese")
    ap.add_argument("--min-diff", type=float, default=4.0, help="diferença mínima para não descartar o frame")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    session = next_session(out_dir, args.pessoa)
    prefix = f"{args.pessoa}_s{session}"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a webcam {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording, saved, skipped, last_saved, last_time = False, 0, 0, None, 0.0
    print(f"Salvando em {out_dir} como {prefix}_XXXX.jpg  (ESPAÇO: {'gravar' if args.mode == 'sintese' else 'foto'}, Q: sair)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        if args.mode == "sintese" and recording and now - last_time >= args.intervalo:
            last_time = now
            if too_similar(frame, last_saved, args.min_diff):
                skipped += 1
            else:
                cv2.imwrite(str(out_dir / f"{prefix}_{saved:04d}.jpg"), frame)
                last_saved, saved = frame.copy(), saved + 1

        view = frame.copy()
        status = (f"GRAVANDO  salvos={saved}  repetidos={skipped}" if recording else f"PAUSADO  salvos={saved}") \
            if args.mode == "sintese" else f"TESTE  fotos={saved}"
        color = (0, 0, 220) if recording else (60, 60, 60)
        cv2.rectangle(view, (0, 0), (view.shape[1], 36), color, -1)
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if args.mode == "sintese":
            cv2.putText(view, "Mexa a cabeca devagar, varie distancia e angulo", (10, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Captura (ESPACO / Q)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if args.mode == "sintese":
                recording = not recording
            else:
                cv2.imwrite(str(out_dir / f"{prefix}_{saved:04d}.jpg"), frame)
                saved += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Fim: {saved} imagens salvas" + (f", {skipped} frames repetidos descartados" if skipped else ""))


if __name__ == "__main__":
    main()
