"""Gera um dataset sintético colando capacetes e óculos em fotos de rostos SEM EPI.

Pipeline de PDI aplicado a cada foto:
1. Detecção do rosto e dos olhos (classificadores Haar do OpenCV).
2. Geometria da cabeça: centro e distância entre os olhos, inclinação (roll) da cabeça.
3. Transformação afim do recorte RGBA (escala + rotação + translação) para encaixar
   o óculos sobre os olhos e o capacete sobre a testa.
4. Harmonização de cor/brilho do objeto com a cena, leve desfoque e ruído.
5. Composição por canal alfa (alpha blending).
6. Rótulos YOLO gerados automaticamente, nas 4 classes do projeto:
   0 helmet, 1 no_helmet, 2 goggles, 3 no_goggles

Cada foto gera até 4 variações: sem EPI, só capacete, só óculos, capacete + óculos.

Uso:
    python scripts/synth_dataset.py --frames data/raw/webcam/sintese --assets assets/epi \\
        --out data/synthetic --preview 20
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

HELMET, NO_HELMET, GOGGLES, NO_GOGGLES = 0, 1, 2, 3
COMBOS = [(False, False), (True, False), (False, True), (True, True)]  # (capacete, óculos)
NAMES = ["helmet", "no_helmet", "goggles", "no_goggles"]
COLORS = [(0, 180, 0), (0, 0, 220), (200, 120, 0), (0, 120, 255)]

FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FACE_ALT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ---------------------------------------------------------------- detecção

def detect_face(gray):
    """Maior rosto frontal da imagem (x, y, w, h) ou None."""
    h, w = gray.shape
    min_size = (int(0.12 * min(h, w)),) * 2
    for cascade in (FACE, FACE_ALT):
        faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=min_size)
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    return None


def detect_eyes(gray, face):
    """Centros do olho esquerdo e direito (na imagem). Se o Haar falhar,
    usa as proporções típicas do rosto."""
    x, y, w, h = face
    roi = gray[y:y + int(0.55 * h), x:x + w]
    eyes = EYE.detectMultiScale(roi, 1.1, 5, minSize=(w // 10, w // 10), maxSize=(w // 3, w // 3))
    centers = sorted(((x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes), key=lambda c: c[0])
    best = None
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (lx, ly), (rx, ry) = centers[i], centers[j]
            d = math.hypot(rx - lx, ry - ly)
            # olhos plausíveis: um de cada lado, distância 25-60% da largura, pouca diferença de altura
            if lx < x + w / 2 < rx and 0.25 * w < d < 0.6 * w and abs(ry - ly) < 0.2 * w:
                if best is None or abs(d - 0.42 * w) < abs(best[2] - 0.42 * w):
                    best = ((lx, ly), (rx, ry), d)
    if best:
        return best[0], best[1], True
    return (x + 0.3 * w, y + 0.4 * h), (x + 0.7 * w, y + 0.4 * h), False


# ---------------------------------------------------------------- composição

def harmonize(rgba, scene, region, rng):
    """Ajusta brilho e tom do objeto à iluminação da cena (região do rosto)."""
    x, y, w, h = region
    patch = scene[max(y, 0):y + h, max(x, 0):x + w].reshape(-1, 3).astype(np.float32)
    mean = patch.mean(0)
    gain = np.clip((mean.mean() / 140.0) ** 0.6, 0.55, 1.25)
    cast = mean / max(mean.mean(), 1)  # "dominante de cor" da cena
    rgb = rgba[..., :3].astype(np.float32) * gain * (0.7 + 0.3 * cast)
    rgb *= rng.uniform(0.9, 1.1)
    return np.dstack([np.clip(rgb, 0, 255), rgba[..., 3]]).astype(np.uint8)


def warp_onto(scene_shape, rgba, anchor, target, angle, scale):
    """Escala e gira o recorte em torno de 'anchor' (coords do recorte) e o leva até 'target'
    (coords da cena). Devolve uma imagem RGBA do tamanho da cena."""
    M = cv2.getRotationMatrix2D(anchor, angle, scale)
    M[:, 2] += np.array(target) - np.array(anchor)
    return cv2.warpAffine(rgba, M, (scene_shape[1], scene_shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def blend(scene, layer, rng):
    """Alpha blending com leve desfoque (borda natural) e ruído no objeto."""
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        layer = cv2.GaussianBlur(layer, (k, k), 0)
    a = layer[..., 3:4].astype(np.float32) / 255
    obj = layer[..., :3].astype(np.float32) + rng.normal(0, 2.5, layer[..., :3].shape)
    return np.clip(obj * a + scene.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def bbox(mask):
    ys, xs = np.where(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def shell_anchor(rgba):
    """Ponto de apoio do capacete: centro da aba (linha mais larga do casco).
    Alças finas abaixo da aba são ignoradas."""
    opaque = rgba[..., 3] > 128
    widths = opaque.sum(1)
    shell_rows = np.where(widths >= 0.6 * widths.max())[0]
    bottom = shell_rows.max()
    h, w = opaque.shape
    if h > w:  # recorte mais alto que largo = capacete com alças: o casco ocupa só o topo
        bottom = min(bottom, int(0.72 * w))
        shell_rows = shell_rows[shell_rows <= bottom]
    xs = np.where(opaque[shell_rows].any(0))[0]
    return float((xs.min() + xs.max()) / 2), float(bottom)


def maybe_flip(rgba, rng):
    return rgba[:, ::-1].copy() if rng.random() < 0.5 else rgba


# ---------------------------------------------------------------- geração

def synthesize(scene, face, eyes, helmet, goggles, rng):
    """Cola os EPIs escolhidos e devolve (imagem, lista de rótulos (classe, x1, y1, x2, y2))."""
    fx, fy, fw, fh = face
    (lx, ly), (rx, ry) = eyes
    d = math.hypot(rx - lx, ry - ly)
    ex, ey = (lx + rx) / 2, (ly + ry) / 2
    roll = -math.degrees(math.atan2(ry - ly, rx - lx))
    out = scene.copy()
    labels = []

    if goggles is not None:
        g = harmonize(maybe_flip(goggles, rng), scene, face, rng)
        target_w = fw * rng.uniform(0.95, 1.15)
        scale = target_w / g.shape[1]
        anchor = (g.shape[1] / 2, g.shape[0] / 2)
        target = (ex + rng.uniform(-0.05, 0.05) * d, ey + rng.uniform(0.0, 0.12) * d)
        layer = warp_onto(scene.shape, g, anchor, target, roll + rng.uniform(-4, 4), scale)
        out = blend(out, layer, rng)
        gb = bbox(layer[..., 3] > 50)
        if gb:  # caixa limitada à região dos olhos (hastes laterais ficam de fora)
            x1, y1, x2, y2 = gb
            labels.append((GOGGLES, max(x1, lx - 0.8 * d), max(y1, ey - 0.6 * d),
                           min(x2, rx + 0.8 * d), min(y2, ey + 0.6 * d)))
    else:
        labels.append((NO_GOGGLES, lx - 0.6 * d, ey - 0.35 * d, rx + 0.6 * d, ey + 0.35 * d))

    if helmet is not None:
        hm = harmonize(maybe_flip(helmet, rng), scene, face, rng)
        target_w = fw * rng.uniform(1.4, 1.65)
        scale = target_w / hm.shape[1]
        anchor = shell_anchor(hm)
        # aba sobre a testa, acima dos olhos
        target = (fx + fw / 2 + rng.uniform(-0.05, 0.05) * fw, ey - rng.uniform(0.6, 0.85) * d)
        layer = warp_onto(scene.shape, hm, anchor, target, roll + rng.uniform(-6, 6), scale)
        out = blend(out, layer, rng)
        hb = bbox(layer[..., 3] > 128)
        if hb:  # convenção do dataset original: caixa da cabeça inteira, capacete até o queixo
            labels.append((HELMET, min(hb[0], fx), hb[1], max(hb[2], fx + fw), fy + 0.98 * fh))
    else:
        labels.append((NO_HELMET, fx - 0.08 * fw, fy - 0.3 * fh, fx + 1.08 * fw, fy + 0.98 * fh))

    return out, labels


def to_yolo(labels, w, h):
    lines = []
    for cls, x1, y1, x2, y2 in labels:
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def draw(img, labels):
    img = img.copy()
    for cls, x1, y1, x2, y2 in labels:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, COLORS[cls], 2)
        cv2.putText(img, NAMES[cls], (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls], 2)
    return img


def load_assets(folder):
    files = sorted(Path(folder).glob("*.png"))
    assets = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in files]
    assets = [a for a in assets if a is not None and a.ndim == 3 and a.shape[2] == 4]
    if not assets:
        raise SystemExit(f"Nenhum PNG com transparência em {folder}")
    return assets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="pasta com fotos de rostos sem EPI")
    ap.add_argument("--assets", default="assets/epi", help="pasta com capacete/ e oculos/ (PNGs RGBA)")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--variantes", type=int, default=4, help="imagens geradas por foto (1-4)")
    ap.add_argument("--preview", type=int, default=0, help="salva N imagens com as caixas desenhadas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    helmets = load_assets(Path(args.assets) / "capacete")
    goggles = load_assets(Path(args.assets) / "oculos")
    out = Path(args.out)
    for sub in ("images", "labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    stats = {"fotos": 0, "sem_rosto": 0, "olhos_estimados": 0, "geradas": 0}
    previews = 0
    for f in frames:
        scene = cv2.imread(str(f))
        if scene is None:
            continue
        stats["fotos"] += 1
        gray = cv2.equalizeHist(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
        face = detect_face(gray)
        if face is None:
            stats["sem_rosto"] += 1
            continue
        le, re_, found = detect_eyes(gray, face)
        stats["olhos_estimados"] += not found

        combos = random.sample(COMBOS, k=min(args.variantes, 4))
        for k, (use_h, use_g) in enumerate(combos):
            img, labels = synthesize(
                scene, face, (le, re_),
                helmets[rng.integers(len(helmets))] if use_h else None,
                goggles[rng.integers(len(goggles))] if use_g else None,
                rng,
            )
            name = f"synth_{f.stem}_v{k}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / f"{name}.txt").write_text("\n".join(to_yolo(labels, img.shape[1], img.shape[0])) + "\n")
            stats["geradas"] += 1
            if previews < args.preview:
                cv2.imwrite(str(out / "preview" / f"{name}.jpg"), draw(img, labels))
                previews += 1

    print(f"{stats['fotos']} fotos lidas | {stats['sem_rosto']} sem rosto detectado (puladas) | "
          f"{stats['olhos_estimados']} com olhos estimados por proporção")
    print(f"{stats['geradas']} imagens sintéticas em {out / 'images'}")
    if args.preview:
        print(f"Confira as caixas em {out / 'preview'}")


if __name__ == "__main__":
    main()
