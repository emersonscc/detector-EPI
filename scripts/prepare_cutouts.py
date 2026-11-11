"""Recorta capacetes e óculos de fotos JPG, gerando PNGs com transparência (RGBA).

As fotos de catálogo costumam vir com fundo branco ou com o xadrez de "transparência
falsa" gravado no próprio JPG. O fundo é removido com técnicas clássicas de PDI:

1. Segmentação por cor no espaço HSV: fundo = pixels claros e pouco saturados.
2. Preenchimento a partir das bordas (flood fill por componentes conexos): só conta como
   fundo o que está ligado à borda da imagem, preservando reflexos brancos do objeto.
3. Regiões "de fundo" cercadas pelo objeto (ex.: lentes transparentes dos óculos) viram
   semitransparentes, com o padrão xadrez apagado por filtro de mediana.
4. Remoção de componentes pequenos/soltos e suavização da borda (erosão + desfoque).

Uso:
    python scripts/prepare_cutouts.py --src fotos_epi --dst assets/epi
Os arquivos devem se chamar "capacete*.jpg" ou "oculos*.jpg".
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

LENS_ALPHA = 0.25  # opacidade das lentes transparentes


def connected_to_border(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, border[border > 0])


def main_object(mask):
    """Mantém o maior componente e os que ficam dentro da faixa vertical dele
    (descarta itens soltos abaixo/acima, como protetores auriculares)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y0, h0 = stats[big, cv2.CC_STAT_TOP], stats[big, cv2.CC_STAT_HEIGHT]
    keep = np.zeros_like(mask)
    for i in range(1, n):
        y, h, a = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if i == big or (a > 0.002 * mask.size and y0 <= y + h / 2 <= y0 + h0):
            keep |= lab == i
    return keep


def local_std(gray, win):
    mean = cv2.boxFilter(gray, -1, (win, win))
    sq = cv2.boxFilter(gray * gray, -1, (win, win))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def checker_period(gray, mid):
    """Tamanho aproximado dos quadrados do xadrez, medido nas bordas da imagem."""
    diffs = []
    for line in (gray[2], gray[-3], gray[:, 2], gray[:, -3]):
        t = np.where(np.diff((line > mid).astype(np.int8)) != 0)[0]
        if len(t) > 2:
            diffs.extend(np.diff(t))
    return float(np.median(diffs)) if diffs else 20.0


def cutout(img, kind):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    edge = np.concatenate([gray[:6].ravel(), gray[-6:].ravel(), gray[:, :6].ravel(), gray[:, -6:].ravel()])
    lo, hi = np.percentile(edge, 10), np.percentile(edge, 90)
    neutral_light = (sat < 25) & (gray > 170)

    mid = (lo + hi) / 2
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)  # ruído do sensor não pode parecer xadrez
    n_transitions = sum(len(np.where(np.diff((line > mid).astype(np.int8)) != 0)[0])
                        for line in (smooth[2], smooth[-3], smooth[:, 2], smooth[:, -3]))
    p = checker_period(smooth, mid)
    std = np.zeros_like(gray)

    if hi - lo < 8 or n_transitions < 12 or p < 5:
        # fundo liso (de qualquer cor, inclusive foto com gradiente de luz):
        # flood fill a partir de sementes espalhadas pelas bordas
        bg_type = "liso"
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
        h, w = gray.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        step = max(1, min(h, w) // 40)
        seeds = [(x, y) for x in range(0, w, step) for y in (0, h - 1)] + [(x, y) for y in range(0, h, step) for x in (0, w - 1)]
        for sx, sy in seeds:
            if mask[sy + 1, sx + 1] == 0:
                # faixa fixa em torno da cor da semente: tolera variação de brilho (L)
                # e pouca variação de cor (a, b); não "escorre" por sombras suaves
                cv2.floodFill(lab, mask, (sx, sy), 0, (22, 7, 7), (22, 7, 7),
                              4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        background = mask[1:-1, 1:-1] > 0
        # regiões cercadas pelo objeto com a mesma cor do fundo (lente, vão entre alças)
        bg_color = np.median(lab[background], axis=0)
        if kind == "oculos":
            # lente transparente "puxa" a cor do fundo, mas com outro brilho: compara só a cor (a, b)
            same_color = np.linalg.norm(lab[..., 1:].astype(np.float32) - bg_color[1:], axis=2) < 15
        else:
            same_color = np.linalg.norm(lab.astype(np.float32) - bg_color, axis=2) < 12
        n, labels, stats, _ = cv2.connectedComponentsWithStats((same_color & ~background).astype(np.uint8))
        seen_through = np.zeros_like(background)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 0.002 * h * w:
                seen_through |= labels == i
    else:
        # fundo xadrez: o que define o fundo é a TEXTURA (desvio padrão local alto),
        # já que reflexos e lentes também são claros e sem cor, mas são lisos.
        # Em uma janela maior que um quadrado, o xadrez tem desvio ~ (hi - lo) / 2.
        bg_type = f"xadrez (quadrados de ~{p:.0f}px)"
        win = int(1.5 * p) | 1
        std = local_std(gray, win)
        texture = neutral_light & (std > 0.35 * (hi - lo))
        background = connected_to_border(texture)
        # regiões com xadrez cercadas pelo objeto (vão entre alças, lente com xadrez atrás)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((texture & ~background).astype(np.uint8))
        seen_through = np.zeros_like(background)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 4 * p * p:
                seen_through |= lab == i
        # devolve ao objeto os pixels de borda que não têm os tons do xadrez
        dark, light = np.median(edge[edge < mid]), np.median(edge[edge >= mid])
        is_tone = (sat < 25) & ((np.abs(gray - dark) < 10) | (np.abs(gray - light) < 10))
        background &= is_tone | cv2.erode(background.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    obj = ~background & ~seen_through
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    obj = main_object(obj)

    alpha = obj.astype(np.float32)
    rgb = img.astype(np.float32)
    if kind == "oculos":
        # lente/plástico transparente: claro, sem cor e liso -> semitransparente
        clear = obj & neutral_light & (gray > 200) & (std < 6)
        alpha[clear] = LENS_ALPHA
        # lente com o fundo visível por trás: vira lente semitransparente de tom neutro
        alpha[seen_through] = LENS_ALPHA
        rgb[seen_through] = 235
    # (capacetes: o que se vê através, entre as alças, fica totalmente transparente)

    # borda suave: erode 1px e desfoca para evitar o "halo" claro do recorte
    solid = cv2.erode((alpha > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(np.where(solid > 0, alpha, 0).astype(np.float32), (3, 3), 0)

    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    ys, xs = np.where(alpha > 0.05)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], bg_type


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="assets/epi")
    args = ap.parse_args()

    dst = Path(args.dst)
    for kind in ("capacete", "oculos"):
        (dst / kind).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.src).glob("*.jpg")):
        kind = "capacete" if f.stem.lower().startswith("capacete") else "oculos" if f.stem.lower().startswith("oculos") else None
        if kind is None:
            print(f"[pulando] {f.name}: nome deve começar com 'capacete' ou 'oculos'")
            continue
        img = cv2.imread(str(f))
        out, bg_type = cutout(img, kind)
        name = f.stem.replace(" ", "_") + ".png"
        cv2.imwrite(str(dst / kind / name), out)
        print(f"{f.name:>16} -> {kind}/{name}  ({out.shape[1]}x{out.shape[0]}, fundo {bg_type})")


if __name__ == "__main__":
    main()
