"""Pré-processamento de imagem aplicado antes da detecção.

Técnicas clássicas de PDI, todas implementadas com OpenCV, para melhorar a imagem
em condições ruins de captura (pouca luz, contraluz, sensor ruidoso, foco suave):

- Balanço de branco "gray world": corrige a dominante de cor (luz amarela, azulada...)
  escalando cada canal para que a média dos três fique igual.
- Correção de gama: curva não linear aplicada por tabela (LUT); gama < 1 clareia
  as sombras, gama > 1 escurece as altas luzes.
- CLAHE (equalização adaptativa de histograma com limite de contraste): equaliza o
  canal L do espaço Lab em blocos (tiles), realçando contraste local sem alterar as cores
  e sem amplificar tanto o ruído quanto a equalização global.
- Redução de ruído: filtro bilateral, que suaviza regiões homogêneas preservando bordas
  (pondera os vizinhos pela distância espacial e pela diferença de intensidade).
- Nitidez: máscara de nitidez (unsharp masking) — soma à imagem a diferença entre ela e
  sua versão desfocada por um filtro gaussiano, realçando bordas.

A ordem de aplicação segue a ordem acima: primeiro corrige cor e brilho, depois contraste,
depois remove ruído e por último realça bordas (para não realçar o ruído).
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessSettings:
    white_balance: bool = False
    gamma_on: bool = False
    gamma: float = 0.7            # < 1 clareia, > 1 escurece
    clahe_on: bool = False
    clahe_clip: float = 2.0       # limite de contraste (1 = quase nada, 4 = forte)
    clahe_tiles: int = 8          # grade de blocos (8x8)
    denoise_on: bool = False
    denoise_strength: int = 50    # sigma de cor/espaço do filtro bilateral
    sharpen_on: bool = False
    sharpen_amount: float = 1.0   # peso do realce (0,5 sutil – 2 forte)

    @property
    def active(self):
        return any((self.white_balance, self.gamma_on, self.clahe_on, self.denoise_on, self.sharpen_on))

    def describe(self):
        """Lista curta das técnicas ligadas (para a barra de status e relatórios)."""
        parts = []
        if self.white_balance:
            parts.append("balanço de branco")
        if self.gamma_on:
            parts.append(f"gama {self.gamma:.2f}")
        if self.clahe_on:
            parts.append(f"CLAHE {self.clahe_clip:.1f}")
        if self.denoise_on:
            parts.append(f"bilateral {self.denoise_strength}")
        if self.sharpen_on:
            parts.append(f"nitidez {self.sharpen_amount:.1f}")
        return ", ".join(parts) or "nenhum"


PRESETS = {
    "Nenhum": PreprocessSettings(),
    "Ambiente escuro": PreprocessSettings(gamma_on=True, gamma=0.6, clahe_on=True, clahe_clip=2.5,
                                          denoise_on=True, denoise_strength=40),
    "Contraluz": PreprocessSettings(clahe_on=True, clahe_clip=3.5, gamma_on=True, gamma=0.8),
    "Câmera ruidosa": PreprocessSettings(denoise_on=True, denoise_strength=75, sharpen_on=True,
                                         sharpen_amount=0.6),
    "Luz artificial (cor)": PreprocessSettings(white_balance=True, clahe_on=True, clahe_clip=1.5),
}


# ---------------------------------------------------------------- técnicas

def gray_world(img):
    """Balanço de branco: cada canal é escalado para que sua média seja a média geral.
    O ganho de cada canal vira uma tabela de consulta (LUT), aplicada sem converter a imagem."""
    means = np.array(cv2.mean(img)[:3])
    gain = means.mean() / np.maximum(means, 1e-6)
    lut = np.clip(np.arange(256)[:, None] * gain[None, :], 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut.reshape(256, 1, 3))


_gamma_cache = {}


def gamma_correction(img, gamma):
    """Saída = 255 * (entrada/255)^gama, aplicada por tabela de consulta (LUT): 256 contas, não milhões."""
    key = round(gamma, 3)
    lut = _gamma_cache.get(key)
    if lut is None:
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        _gamma_cache[key] = lut
    return cv2.LUT(img, lut)


def clahe(img, clip, tiles):
    """CLAHE só na luminância (canal L do Lab): realça contraste sem distorcer as cores."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def bilateral(img, strength):
    """Filtro bilateral. O diâmetro cresce com a resolução para ter efeito parecido em
    640x480 e em fotos grandes; em imagens enormes, filtra uma versão reduzida (desempenho)."""
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 1600:  # filtro bilateral é caro: reduz, filtra e volta ao tamanho original
        scale = 1600 / max(h, w)
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = cv2.bilateralFilter(small, 7, strength, strength)
        # devolve só a "correção" (diferença) em resolução cheia, preservando os detalhes finos
        diff = out.astype(np.int16) - small.astype(np.int16)
        diff = cv2.resize(diff, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(img.astype(np.int16) + diff, 0, 255).astype(np.uint8)
    d = 5 if max(h, w) <= 800 else 7
    return cv2.bilateralFilter(img, d, strength, strength)


def unsharp(img, amount, sigma=None):
    """Máscara de nitidez: img + amount * (img - gauss(img))."""
    if sigma is None:
        sigma = max(1.0, max(img.shape[:2]) / 800)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


# ---------------------------------------------------------------- aplicação

class Preprocessor:
    """Função frame -> frame configurável.

    As configurações ficam num objeto imutável trocado de uma vez (self.settings = ...),
    então a interface pode alterá-las enquanto a thread do vídeo processa quadros sem
    que um quadro seja processado com metade das opções antigas e metade das novas.
    """

    def __init__(self, settings=None):
        self.settings = settings or PreprocessSettings()

    def update(self, **changes):
        self.settings = replace(self.settings, **changes)

    def __call__(self, img):
        s = self.settings  # cópia local: consistente durante todo o quadro
        if not s.active:
            return img
        out = img
        if s.white_balance:
            out = gray_world(out)
        if s.gamma_on and abs(s.gamma - 1.0) > 1e-3:
            out = gamma_correction(out, s.gamma)
        if s.clahe_on:
            out = clahe(out, s.clahe_clip, s.clahe_tiles)
        if s.denoise_on:
            out = bilateral(out, s.denoise_strength)
        if s.sharpen_on:
            out = unsharp(out, s.sharpen_amount)
        return out


# ---------------------------------------------------------------- visualizações

def luminance_histogram(before, after, size=(300, 110)):
    """Imagem BGR com os histogramas de luminância antes (cinza) e depois (amarelo)."""
    w, h = size
    canvas = np.full((h, w, 3), (39, 32, 28), np.uint8)  # mesmo tom do painel da interface
    for x in (w // 4, w // 2, 3 * w // 4):
        cv2.line(canvas, (x, 0), (x, h), (58, 48, 42), 1)

    def curve(img):
        gray = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 3), 0).ravel()
        return hist / max(hist.max(), 1)

    hb, ha = curve(before), curve(after)
    peak = max(hb.max(), ha.max(), 1e-6)
    xs = np.linspace(0, w - 1, len(hb)).astype(np.int32)

    def poly(hist):
        ys = (h - 4 - (hist / peak) * (h - 12)).astype(np.int32)
        return np.stack([xs, ys], 1)

    pb, pa = poly(hb), poly(ha)
    fill = np.vstack([[[0, h - 1]], pb, [[w - 1, h - 1]]])
    cv2.fillPoly(canvas, [fill], (70, 64, 60))
    cv2.polylines(canvas, [pb], False, (150, 140, 130), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [pa], False, (38, 165, 224), 2, cv2.LINE_AA)  # amarelo (BGR)
    return canvas


def side_by_side(original, processed, labels=("Original", "Processada")):
    """Original e processada lado a lado, com rótulo em cada metade.
    (Rótulos sem acento: o putText do OpenCV só desenha ASCII.)"""
    h = processed.shape[0]
    if original.shape[0] != h:
        original = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    gap = np.full((h, max(4, h // 120), 3), 20, np.uint8)
    out = np.hstack([original, gap, processed])
    s = max(0.5, h / 900)
    x2 = original.shape[1] + gap.shape[1]
    for x, text in ((0, labels[0]), (x2, labels[1])):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s)))
        pad = int(8 * s)
        cv2.rectangle(out, (x, h - th - 3 * pad), (x + tw + 2 * pad, h), (20, 20, 20), -1)
        cv2.putText(out, text, (x + pad, h - pad - int(2 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return out
