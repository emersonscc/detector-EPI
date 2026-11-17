"""Leitura e gravação de imagens compatível com caminhos acentuados no Windows.

cv2.imread/cv2.imwrite falham em caminhos com acentos no Windows
(ex.: "C:/Users/João/Imagens/obra.jpg"). Ler os bytes com NumPy e decodificar
com cv2.imdecode resolve.
"""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm", ".mpg", ".mpeg"}


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def read_image(path):
    """Lê qualquer imagem como BGR de 3 canais (converte cinza e PNG com transparência)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return img


def write_image(path, img, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"Não foi possível salvar a imagem: {path}")
    buf.tofile(str(path))


def list_images(folder):
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and is_image(p)),
                  key=lambda p: p.name.lower())
