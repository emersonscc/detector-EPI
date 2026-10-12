"""Filtra o dataset PPEs (Roboflow) mantendo só as classes de capacete e óculos.

Classes originais (data.yaml do Roboflow):
    0 glove, 1 goggles, 2 helmet, 3 mask, 4 no-suit, 5 no_glove,
    6 no_goggles, 7 no_helmet, 8 no_mask, 9 no_shoes, 10 shoes, 11 suit

Classes finais (reindexadas):
    0 helmet, 1 no_helmet, 2 goggles, 3 no_goggles

Uso:
    python scripts/filter_dataset.py --src data/raw/ppes --dst data/processed/epi
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

# índice original -> índice novo
CLASS_MAP = {2: 0, 7: 1, 1: 2, 6: 3}
NEW_NAMES = ["helmet", "no_helmet", "goggles", "no_goggles"]
SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def filter_label_file(path):
    """Lê um .txt YOLO e devolve só as linhas das classes de interesse, reindexadas."""
    kept = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in CLASS_MAP:
            kept.append(" ".join([str(CLASS_MAP[cls])] + parts[1:]))
    return kept


def process_split(src_split, dst_split, keep_background):
    img_dir, lbl_dir = src_split / "images", src_split / "labels"
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        lines = filter_label_file(lbl) if lbl.exists() else []

        if not lines and not keep_background:
            stats["descartadas"] += 1
            continue

        shutil.copy2(img, dst_split / "images" / img.name)
        (dst_split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["imagens"] += 1
        for l in lines:
            stats[NEW_NAMES[int(l.split()[0])]] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="pasta extraída do Roboflow (com train/ valid/ test/)")
    ap.add_argument("--dst", required=True, help="pasta de saída")
    ap.add_argument("--keep-background", action="store_true",
                    help="mantém imagens sem nenhuma das 4 classes (como exemplos negativos)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)

    yaml_cfg = {"path": str(dst.resolve()), "names": dict(enumerate(NEW_NAMES))}
    for folder, key in SPLITS.items():
        if not (src / folder / "images").is_dir():
            print(f"[aviso] split '{folder}' não encontrado, pulando")
            continue
        stats = process_split(src / folder, dst / folder, args.keep_background)
        yaml_cfg[key] = f"{folder}/images"
        resumo = ", ".join(f"{n}={stats[n]}" for n in NEW_NAMES)
        print(f"{folder:>5}: {stats['imagens']} imagens mantidas, {stats['descartadas']} descartadas | {resumo}")

    with open(dst / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"\ndata.yaml gerado em {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
