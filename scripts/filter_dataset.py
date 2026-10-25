"""Filtra o dataset PPEs (Roboflow) mantendo só as classes de capacete e óculos.

Classes originais (data.yaml do Roboflow):
    0 glove, 1 goggles, 2 helmet, 3 mask, 4 no-suit, 5 no_glove,
    6 no_goggles, 7 no_helmet, 8 no_mask, 9 no_shoes, 10 shoes, 11 suit

Classes finais (reindexadas):
    0 helmet, 1 no_helmet, 2 goggles, 3 no_goggles

Por padrão, junta as imagens de todos os splits e redivide em treino/validação/teste
(70/20/10), separando imagens de capacete e de óculos em proporções iguais em cada split.
Isso é necessário porque no dataset original o treino não tem nenhum capacete.
Cópias aumentadas da mesma foto (nomes "foto_jpg.rf.<hash>.jpg") ficam sempre no
mesmo split, para não vazar informação do treino para o teste.

Uso:
    python scripts/filter_dataset.py --src data/raw/ppes --dst data/processed/epi
    python scripts/filter_dataset.py --src data/raw/ppes --dst data/processed/epi --keep-splits
"""
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# índice original -> índice novo
CLASS_MAP = {2: 0, 7: 1, 1: 2, 6: 3}
NEW_NAMES = ["helmet", "no_helmet", "goggles", "no_goggles"]
SRC_SPLITS = ["train", "valid", "test"]
YAML_KEYS = {"train": "train", "valid": "val", "test": "test"}
RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
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


def find_dataset_root(src):
    """Acha a pasta que contém train/images, mesmo se o zip foi extraído com subpastas extras."""
    if not src.is_dir():
        raise SystemExit(f"Pasta não encontrada: {src.resolve()}")
    if (src / "train" / "images").is_dir():
        return src
    candidates = sorted(p.parent.parent for p in src.rglob("images") if p.parent.name == "train")
    if candidates:
        return candidates[0]
    conteudo = ", ".join(p.name for p in src.iterdir()) or "(vazia)"
    raise SystemExit(
        f"Não achei a pasta 'train/images' dentro de {src.resolve()}.\n"
        f"Conteúdo atual: {conteudo}\n"
        "Extraia o zip do Roboflow de forma que fique: <pasta>/train/images, <pasta>/valid/images ..."
    )


def collect(src, keep_background):
    """Lê todas as imagens e devolve lista de (split_original, imagem, linhas_filtradas)."""
    items, descartadas = [], 0
    for split in SRC_SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.is_dir():
            print(f"[aviso] split '{split}' não encontrado, pulando")
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            lines = filter_label_file(lbl) if lbl.exists() else []
            if not lines and not keep_background:
                descartadas += 1
                continue
            items.append((split, img, lines))
    return items, descartadas


def source_key(img):
    """Nome da foto original, ignorando o sufixo que o Roboflow adiciona às cópias aumentadas."""
    return img.stem.split(".rf.")[0]


def resplit(items, seed):
    """Agrupa por foto original e divide 70/20/10, separadamente para capacete e óculos."""
    groups = defaultdict(list)
    for it in items:
        groups[source_key(it[1])].append(it)

    by_category = defaultdict(list)
    for key, its in groups.items():
        classes = {int(l.split()[0]) for _, _, lines in its for l in lines}
        by_category["capacete" if classes & {0, 1} else "oculos"].append(key)

    rng = random.Random(seed)
    assignment = {}
    for keys in by_category.values():
        keys.sort()
        rng.shuffle(keys)
        n_train = round(len(keys) * RATIOS["train"])
        n_val = round(len(keys) * RATIOS["valid"])
        for i, k in enumerate(keys):
            assignment[k] = "train" if i < n_train else "valid" if i < n_train + n_val else "test"

    return [(assignment[source_key(img)], img, lines) for _, img, lines in items]


def write(items, dst):
    stats = defaultdict(Counter)
    for split, img, lines in items:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / split / "images" / img.name)
        (dst / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[split]["imagens"] += 1
        for l in lines:
            stats[split][NEW_NAMES[int(l.split()[0])]] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="pasta extraída do Roboflow (com train/ valid/ test/)")
    ap.add_argument("--dst", required=True, help="pasta de saída")
    ap.add_argument("--keep-background", action="store_true",
                    help="mantém imagens sem nenhuma das 4 classes (como exemplos negativos)")
    ap.add_argument("--keep-splits", action="store_true",
                    help="mantém a divisão original do Roboflow em vez de redividir")
    ap.add_argument("--seed", type=int, default=42, help="semente da divisão aleatória")
    args = ap.parse_args()

    src, dst = find_dataset_root(Path(args.src)), Path(args.dst)
    print(f"Dataset encontrado em: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    items, descartadas = collect(src, args.keep_background)
    if not items:
        raise SystemExit("Nenhuma imagem com as classes de interesse foi encontrada.")
    print(f"{len(items)} imagens com capacete/óculos, {descartadas} descartadas (sem essas classes)")

    if not args.keep_splits:
        items = resplit(items, args.seed)
        print("Divisão refeita: 70% treino, 20% validação, 10% teste\n")

    stats = write(items, dst)
    yaml_cfg = {"path": str(dst.resolve()), "names": dict(enumerate(NEW_NAMES))}
    for split in SRC_SPLITS:
        if split in stats:
            yaml_cfg[YAML_KEYS[split]] = f"{split}/images"
            resumo = ", ".join(f"{n}={stats[split][n]}" for n in NEW_NAMES)
            print(f"{split:>5}: {stats[split]['imagens']} imagens | {resumo}")

    with open(dst / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"\ndata.yaml gerado em {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
