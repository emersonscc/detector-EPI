"""Monta o dataset final de capacete e óculos a partir de até três fontes.

Classes finais:  0 helmet, 1 no_helmet, 2 goggles, 3 no_goggles

Fontes e regras de divisão
--------------------------
1. Roboflow PPEs (--src, obrigatório)
   Filtra as 4 classes de interesse e reindexa (helmet 2->0, no_helmet 7->1,
   goggles 1->2, no_goggles 6->3). Junta train/valid/test e redivide 70/20/10
   equilibrando as 4 classes ao mesmo tempo (estratificação iterativa por grupo).
   Para evitar vazamento, a divisão é feita por GRUPO, não por imagem:
   - cópias aumentadas da mesma foto ("foto_jpg.rf.<hash>.jpg") ficam juntas;
   - frames de vídeo ("packing1085", "packing1086"...) são agrupados em blocos de
     --bloco frames consecutivos, para quase-duplicatas não caírem em splits diferentes.

2. Sintéticas (--sinteticas, opcional; saída de synth_dataset.py)
   Divididas por PESSOA (ou por sessão, se houver uma pessoa só) em 80/20 entre
   treino e validação. Nunca entram no teste: o teste tem que medir imagens reais.

3. Teste real (--teste-real, opcional; exportação YOLO do Roboflow com fotos reais anotadas)
   Vai inteiro para um split separado, test_real/. As classes são casadas pelo NOME
   lido do data.yaml da exportação, então a ordem das classes lá não importa.

Saída
-----
    <dst>/train, valid, test[, test_real]   (images/ + labels/)
    <dst>/data.yaml             teste = teste do Roboflow
    <dst>/data_teste_real.yaml  teste = fotos reais (se --teste-real)
Os .yaml não têm caminho absoluto: a pasta pode ser zipada e levada para o Colab.

Uso:
    python scripts/filter_dataset.py --src data/raw/ppes --dst data/processed/epi \\
        --sinteticas data/synthetic --teste-real data/raw/teste_real
"""
import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

# índice original (Roboflow PPEs) -> índice novo
CLASS_MAP = {2: 0, 7: 1, 1: 2, 6: 3}
NEW_NAMES = ["helmet", "no_helmet", "goggles", "no_goggles"]
SPLITS = ["train", "valid", "test", "test_real"]
RATIOS_ROBOFLOW = {"train": 0.7, "valid": 0.2, "test": 0.1}
RATIOS_SYNTH = {"train": 0.8, "valid": 0.2}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# nomes alternativos aceitos no teste real (minúsculas, sem espaços/hífens)
NAME_ALIASES = {
    "helmet": 0, "capacete": 0, "comcapacete": 0,
    "nohelmet": 1, "semcapacete": 1,
    "goggles": 2, "oculos": 2, "comoculos": 2,
    "nogoggles": 3, "semoculos": 3,
}


# ------------------------------------------------------------------ leitura

def remap_labels(path, mapping):
    """Lê um .txt YOLO e devolve as linhas das classes de interesse, reindexadas."""
    if not path.exists():
        return []
    kept = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and int(parts[0]) in mapping:
            kept.append(" ".join([str(mapping[int(parts[0])])] + parts[1:5]))
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


def image_label_pairs(folder):
    """Todas as imagens sob 'folder' que estão numa pasta images/, com o .txt correspondente em labels/."""
    for img in sorted(folder.rglob("*")):
        if img.suffix.lower() in IMG_EXTS and img.parent.name == "images":
            yield img, img.parent.parent / "labels" / f"{img.stem}.txt"


# ------------------------------------------------------------------ agrupamento

def roboflow_group(img, block):
    """Grupo de uma imagem do Roboflow: foto original, ou bloco de frames do mesmo vídeo."""
    base = re.sub(r"_(jpe?g|png|bmp|webp)$", "", img.stem.split(".rf.")[0], flags=re.I)
    m = re.match(r"^(.*?[A-Za-z_-])(\d+)$", base)
    if m:  # nome tipo "packing1085": prefixo + número de frame
        prefix, num = m.groups()
        return f"{prefix}#{int(num) // block}", prefix
    return base, None  # foto avulsa (ex.: "005466")


def synth_group(img, by):
    """Pessoa ou pessoa+sessão de uma imagem sintética (synth_<pessoa>_s<N>_<frame>_v<k>)."""
    m = re.match(r"^synth_(.+?)_s(\d+)_", img.stem)
    if not m:
        return img.stem
    return m.group(1) if by == "pessoa" else f"{m.group(1)}_s{m.group(2)}"


def split_groups(groups, ratios, rng):
    """Embaralha as chaves de grupo e reparte de acordo com as proporções (em nº de grupos)."""
    keys = sorted(groups)
    rng.shuffle(keys)
    names, assignment, start = list(ratios), {}, 0
    for i, name in enumerate(names):
        n = len(keys) - start if i == len(names) - 1 else round(len(keys) * ratios[name])
        for k in keys[start:start + n]:
            assignment[k] = name
        start += n
    return assignment


def class_vector(items):
    """Quantidade de objetos de cada uma das 4 classes num conjunto de (imagem, linhas)."""
    v = np.zeros(len(NEW_NAMES))
    for _, lines in items:
        for l in lines:
            v[int(l.split()[0])] += 1
    return v


def stratified_split(groups, ratios, rng):
    """Estratificação iterativa por grupo: equilibra as 4 classes ao mesmo tempo.

    Os grupos (inteiros) são distribuídos do maior para o menor. Cada um vai para o
    split mais "carente" das classes que ele contém: para cada split, soma-se, classe a
    classe, quanto falta para a meta (proporção x total da classe), ponderado pela
    quantidade daquela classe no grupo. Classes raras pesam mais (meta menor).
    """
    keys = sorted(groups)
    rng.shuffle(keys)
    vecs = {k: class_vector(groups[k]) for k in keys}
    total = sum(vecs.values())
    names = list(ratios)
    target = {n: ratios[n] * total for n in names}
    current = {n: np.zeros(len(NEW_NAMES)) for n in names}
    assignment = {}
    for k in sorted(keys, key=lambda k: vecs[k].sum(), reverse=True):  # sort estável
        v = vecs[k]

        def need(n):
            return float(np.sum(v * (target[n] - current[n]) / np.maximum(target[n], 1)))

        dest = max(names, key=lambda n: (need(n), ratios[n]))
        assignment[k] = dest
        current[dest] += v
    return assignment


# ------------------------------------------------------------------ fontes

def load_roboflow(src, block, keep_background, seed):
    items, discarded = [], 0
    for img, lbl in image_label_pairs(src):
        lines = remap_labels(lbl, CLASS_MAP)
        if not lines and not keep_background:
            discarded += 1
            continue
        items.append((img, lines))

    groups, videos = defaultdict(list), Counter()
    for img, lines in items:
        g, prefix = roboflow_group(img, block)
        groups[g].append((img, lines))
        if prefix:
            videos[prefix] += 1

    assignment = stratified_split(groups, RATIOS_ROBOFLOW, random.Random(seed))
    out = [(assignment[g], img, lines) for g, its in groups.items() for img, lines in its]

    print(f"[Roboflow] {len(items)} imagens com capacete/óculos, {discarded} descartadas, "
          f"{len(groups)} grupos")
    if videos:
        resumo = ", ".join(f"{p} ({n} imgs)" for p, n in videos.most_common(8))
        print(f"           sequências de vídeo agrupadas em blocos de {block} frames: {resumo}")
    per_split = {sp: class_vector([it for g, its in groups.items() if assignment[g] == sp for it in its])
                 for sp in RATIOS_ROBOFLOW}
    total = sum(per_split.values())
    for i, name in enumerate(NEW_NAMES):
        print(f"           {name:>10}: " + " / ".join(
            f"{sp} {per_split[sp][i] / max(total[i], 1):.0%}" for sp in RATIOS_ROBOFLOW))
    empty = [(NEW_NAMES[i], sp) for sp in RATIOS_ROBOFLOW for i in range(4) if total[i] and per_split[sp][i] == 0]
    if empty:
        print(f"           [aviso] classe ausente em algum split {empty}: diminua --bloco para ter mais grupos")
    return out


def load_synthetic(folder, seed):
    pairs = list(image_label_pairs(folder))
    if not pairs:
        raise SystemExit(f"Nenhuma imagem em {folder}/images")
    people = {synth_group(img, "pessoa") for img, _ in pairs}
    by = "pessoa" if len(people) >= 3 else "sessao"
    groups = defaultdict(list)
    for img, lbl in pairs:
        groups[synth_group(img, by)].append((img, lbl))

    if len(groups) < 2:
        print(f"[Sintéticas] [aviso] só 1 {by} encontrada: tudo vai para o treino "
              "(grave mais uma sessão para ter validação sem vazamento)")
        assignment = {g: "train" for g in groups}
    else:
        assignment = split_groups(groups, RATIOS_SYNTH, random.Random(seed))

    identity = {i: i for i in range(len(NEW_NAMES))}
    out = [(assignment[g], img, remap_labels(lbl, identity)) for g, its in groups.items() for img, lbl in its]
    detalhe = ", ".join(f"{g}->{assignment[g]}" for g in sorted(groups))
    print(f"[Sintéticas] {len(out)} imagens, divididas por {by}: {detalhe}")
    return out


def load_real_test(folder):
    yamls = sorted(folder.rglob("data.yaml"))
    if not yamls:
        raise SystemExit(f"Não achei o data.yaml da exportação em {folder}")
    names = yaml.safe_load(yamls[0].read_text()).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    mapping, unknown = {}, []
    for i, n in enumerate(names):
        key = re.sub(r"[\s_\-]", "", str(n).lower()).replace("ó", "o")
        if key in NAME_ALIASES:
            mapping[i] = NAME_ALIASES[key]
        else:
            unknown.append(n)
    if unknown:
        print(f"[Teste real] [aviso] classes ignoradas (nome não reconhecido): {unknown}")
    out = [("test_real", img, remap_labels(lbl, mapping)) for img, lbl in image_label_pairs(folder)]
    mapa = ", ".join(f"{names[i]}->{NEW_NAMES[j]}" for i, j in mapping.items())
    print(f"[Teste real] {len(out)} imagens | classes: {mapa}")
    return out


# ------------------------------------------------------------------ escrita

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


def write_yaml(path, test_split):
    # sem 'path': o Ultralytics usa a pasta do próprio .yaml (portável para o Colab)
    cfg = {"train": "train/images", "val": "valid/images", "test": f"{test_split}/images",
           "names": dict(enumerate(NEW_NAMES))}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="pasta extraída do Roboflow PPEs")
    ap.add_argument("--dst", required=True, help="pasta de saída")
    ap.add_argument("--sinteticas", help="pasta gerada por synth_dataset.py (com images/ e labels/)")
    ap.add_argument("--teste-real", help="exportação YOLO do Roboflow com as fotos reais anotadas")
    ap.add_argument("--bloco", type=int, default=100, help="frames consecutivos por grupo nos vídeos")
    ap.add_argument("--keep-background", action="store_true",
                    help="mantém imagens do Roboflow sem nenhuma das 4 classes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    items = load_roboflow(find_dataset_root(Path(args.src)), args.bloco, args.keep_background, args.seed)
    if args.sinteticas:
        items += load_synthetic(Path(args.sinteticas), args.seed)
    if args.teste_real:
        items += load_real_test(Path(args.teste_real))

    names = [p.name for _, p, _ in items]
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        raise SystemExit(f"Nomes de arquivo repetidos entre as fontes (ex.: {dup[:3]}). Renomeie antes de juntar.")

    stats = write(items, dst)
    print()
    for split in SPLITS:
        if split in stats:
            resumo = ", ".join(f"{n}={stats[split][n]}" for n in NEW_NAMES)
            print(f"{split:>9}: {stats[split]['imagens']:>6} imagens | {resumo}")

    write_yaml(dst / "data.yaml", "test")
    print(f"\ndata.yaml gerado em {dst / 'data.yaml'}")
    if "test_real" in stats:
        write_yaml(dst / "data_teste_real.yaml", "test_real")
        print(f"data_teste_real.yaml gerado em {dst / 'data_teste_real.yaml'}")


if __name__ == "__main__":
    main()
