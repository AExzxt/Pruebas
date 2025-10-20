# scripts/prepare_materials.py
# -*- coding: utf-8 -*-
"""
Prepara datasets de materiales (MINC_2500, DTD) y los vuelca a un dataset
unificado de clasificación con estructura:
  out/
    train|val|test/
      liso | tierra | grava | obstaculo

- MINC_2500: mapea solo algunas clases (liso, grava).
- DTD: usa un whitelist de texturas "planas"/regulares → liso.

NO borra lo existente; puedes correrlo varias veces.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# =========================
# Configuración general
# =========================
RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem, ext = dst.stem, dst.suffix
    k = 1
    while True:
        new_dst = dst.with_name(f"{stem}_{k}{ext}")
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return new_dst
        k += 1

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS

def scan_images(root: Path) -> List[Path]:
    imgs = []
    if not root.exists():
        return imgs
    for p in root.rglob("*"):
        if is_image(p):
            imgs.append(p)
    return imgs

def split_list(items: List[Path], ratios=(0.8, 0.1, 0.1)) -> Tuple[List[Path], List[Path], List[Path]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(items)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    train = items[:n_train]
    val   = items[n_train:n_train+n_val]
    test  = items[n_train+n_val:]
    return train, val, test

def ensure_out_dirs(out_base: Path, classes: Iterable[str] = ("liso","tierra","grava","obstaculo")):
    for split in ("train","val","test"):
        for c in classes:
            (out_base / split / c).mkdir(parents=True, exist_ok=True)

def summarize_counts(out_base: Path):
    print("\n[RESUMEN] Archivos por split/clase:")
    for split in ("train","val","test"):
        print(f"  - {split}:")
        for c in ("liso","tierra","grava","obstaculo"):
            cnt = len(scan_images(out_base / split / c))
            print(f"      {c:10s}: {cnt}")
    print("")

# =========================
# Mapas de clases por dataset
# =========================

# MINC_2500 → tomamos solo algunas clases útiles para terreno
#    grava ← stone, polishedstone
#    liso  ← tile, painted, paper, plastic, metal, glass
MINC_TO_TARGET: Dict[str, str] = {
    "stone": "grava",
    "polishedstone": "grava",
    "tile": "liso",
    "painted": "liso",
    "paper": "liso",
    "plastic": "liso",
    "metal": "liso",
    "glass": "liso",
}

# DTD → solo algunas texturas razonables como "liso"
DTD_LISO_WHITELIST = {
    "grid", "lined", "striped", "waffled", "woven", "marbled",
    "polka-dotted", "stained", "gauzy", "porous",
    "spiralled", "meshed", "knitted", "interlaced", "lacelike",
    "chequered", "sprinkled"
}

# =========================
# Lógica por dataset
# =========================

def prepare_minc(src_images: Path, out_base: Path):
    if not src_images.exists():
        print(f"[ERROR] No existe la carpeta MINC_2500: {src_images}")
        return

    ensure_out_dirs(out_base)
    random.seed(RANDOM_SEED)

    total_mapped = 0
    per_class_counts = {"liso": 0, "grava": 0}

    for src_class, target_class in MINC_TO_TARGET.items():
        src_dir = src_images / src_class
        if not src_dir.exists():
            print(f"[WARN] Clase MINC no encontrada (se omite): {src_dir}")
            continue

        imgs = [p for p in src_dir.iterdir() if is_image(p)]
        if not imgs:
            print(f"[WARN] Sin imágenes en: {src_dir}")
            continue

        random.shuffle(imgs)
        train, val, test = split_list(imgs, SPLIT_RATIOS)

        for split_name, group in (("train", train), ("val", val), ("test", test)):
            for img in group:
                dst = out_base / split_name / target_class / f"minc_{src_class}_{img.name}"
                safe_copy(img, dst)
                per_class_counts[target_class] += 1
                total_mapped += 1

    print(f"[OK] MINC: total copiado={total_mapped} -> liso={per_class_counts['liso']}, grava={per_class_counts['grava']}")

def prepare_dtd(src_images: Path, out_base: Path):
    if not src_images.exists():
        print(f"[ERROR] No existe la carpeta DTD: {src_images}")
        return

    ensure_out_dirs(out_base)
    random.seed(RANDOM_SEED)

    total_liso = 0
    found_any = False

    for cls in sorted(DTD_LISO_WHITELIST):
        src_dir = src_images / cls
        if not src_dir.exists():
            continue
        found_any = True
        imgs = [p for p in src_dir.iterdir() if is_image(p)]
        if not imgs:
            print(f"[WARN] Sin imágenes en DTD clase: {src_dir}")
            continue

        random.shuffle(imgs)
        train, val, test = split_list(imgs, SPLIT_RATIOS)

        for split_name, group in (("train", train), ("val", val), ("test", test)):
            for img in group:
                dst = out_base / split_name / "liso" / f"dtd_{cls}_{img.name}"
                safe_copy(img, dst)
                total_liso += 1

    if not found_any:
        print("[WARN] No se encontró ninguna subcarpeta de DTD del whitelist. ¿Ruta correcta?")
    print(f"[OK] DTD: liso +{total_liso}")

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Prepara MINC_2500 y/o DTD → data/{split}/{clase}")
    ap.add_argument("--which", required=True, choices=["minc", "dtd"],
                    help="Qué dataset preparar (minc o dtd)")
    ap.add_argument("--src", required=True, help="Carpeta raíz de imágenes del dataset de origen")
    ap.add_argument("--out", required=True, help="Carpeta base de salida (por ej. A:/.../cnn-terreno/data)")
    args = ap.parse_args()

    src = Path(args.src)
    out_base = Path(args.out)

    if args.which == "minc":
        prepare_minc(src_images=src, out_base=out_base)
    elif args.which == "dtd":
        prepare_dtd(src_images=src, out_base=out_base)

    summarize_counts(out_base)

if __name__ == "__main__":
    main()



