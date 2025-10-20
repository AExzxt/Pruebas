# scripts/prepare_gtos_mobile.py
from pathlib import Path
from PIL import Image
import random

# === CONFIG ===
SRC = Path(r"A:\Varios\MobileNet\cnn-terreno\dataset\gtos_mobile")  # contiene train/ y test/
OUT = Path(r"A:\Varios\MobileNet\cnn-terreno\data")                 # salida final para MobileNetV3
IMG_SIZE = (128, 128)
# Usaremos ambos splits (train y test del dataset) como FUENTE, y nosotros haremos nuestro propio split 80/10/10
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Mapeo de clases origen (GTOS_mobile) -> clases destino (tus 3 clases)
CLASS_MAP = {
    # liso
    "asphalt": "liso",
    "stone_asphalt": "liso",
    "cement": "liso",
    "stone_cement": "liso",
    "painting_turf": "liso",   # marcas pintadas sobre asfalto/cemento

    # grava (texturas pétreas granulares)
    "pebble": "grava",
    "small_limestone": "grava",
    "large_limestone": "grava",
    "shale": "grava",
    "stone_brick": "grava",

    # tierra (blando/vegetación)
    "soil": "tierra",
    "sand": "tierra",
    "grass": "tierra",
    "turf": "tierra",
    "moss": "tierra",
    "dry_leaf": "tierra",
    "leaf": "tierra",
    "root": "tierra",
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

def iter_source_images():
    # recorre train/ y test/ del dataset GTOS_mobile
    for split in ("train", "test"):
        base = SRC / split
        if not base.exists():
            continue
        for cls_dir in base.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            dst_name = CLASS_MAP.get(cls_name)  # None si no mapeamos esta clase
            if dst_name is None:
                continue
            for img_path in cls_dir.rglob("*"):
                if img_path.suffix in VALID_EXTS:
                    yield img_path, dst_name

def save_resized(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB").resize(IMG_SIZE)
        im.save(dst, format="JPEG", quality=90)

def main():
    random.seed(42)
    all_items = []  # [(path, dst_class)]
    for p, dst in iter_source_images():
        all_items.append((p, dst))

    if not all_items:
        raise SystemExit("[ERROR] No se encontraron imágenes mapeables. Revisa SRC y CLASS_MAP.")

    # separa por clase para hacer split estratificado 80/10/10
    by_class = {}
    for p, dst in all_items:
        by_class.setdefault(dst, []).append(p)

    for dst, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val   = int(n * SPLIT_RATIOS["val"])
        n_test  = n - n_train - n_val
        splits = {
            "train": items[:n_train],
            "val":   items[n_train:n_train+n_val],
            "test":  items[n_train+n_val:],
        }
        # guarda redimensionado en OUT/split/dst/
        for split, paths in splits.items():
            for i, src in enumerate(paths):
                out_path = OUT / split / dst / f"{src.stem}.jpg"
                try:
                    save_resized(src, out_path)
                except Exception as e:
                    print(f"[WARN] no pude procesar {src}: {e}")
        print(f"[OK] {dst}: total={n}  -> train={n_train}, val={n_val}, test={n_test}")

    print("[DONE] Dataset listo en:", OUT.resolve())

if __name__ == "__main__":
    main()

