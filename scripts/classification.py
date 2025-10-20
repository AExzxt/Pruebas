# scripts/coco_to_classification.py
import os, json, random
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

# === Configuración general ===
OUT_SIZE = (128, 128)          # tamaño del parche para la CNN
POS_CLASS = "obstaculo"        # nombre de la clase positiva (bache)
NEG_CLASS = "liso"             # clase negativa (sin bache)
PADDING = 0.2                  # % de padding alrededor del bbox (20% de ancho/alto)
NEGATIVES_PER_IMAGE = 2        # parches negativos por imagen
IOU_NEG_MAX = 0.05             # iou máx. para aceptar un negativo (evitar solape con bache)
MIN_BOX = 12                   # tamaño mínimo (px) para parches válidos

def imread_robust(path: Path):
    """Lee imagen robustamente en Windows (unicode, locks, CMYK, etc.)."""
    if not path.exists():
        return None
    # 1) OpenCV + fromfile/imdecode
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    # 2) Fallback PIL
    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def load_coco(json_path: str):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    id2img = {im["id"]: im for im in data["images"]}
    id2cat = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return id2img, id2cat, anns_by_img

def clip_box(x, y, w, h, W, H):
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    inter = max(0, min(ax2,bx2) - max(ax,bx)) * max(0, min(ay2,by2) - max(ay,by))
    ua = aw*ah + bw*bh - inter
    return inter / ua if ua > 0 else 0.0

def ensure_dirs(base: str, splits=("train","val","test"), classes=("liso","grava","tierra","obstaculo")):
    for s in splits:
        for c in classes:
            Path(base, s, c).mkdir(parents=True, exist_ok=True)

def save_patch(img, box, out_path):
    x,y,w,h = [int(v) for v in box]
    if w < MIN_BOX or h < MIN_BOX: return False
    crop = img[y:y+h, x:x+w]
    if crop.size == 0: return False
    crop = cv2.resize(crop, OUT_SIZE, interpolation=cv2.INTER_AREA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return True

def random_negatives(img_w, img_h, bboxes: List[Tuple[int,int,int,int]], n: int, rng):
    negs = []
    tries = 0
    while len(negs) < n and tries < n*60:
        tries += 1
        # tamaño variable del parche negativo (cuadrado)
        side = rng.randint(40, max(41, min(img_w, img_h)//2))
        x = rng.randint(0, max(1, img_w - side))
        y = rng.randint(0, max(1, img_h - side))
        cand = (x, y, side, side)
        if all(iou(cand, b) <= IOU_NEG_MAX for b in bboxes):
            negs.append(cand)
    return negs

def resolve_image_path(in_images_dir: str, file_name_field: str) -> Path:
    """
    Roboflow a veces guarda 'file_name' con subcarpetas (train/images/xxx.jpg) o mayúsculas/minúsculas.
    Este resolvedor:
      - usa solo el nombre base,
      - busca variantes lower/upper,
      - y si no, intenta coincidencia parcial.
    """
    images_dir = Path(in_images_dir)
    base = Path(file_name_field).name  # solo nombre
    p = images_dir / base
    if p.exists():
        return p
    # prueba lower/upper
    cands = list(images_dir.glob(base.lower()))
    if not cands:
        cands = list(images_dir.glob(base.upper()))
    # coincidencia parcial (por si hay sufijos .jpeg/.jpg o guiones)
    if not cands:
        stem = Path(base).stem
        cands = list(images_dir.glob(f"*{stem}*"))
    return cands[0] if cands else p  # si no hay, devolver ruta "esperada"

def process_split(in_images_dir, in_json, out_base, split_name, rng):
    id2img, id2cat, anns_by_img = load_coco(in_json)

    # intenta detectar nombres típicos de la clase "bache"
    pothole_cat_ids = [cid for cid, name in id2cat.items()
                       if name.lower() in ("pothole","potholes","bache","baches","hoyo","hoyo_vial","hole","obstacle","obstaculo")]
    if not pothole_cat_ids:
        pothole_cat_ids = list(id2cat.keys())  # fallback

    for img_id, im in id2img.items():
        filename = im.get("file_name", "")
        img_path = resolve_image_path(in_images_dir, filename)

        img = imread_robust(img_path)
        if img is None:
            print(f"[WARN] no se pudo leer {img_path}")
            continue

        H, W = img.shape[:2]
        anns = anns_by_img.get(img_id, [])
        boxes = []

        # Positivos (obstáculo)
        for ann in anns:
            if ann.get("category_id") in pothole_cat_ids:
                x, y, w, h = ann["bbox"]
                padw, padh = w*PADDING, h*PADDING
                xb = x - padw; yb = y - padh
                wb = w + 2*padw; hb = h + 2*padh
                xb,yb,wb,hb = clip_box(xb,yb,wb,hb,W,H)
                boxes.append((xb,yb,wb,hb))
                out_p = Path(out_base) / split_name / POS_CLASS / f"{img_path.stem}_x{xb}_y{yb}_w{wb}_h{hb}.jpg"
                save_patch(img, (xb,yb,wb,hb), out_p)

        # Negativos (sin solape con ningún bbox)
        negs = random_negatives(W, H, boxes, NEGATIVES_PER_IMAGE, rng)
        for (xn,yn,wn,hn) in negs:
            out_n = Path(out_base) / split_name / NEG_CLASS / f"{img_path.stem}_negx{xn}_negy{yn}_w{wn}_h{hn}.jpg"
            save_patch(img, (xn,yn,wn,hn), out_n)

if __name__ == "__main__":
    """
    Uso:
      python scripts/coco_to_classification.py --in dataset\\baches.v5i.coco\\train\\images --ann dataset\\baches.v5i.coco\\train\\_annotations.coco.json --split train --out data
      (Repite para valid->val y test)
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Carpeta de imágenes del split")
    ap.add_argument("--ann", dest="ann_path", required=True, help="Ruta al _annotations.coco.json")
    ap.add_argument("--split", dest="split", required=True, choices=["train","val","valid","test"])
    ap.add_argument("--out", dest="out_dir", required=True, help="Carpeta base de salida (data)")
    args = ap.parse_args()

    split = "val" if args.split == "valid" else args.split
    ensure_dirs(args.out_dir,
                splits=("train","val","test"),
                classes=("liso","grava","tierra","obstaculo"))
    rng = random.Random(42)

    process_split(args.in_dir, args.ann_path, args.out_dir, split, rng)
    print("[OK] Convertido split:", split)


