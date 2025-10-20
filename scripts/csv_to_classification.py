# scripts/csv_to_classification.py
import os, csv, random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2

# === Configuración general ===
OUT_SIZE = (128, 128)          # tamaño del parche para la CNN
POS_CLASS = "obstaculo"        # clase positiva (bache)
NEG_CLASS = "liso"             # clase negativa (sin bache)
PADDING = 0.2                  # % de padding alrededor del bbox
NEGATIVES_PER_IMAGE = 2        # parches negativos por imagen
IOU_NEG_MAX = 0.05             # iou máx para aceptar un negativo
MIN_BOX = 12                   # tamaño mínimo (px) para parches válidos
POSITIVE_NAMES = {"pothole","bache","baches","hole","obstaculo","obstacle"}

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    inter = max(0, min(ax2,bx2) - max(ax,bx)) * max(0, min(ay2,by2) - max(ay,by))
    ua = aw*ah + bw*bh - inter
    return inter / ua if ua > 0 else 0.0

def clip_box(x, y, w, h, W, H):
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def ensure_dirs(base: str, split: str):
    Path(base, split, POS_CLASS).mkdir(parents=True, exist_ok=True)
    Path(base, split, NEG_CLASS).mkdir(parents=True, exist_ok=True)

def save_patch(img, box, out_path):
    x,y,w,h = [int(v) for v in box]
    if w < MIN_BOX or h < MIN_BOX: return False
    crop = img[y:y+h, x:x+w]
    if crop.size == 0: return False
    crop = cv2.resize(crop, OUT_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, crop)
    return True

def random_negatives(img_w, img_h, bboxes: List[Tuple[int,int,int,int]], n: int, rng):
    negs = []
    tries = 0
    while len(negs) < n and tries < n*50:
        tries += 1
        side = rng.randint(40, max(41, min(img_w, img_h)//2))
        x = rng.randint(0, max(1, img_w - side))
        y = rng.randint(0, max(1, img_h - side))
        cand = (x, y, side, side)
        if all(iou(cand, b) <= IOU_NEG_MAX for b in bboxes):
            negs.append(cand)
    return negs

def read_annotations_csv(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Devuelve: dict filename -> lista de bboxes (cada bbox: dict con {cls, xmin,ymin,xmax,ymax,width,height})
    Soporta encabezados típicos de Roboflow:
      file_name or filename or image, width, height, class, xmin, ymin, xmax, ymax
      y también 'image_path' si viene ruta absoluta/relativa
    """
    per_image = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # normaliza nombres de columnas (lower)
        field_map = {k.lower(): k for k in reader.fieldnames}
        def get(row, key, default=None):
            # busca por lower
            for cand in (key, key.replace("_"," "), key.replace("_","-")):
                if cand in field_map:
                    return row[field_map[cand]]
            # alternativas comunes
            alt = {
                "file_name": ["filename","file","image","image_path","path"],
                "class": ["label","category","name"],
                "xmin": ["x_min","x1"],
                "ymin": ["y_min","y1"],
                "xmax": ["x_max","x2"],
                "ymax": ["y_max","y2"]
            }
            if key in alt:
                for a in alt[key]:
                    if a in field_map:
                        return row[field_map[a]]
            return default

        for row in reader:
            fn = get(row, "file_name")
            if not fn:
                # si solo hay "image_path", úsalo
                fn = get(row, "image_path")
            if not fn:
                # último recurso
                fn = get(row, "filename")
            cls = str(get(row, "class", "")).strip().lower()

            try:
                xmin = float(get(row, "xmin"))
                ymin = float(get(row, "ymin"))
                xmax = float(get(row, "xmax"))
                ymax = float(get(row, "ymax"))
                width = float(get(row, "width")) if get(row, "width") else None
                height = float(get(row, "height")) if get(row, "height") else None
            except Exception:
                # fila malformada
                continue

            if None in (xmin, ymin, xmax, ymax): 
                continue
            ann = {"cls": cls, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                   "width": width, "height": height, "filename": fn}
            per_image.setdefault(fn, []).append(ann)
    return per_image

def resolve_image_path(images_dir: str, filename: str) -> str:
    """
    Intenta resolver rutas relativas/absolutas y casos donde el CSV trae 'images/...' o 'test/images/...'
    probando varias combinaciones sensatas.
    """
    p_images = Path(images_dir)
    p_file = Path(filename)

    # 1) Si el CSV trae ruta absoluta/relativa válida desde cwd
    if p_file.exists():
        return str(p_file)

    # 2) Si el CSV trae 'images/xxx.jpg' o 'test/images/xxx.jpg' y --in ya es .../test
    p2 = p_images / p_file
    if p2.exists():
        return str(p2)

    # 3) Solo basename dentro de --in (útil si CSV trae subcarpetas)
    p3 = p_images / p_file.name
    if p3.exists():
        return str(p3)

    # 4) Si --in es .../test/images pero CSV trae 'images/xxx.jpg', subimos un nivel
    p4 = p_images.parent / p_file
    if p4.exists():
        return str(p4)

    # 5) Si CSV trae 'test/images/xxx.jpg' y --in es .../test/images, probamos abuelo
    p5 = p_images.parent.parent / p_file
    if p5.exists():
        return str(p5)

    # 6) Último intento: normaliza separadores y usa solo basename en abuelo/parent
    p6 = p_images.parent / p_file.name
    if p6.exists():
        return str(p6)

    # 7) Fallback a combinación directa (servirá para el print del warning)
    return str(p2)


def process_split(images_dir: str, csv_path: str, out_base: str, split_name: str, rng):
    anns_by_image = read_annotations_csv(csv_path)
    ensure_dirs(out_base, split_name)

    # Recorre todas las imágenes mencionadas en el CSV
    count_pos, count_neg = 0, 0
    for fn, anns in anns_by_image.items():
        img_path = resolve_image_path(images_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] no se pudo leer", img_path)
            continue
        H, W = img.shape[:2]

        boxes = []
        # POSITIVOS: cualquier clase en POSITIVE_NAMES cuenta como bache/obstáculo
        for a in anns:
            if a["cls"] not in POSITIVE_NAMES:
                continue
            x1, y1 = a["xmin"], a["ymin"]
            x2, y2 = a["xmax"], a["ymax"]
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1: 
                continue
            padw, padh = w * PADDING, h * PADDING
            xb, yb = x1 - padw, y1 - padh
            wb, hb = w + 2*padw, h + 2*padh
            xb, yb, wb, hb = clip_box(xb, yb, wb, hb, W, H)
            boxes.append((xb, yb, wb, hb))

            stem = Path(fn).stem
            out_p = os.path.join(out_base, split_name, POS_CLASS,
                                 f"{stem}_x{int(xb)}_y{int(yb)}_w{int(wb)}_h{int(hb)}.jpg")
            if save_patch(img, (xb, yb, wb, hb), out_p):
                count_pos += 1

        # NEGATIVOS (zonas sin solape)
        negs = random_negatives(W, H, boxes, NEGATIVES_PER_IMAGE, rng)
        for (xn, yn, wn, hn) in negs:
            stem = Path(fn).stem
            out_n = os.path.join(out_base, split_name, NEG_CLASS,
                                 f"{stem}_negx{int(xn)}_negy{int(yn)}_w{int(wn)}_h{int(hn)}.jpg")
            if save_patch(img, (xn, yn, wn, hn), out_n):
                count_neg += 1

    print(f"[OK] {split_name}: +{count_pos} {POS_CLASS}, +{count_neg} {NEG_CLASS}")

if __name__ == "__main__":
    """
    Uso (PowerShell):
      python scripts\csv_to_classification.py `
        --in  dataset\Pothole\train\images `
        --ann dataset\Pothole\train\_annotations.csv `
        --split train `
        --out data
      # Repite para valid -> split "valid" (mapeará a 'val') y test
    """
    import argparse, random as pyrand
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Carpeta de imágenes del split")
    ap.add_argument("--ann", dest="ann_path", required=True, help="Ruta al _annotations.csv")
    ap.add_argument("--split", dest="split", required=True, choices=["train","val","valid","test"])
    ap.add_argument("--out", dest="out_dir", required=True, help="Carpeta base de salida (data)")
    args = ap.parse_args()

    split = "val" if args.split == "valid" else args.split
    rng = pyrand.Random(42)
    process_split(args.in_dir, args.ann_path, args.out_dir, split, rng)
    print("[OK] Convertido split:", split)


