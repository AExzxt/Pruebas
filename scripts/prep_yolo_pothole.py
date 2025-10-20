# scripts/prep_yolo_pothole.py
import argparse, csv, json, random, shutil
from pathlib import Path
from PIL import Image

random.seed(1337)

# --------------------------------------------
# Utilidad: VOC XML -> YOLO
# --------------------------------------------
def parse_voc_xml(xml_path):
    import xml.etree.ElementTree as ET
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return boxes
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in {"pothole", "potholes", "pothole1"}:
            # ignoramos otras clases (si existieran)
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        # VOC -> YOLO (cx, cy, w, h) normalizados
        cx = ((xmin + xmax) / 2.0) / W
        cy = ((ymin + ymax) / 2.0) / H
        ww = (xmax - xmin) / W
        hh = (ymax - ymin) / H
        boxes.append((0, cx, cy, ww, hh))  # cls=0 (pothole)
    return boxes

# --------------------------------------------
# Utilidad: CSV -> YOLO (asume columnas típicas)
# filename,xmin,ymin,xmax,ymax,(class)
# --------------------------------------------
def parse_csv_annotations(csv_path):
    rows = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Columnas toleradas
        cand_file = [c for c in reader.fieldnames if c.lower() in ("filename","file","image","img","path")]
        cand_xmin = [c for c in reader.fieldnames if "xmin" in c.lower()]
        cand_ymin = [c for c in reader.fieldnames if "ymin" in c.lower()]
        cand_xmax = [c for c in reader.fieldnames if "xmax" in c.lower()]
        cand_ymax = [c for c in reader.fieldnames if "ymax" in c.lower()]
        cand_cls  = [c for c in reader.fieldnames if c.lower() in ("class","name","label")]
        if not (cand_file and cand_xmin and cand_ymin and cand_xmax and cand_ymax):
            raise RuntimeError(f"CSV sin columnas bbox esperadas: {reader.fieldnames}")

        cf, cx1, cy1, cx2, cy2 = cand_file[0], cand_xmin[0], cand_ymin[0], cand_xmax[0], cand_ymax[0]
        cc = cand_cls[0] if cand_cls else None

        for r in reader:
            fn = Path(r[cf]).name
            if cc:
                if str(r[cc]).strip().lower() not in ("pothole","potholes","pothole1"):
                    continue
            xmin = float(r[cx1]); ymin = float(r[cy1]); xmax = float(r[cx2]); ymax = float(r[cy2])
            # altura/anchura las obtendremos al abrir la imagen cuando toque
            rows.setdefault(fn, []).append(("bbox", xmin, ymin, xmax, ymax))
    return rows  # dict: filename -> list of ("bbox", xmin, ymin, xmax, ymax)

# --------------------------------------------
# Recorte recomendado por StreetSurfaceVis:
# crop((0.25W, 0.5H) -> (0.75W, H))
# --------------------------------------------
def crop_bottom_center(img: Image.Image) -> Image.Image:
    w, h = img.size
    x1 = int(0.25 * w); x2 = int(0.75 * w)
    y1 = int(0.50 * h); y2 = h
    return img.crop((x1, y1, x2, y2))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_yolo_txt(txt_path: Path, yolo_boxes):
    if not yolo_boxes:
        # archivo vacío = negativo
        txt_path.write_text("", encoding="utf-8")
    else:
        with open(txt_path, "w", encoding="utf-8") as f:
            for (clsid, cx, cy, ww, hh) in yolo_boxes:
                f.write(f"{clsid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")

def convert_kaggle_ssd_to_yolo(kaggle_root: Path, out_root: Path, split_ratio=0.1):
    """
    Intenta:
      - VOC: busca 'Annotations/*.xml' y 'Images/*'
      - CSV: busca un .csv con bboxes y 'Images/*'
    Crea: images/train,val y labels/train,val
    """
    print(f"[KAGGLE] Origen: {kaggle_root}")
    images_dir = None
    ann_mode = None
    ann_dir = None
    ann_csv = None

    # Buscar imágenes
    for cand in ["Images", "images", "img", "JPEGImages"]:
        d = kaggle_root / cand
        if d.exists():
            images_dir = d
            break
    if images_dir is None:
        # a veces hay una subcarpeta dentro de 'Images'
        if (kaggle_root / "Images" / "Images").exists():
            images_dir = kaggle_root / "Images" / "Images"
    if images_dir is None:
        raise RuntimeError("No encontré carpeta de imágenes en el dataset de Kaggle.")

    # Buscar anotaciones VOC
    for cand in ["Annotations","annotations","ann","xml"]:
        d = kaggle_root / cand
        if d.exists() and any(p.suffix.lower()==".xml" for p in d.glob("*.xml")):
            ann_mode = "voc"
            ann_dir = d
            break

    # Si no hay VOC, probar CSV
    if ann_mode is None:
        csvs = list(kaggle_root.glob("*.csv"))
        if csvs:
            ann_mode = "csv"
            ann_csv = csvs[0]

    if ann_mode is None:
        raise RuntimeError("No encontré anotaciones (ni VOC-XML ni CSV) en el dataset de Kaggle.")

    print(f"[KAGGLE] Imágenes: {images_dir}")
    print(f"[KAGGLE] Anotaciones modo: {ann_mode}")

    # listar imágenes
    img_paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    if len(img_paths) == 0:
        raise RuntimeError("No se encontraron imágenes en Kaggle Images/")

    # cargar csv si hace falta
    csv_map = None
    if ann_mode == "csv":
        csv_map = parse_csv_annotations(ann_csv)

    # split
    random.shuffle(img_paths)
    n_val = max(1, int(len(img_paths)*split_ratio))
    val_set = set(img_paths[:n_val])
    train_set = set(img_paths[n_val:])

    for split, subset in [("train", train_set), ("val", val_set)]:
        for p in subset:
            rel = p.stem
            # cargar imagen para normalizar (si CSV necesita W/H)
            img = Image.open(p).convert("RGB")
            W,H = img.size

            # construir boxes
            yolo_boxes = []
            if ann_mode == "voc":
                # buscar xml con mismo stem (carpetas distintas)
                xml_guess = list((ann_dir).glob(f"{p.stem}.xml"))
                if not xml_guess:
                    # sin anotación -> negativo
                    pass
                else:
                    yolo_boxes = parse_voc_xml(xml_guess[0])
            else:
                # CSV
                rows = csv_map.get(p.name, [])
                for _, xmin, ymin, xmax, ymax in rows:
                    cx = ((xmin + xmax) / 2.0) / W
                    cy = ((ymin + ymax) / 2.0) / H
                    ww = (xmax - xmin) / W
                    hh = (ymax - ymin) / H
                    yolo_boxes.append((0, cx, cy, ww, hh))

            # copiar imagen y txt
            out_img = out_root / "images" / split / f"{rel}.jpg"
            out_lbl = out_root / "labels" / split / f"{rel}.txt"
            ensure_dir(out_img.parent); ensure_dir(out_lbl.parent)
            img.save(out_img, quality=95)
            write_yolo_txt(out_lbl, yolo_boxes)

    print("[KAGGLE] Conversión a YOLO terminada.")

def add_ssv_negatives(ssv_root: Path, out_root: Path, max_train=3000, max_val=400):
    """
    Añade negativos desde StreetSurfaceVis (carpeta s_1024 o s_256…).
    - Recorta la parte baja-centro (recomendación del paper)
    - Genera archivos .txt vacíos (negativos)
    Usamos su columna 'train' del CSV si está disponible (streetSurfaceVis_v1_0.csv).
    Si no está, haremos split fijo.
    """
    # localizar CSV
    ssv_csv = None
    for c in ["streetSurfaceVis_v1_0.csv","streetsurfacevis_v1_0.csv"]:
        if (ssv_root.parent / c).exists():
            ssv_csv = ssv_root.parent / c
            break

    # recolectar imágenes
    imgs = [p for p in ssv_root.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    if not imgs:
        print("[SSV] No encontré imágenes en", ssv_root)
        return

    # si hay CSV y columna 'train' -> respetamos split
    train_imgs, val_imgs = [], []
    if ssv_csv:
        import pandas as pd
        df = pd.read_csv(ssv_csv)
        # mapillary_image_id -> nombre de archivo
        # Los nombres de archivo son el id con extensión .jpg (suele ser así)
        ssv_index = {str(r["mapillary_image_id"]): bool(r["train"]) for _,r in df.iterrows()}
        for p in imgs:
            stem = p.stem
            if stem in ssv_index:
                (train_imgs if ssv_index[stem] else val_imgs).append(p)
        if not train_imgs and not val_imgs:
            # fallback si no coincidieron nombres
            random.shuffle(imgs)
            val_cut = max(1, int(len(imgs)*0.1))
            val_imgs = imgs[:val_cut]
            train_imgs = imgs[val_cut:]
    else:
        # sin CSV -> split simple
        random.shuffle(imgs)
        val_cut = max(1, int(len(imgs)*0.1))
        val_imgs = imgs[:val_cut]
        train_imgs = imgs[val_cut:]

    # limitar cantidades
    train_imgs = train_imgs[:max_train]
    val_imgs   = val_imgs[:max_val]

    for split, subset in [("train", train_imgs), ("val", val_imgs)]:
        for p in subset:
            try:
                img = Image.open(p).convert("RGB")
                img = crop_bottom_center(img)
                out_img = out_root / "images" / split / f"ssv_{p.stem}.jpg"
                out_lbl = out_root / "labels" / split / f"ssv_{p.stem}.txt"
                ensure_dir(out_img.parent); ensure_dir(out_lbl.parent)
                img.save(out_img, quality=95)
                write_yolo_txt(out_lbl, [])  # negativo
            except Exception as e:
                print("[SSV] skip", p, e)

    print(f"[SSV] Añadidos negativos: train={len(train_imgs)}, val={len(val_imgs)}")

def write_data_yaml(out_yaml: Path, out_root: Path):
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "nc": 1,
        "names": ["pothole"]
    }
    import yaml
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print("[YAML] escrito en:", out_yaml)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_root", required=True, help="Carpeta raíz del dataset de Kaggle (SSD pothole)")
    ap.add_argument("--ssv_root",    required=True, help="Carpeta con imágenes SSV (usa s_1024 o s_256)")
    ap.add_argument("--out_root",    required=True, help="Salida YOLO (e.g. A:\\...\\data_yolo_pothole)")
    ap.add_argument("--val_ratio",   type=float, default=0.1, help="Porción val si Kaggle no trae split")
    args = ap.parse_args()

    kaggle_root = Path(args.kaggle_root)
    ssv_root    = Path(args.ssv_root)
    out_root    = Path(args.out_root)

    if out_root.exists():
        print("[WARN] out_root ya existe. No se borra nada. Se sobreescribe contenido coincidente.")
    ensure_dir(out_root)

    convert_kaggle_ssd_to_yolo(kaggle_root, out_root, split_ratio=args.val_ratio)
    add_ssv_negatives(ssv_root, out_root)
    write_data_yaml(out_root / "pothole_yolo.yaml", out_root)
    print("[DONE] Dataset YOLO listo.")

if __name__ == "__main__":
    main()




