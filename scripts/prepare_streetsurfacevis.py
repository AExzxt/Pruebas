# scripts/prepare_streetsurfacevis.py
import argparse, csv, os, random
from pathlib import Path
from PIL import Image

MAP_SURF = {
    "asphalt": "liso",
    "concrete": "liso",
    "paving_stones": "grava",
    "sett": "grava",
    "unpaved": "tierra",
}

def crop_bottom_center(img: Image.Image):
    w,h = img.size
    # recorte recomendado: mitad inferior y zona central 50%
    # (x1=0.25w, y1=0.5h) a (x2=0.75w, y2=h)
    return img.crop((0.25*w, 0.5*h, 0.75*w, h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Carpeta del dataset StreetSurfaceVis (contiene s_1024, CSV...)")
    ap.add_argument("--size_dir", default="s_1024", help="s_256 | s_1024 | s_2048 | s_original")
    ap.add_argument("--csv", default="streetSurfaceVis_v1_0.csv")
    ap.add_argument("--out", required=True, help="Destino base (data/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    img_dir = root / args.size_dir
    csv_path = root / args.csv
    out = Path(args.out)

    # crea carpetas
    for split in ["train","val","test"]:
        for c in ["liso","grava","tierra","obstaculo"]:
            (out/split/c).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            st = (r["surface_type"] or "").strip().lower()
            if st not in MAP_SURF:
                continue
            cls = MAP_SURF[st]
            img_name = r["mapillary_image_id"] + ".jpg"
            # imágenes están distribuidas en subcarpetas por tamaño:
            p = img_dir / img_name
            if not p.exists():
                # algunas versiones incluyen subcarpetas, buscar recursivo
                matches = list(img_dir.rglob(img_name))
                if not matches: 
                    continue
                p = matches[0]
            train_flag = (r["train"].strip().lower() == "true")
            rows.append((str(p), cls, train_flag))

    # split: su test sugerido = test, del resto 90/10 -> train/val
    test = [r for r in rows if not r[2]]
    traincand = [r for r in rows if r[2]]
    random.shuffle(traincand)
    n_val = max(1, int(0.1*len(traincand)))
    val = traincand[:n_val]
    train = traincand[n_val:]

    print(f"[INFO] StreetSurfaceVis -> train={len(train)} val={len(val)} test={len(test)}")

    def dump(items, split):
        for src, cls, _ in items:
            try:
                img = Image.open(src).convert("RGB")
                img = crop_bottom_center(img)
                dst = out / split / cls / (Path(src).stem + "_crop.jpg")
                img.save(dst, quality=90)
            except Exception as e:
                print(f"[WARN] fallo {src}: {e}")

    dump(train, "train")
    dump(val, "val")
    dump(test, "test")
    print("[DONE] StreetSurfaceVis preparado.")

if __name__ == "__main__":
    main()



