import cv2, os, glob
import argparse

def process_folder(in_dir, out_dir, roi_box, gray=False, size=(128,128)):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths += glob.glob(os.path.join(in_dir, ext))
    if not paths:
        print(f"[WARN] No se encontraron imágenes en {in_dir}")
        return
    x, y, w, h = roi_box
    count = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None: 
            print(f"[WARN] No se pudo leer {p}")
            continue
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            print(f"[WARN] ROI fuera de rango en {p}")
            continue
        roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        if gray:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        name = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{name}_roi.jpg")
        # si es gris, guardar en 1 canal
        if gray:
            cv2.imwrite(out_path, roi)
        else:
            cv2.imwrite(out_path, roi)
        count += 1
    print(f"[OK] Guardadas {count} imágenes en {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Carpeta con imágenes originales")
    ap.add_argument("--out_dir", required=True, help="Carpeta destino (ej. data/train/liso)")
    ap.add_argument("--roi", required=True, help="ROI en formato x,y,w,h")
    ap.add_argument("--gray", action="store_true", help="Convertir a escala de grises")
    ap.add_argument("--size", default="128,128", help="Tamaño de salida W,H")
    args = ap.parse_args()

    x,y,w,h = map(int, args.roi.split(","))
    W,H = map(int, args.size.split(","))
    process_folder(args.in_dir, args.out_dir, (x,y,w,h), gray=args.gray, size=(W,H))
