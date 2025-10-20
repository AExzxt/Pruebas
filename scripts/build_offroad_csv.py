import argparse
from pathlib import Path
import pandas as pd
import re
import sys

def guess_label_column(cols, prefer_substrings=("k=3","k_3","kmeans","cluster")):
    # intenta elegir una columna de etiquetas discretas (k clusters)
    low = [c.lower() for c in cols]
    # 1) preferidas
    for want in prefer_substrings:
        for i,c in enumerate(low):
            if want in c:
                return cols[i]
    # 2) columnas con pocos valores únicos (<=10) y numéricas
    return None

def normalize_label_series(s):
    # convierte cualquier etiqueta a enteros 0..K-1 ordenados por valor
    uniq = sorted(pd.unique(s.dropna()))
    mapping = {v:i for i,v in enumerate(uniq)}
    return s.map(mapping), mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"Carpeta data con train/val/test (A:\...\data)")
    ap.add_argument("--offroad_root", required=True, help=r"Carpeta Off_Road_Terrain_Dataset\Images\Images")
    ap.add_argument("--label_csv", required=True, help=r"CSV de ImageLabels (p.ej. tsm_2_labels.csv)")
    ap.add_argument("--label_column", default=None, help="Nombre exacto de la columna de clase (si no, se intenta detectar)")
    ap.add_argument("--out_csv", default=None, help="Salida (por defecto: data/offroad_multitask_labels.csv)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    off_img_root = Path(args.offroad_root).resolve()
    label_csv = Path(args.label_csv).resolve()
    out_csv = Path(args.out_csv) if args.out_csv else (data_root / "offroad_multitask_labels.csv")

    if not data_root.exists():
        sys.exit(f"[ERROR] data_root no existe: {data_root}")
    if not off_img_root.exists():
        sys.exit(f"[ERROR] offroad_root no existe: {off_img_root}")
    if not label_csv.exists():
        sys.exit(f"[ERROR] label_csv no existe: {label_csv}")

    print(f"[INFO] Cargando etiquetas: {label_csv}")
    df = pd.read_csv(label_csv)

    # Intentamos columnas típicas para id de imagen
    # El dataset Off-Road suele nombrar archivos por timestamp; ajustamos a tu CSV real:
    candidate_id_cols = ["image", "filename", "img", "utc_timestamp", "timestamp", "name"]
    id_col = None
    for c in candidate_id_cols:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        # si hay columna con “.jpg” o similar, úsala
        for c in df.columns:
            sample = str(df[c].astype(str).iloc[0]).lower()
            if any(ext in sample for ext in [".jpg",".jpeg",".png",".bmp",".webp"]):
                id_col = c
                break
    if id_col is None:
        print("[ERROR] No encontré columna de nombre/archivo en el CSV.")
        print("Columnas disponibles:", list(df.columns))
        sys.exit(1)

    # Detecta/elige columna de etiqueta
    label_col = args.label_column
    if label_col is None or label_col not in df.columns:
        label_col = guess_label_column(list(df.columns))
        if label_col is None:
            print("[ERROR] No pude detectar automáticamente la columna de etiquetas discretas.")
            print("Columnas disponibles:", list(df.columns))
            sys.exit(1)

    # nos quedamos con id + label
    sub = df[[id_col, label_col]].copy()
    sub = sub.dropna()
    # normalizamos nombre base (stem)
    def to_stem(x):
        # extrae nombre sin extensión; si viene timestamp/clave, deja sólo parte alfanumérica
        s = Path(str(x)).name  # por si trae ruta
        s = re.sub(r"\.(jpg|jpeg|png|bmp|webp)$","", s, flags=re.IGNORECASE)
        return s

    sub["stem"] = sub[id_col].astype(str).apply(to_stem)

    # normaliza labels a 0..K-1
    sub[label_col], mapping = normalize_label_series(sub[label_col])
    k_classes = len(mapping)
    if k_classes < 2:
        print("[WARN] La columna elegida parece no ser una clasificación discreta (K<2).")
    print(f"[INFO] Columna de clase: '{label_col}' -> K={k_classes} (mapeo: {mapping})")

    # index por stem
    label_by_stem = dict(zip(sub["stem"], sub[label_col]))

    # recorre data/{train,val} y junta las que coinciden por stem
    rows = []
    total_imgs = 0
    matched = 0
    for split in ["train", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp"):
                total_imgs += 1
                stem = p.stem
                if stem in label_by_stem:
                    rows.append({"path": str(p.resolve()), "roughness_label": int(label_by_stem[stem])})
                    matched += 1

    if not rows:
        print("[WARN] No se encontraron coincidencias por 'stem'.")
        print("Posibles causas:")
        print(" - Los nombres de archivo cambiaron al copiar a data/")
        print(" - En el CSV los nombres no coinciden con los de data/")
        print("Sugerencia: imprime unos ejemplos de 'stem' del CSV y de data/")
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] offroad_multitask_labels.csv escrito: {out_csv}")
        print(f"[OK] Coincidencias: {matched} / {total_imgs} imágenes en (train+val)")

if __name__ == "__main__":
    main()


