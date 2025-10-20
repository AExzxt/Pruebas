# prepare_offroad.py
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
import random

# --- Utilidades ---
def find_image(root: Path, image_name: str) -> Path | None:
    """
    Off-Road estructura: Images/Images/YYYY-MM-DD/<filename>
    image_name normalmente viene como 'YYYY-MM-DD_HH-MM-SS.jpg' o similar.
    Recorremos subcarpetas de fecha y buscamos coincidencia.
    """
    images_root = root / "Images" / "Images"
    if not images_root.exists():
        return None
    # búsqueda directa por glob (más simple)
    matches = list(images_root.rglob(image_name))
    return matches[0] if matches else None

def copy_img(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def stratified_split(df, label_col, val_ratio=0.1, seed=42):
    random.seed(seed)
    val_idxs = []
    for lbl, sub in df.groupby(label_col):
        idxs = list(sub.index)
        random.shuffle(idxs)
        k = max(1, int(len(idxs) * val_ratio))
        val_idxs += idxs[:k]
    df_val = df.loc[val_idxs]
    df_train = df.drop(val_idxs)
    return df_train, df_val

# --- Mapeos de rugosidad a tus clases (para modo classification) ---
# Asumimos:
# - 2 clases: {0,1} -> {liso, tierra} (o {liso, grava}) según preferencia; usamos {liso, tierra}.
# - 3 clases: {0,1,2} -> {liso, grava, tierra}
# - 4 clases: {0,1,2,3} -> {liso, grava, tierra} colapsando 2,3 a tierra (o dejando 3 como 'muy_tierra').
MAP_2 = {0: "liso", 1: "tierra"}
MAP_3 = {0: "liso", 1: "grava", 2: "tierra"}
MAP_4 = {0: "liso", 1: "grava", 2: "tierra", 3: "tierra"}  # colapsamos 2 y 3

def pick_mapping(k):
    if k == 2: return MAP_2
    if k == 3: return MAP_3
    if k == 4: return MAP_4
    raise ValueError("k debe ser 2, 3 o 4")

def main():
    ap = argparse.ArgumentParser(description="Prepara Off-Road Terrain para clasificación o multitarea.")
    ap.add_argument("--root", required=True, help="Carpeta raíz del dataset Off_Road_Terrain_Dataset")
    ap.add_argument("--labels", default="tsm_1_labels.csv",
                    help="Nombre de CSV en ImageLabels (tsm_1_labels.csv o tsm_2_labels.csv)")
    ap.add_argument("--out", required=True, help="Carpeta de salida (p.ej. A:\\...\\cnn-terreno\\data)")
    ap.add_argument("--mode", choices=["classification","multitask"], default="classification",
                    help="classification: copia a carpetas por clase; multitask: genera CSV maestro")
    ap.add_argument("--k", type=int, choices=[2,3,4], default=3, help="Nº de clases de rugosidad (2/3/4)")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Porción para validación")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    csv_path = root / "ImageLabels" / args.labels
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}")

    df = pd.read_csv(csv_path)
    # Inferimos columnas de rugosidad según k:
    # Suele haber columnas como: roughness_k2, roughness_k3, roughness_k4, etc.
    # Buscamos heurísticamente:
    candidates = [c for c in df.columns if "k=" in c or "k2" in c or "k3" in c or "k=2" in c or "k=3" in c or "k=4" in c]
    # Intento conocido: columnas típicas en el readme
    prefer = {
        2: ["k=2", "k2"],
        3: ["k=3", "k3"],
        4: ["k=4", "k4"],
    }[args.k]
    label_col = None
    for p in prefer:
        for c in df.columns:
            if p in c.replace(" ", "").lower():
                label_col = c
                break
        if label_col: break
    # fallback: primera que contenga 'k'
    if not label_col:
        kcols = [c for c in df.columns if "k" in c.lower()]
        if not kcols:
            raise RuntimeError("No se encontró columna de rugosidad con k=2/3/4 en el CSV.")
        label_col = kcols[0]

    # columna de nombre de imagen: suele ser 'image_name' o 'filename' o similar; busquemos
    name_col = None
    for cand in ["image_name","filename","file","image","img","utc_timestamp_image","img_name"]:
        if cand in df.columns:
            name_col = cand
            break
    if not name_col:
        # intenta detectar por columnas que tengan 'jpg' en valores
        for c in df.columns:
            vals = df[c].astype(str)
            if vals.str.contains(".jpg", case=False, regex=False).any():
                name_col = c
                break
    if not name_col:
        raise RuntimeError("No encontré columna con el nombre de la imagen.")

    # filtra filas válidas (sin NaN en label)
    df = df[~df[label_col].isna()].copy()
    df[label_col] = df[label_col].astype(int)

    # agrega ruta real
    paths = []
    missing = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="[buscar imágenes]"):
        img_name = str(r[name_col]).strip()
        if not img_name.lower().endswith((".jpg",".jpeg",".png")):
            img_name += ".jpg"
        p = find_image(root, img_name)
        if p is None:
            missing += 1
            paths.append(None)
        else:
            paths.append(str(p))
    df["path"] = paths
    df = df[~df["path"].isna()].copy()

    print(f"[INFO] Imágenes con etiqueta rugosidad: {len(df)} (perdidas: {missing})")
    print(f"[INFO] Columna de rugosidad usada: '{label_col}' ; archivo labels: {csv_path.name}")

    if args.mode == "multitask":
        # Guardamos un CSV maestro con path y roughness_label
        out_csv = out / "offroad_multitask_labels.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df[["path", label_col]].rename(columns={label_col: "roughness_label"}).to_csv(out_csv, index=False)
        print(f"[DONE] CSV multitarea generado: {out_csv}")
        return

    # --- Modo classification: mapeo rugosidad -> {liso,grava,tierra}
    mapping = pick_mapping(args.k)
    df["cls"] = df[label_col].map(mapping)

    # split estratificado por etiqueta rugosidad original (mantiene distribución)
    df_train, df_val = stratified_split(df, label_col, val_ratio=args.val_ratio, seed=42)
    print(f"[INFO] split -> train={len(df_train)}  val={len(df_val)}  test=0 (este dataset no trae test)")

    # Copiamos
    stats = defaultdict(Counter)
    for split_name, frame in [("train", df_train), ("val", df_val)]:
        for _, r in tqdm(frame.iterrows(), total=len(frame), desc=f"[copiar {split_name}]"):
            src = Path(r["path"])
            cls = r["cls"]
            dst = out / split_name / cls / src.name
            try:
                copy_img(src, dst)
                stats[split_name][cls] += 1
            except Exception:
                pass

    print("\n[RESUMEN] Conteo por split/clase (Off-Road -> clasificación):")
    for s in ["train","val"]:
        print(f"  - {s}:")
        for cls, n in sorted(stats[s].items()):
            print(f"      {cls:8s}: {n}")

    print("\n[NOTE] Este mapeo es un proxy (rugosidad→tipo). Para resultados sólidos, usa 'multitask' y entrena con dos cabezas.")

if __name__ == "__main__":
    main()

