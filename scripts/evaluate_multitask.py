# evaluate_multitask.py
# Evalúa el modelo multitarea:
#  - Tarea 1 (tipo de superficie) usando data/{val,test}/(liso|grava|tierra|obstaculo)
#  - Tarea 2 (rugosidad k-clases) usando Off_Road_Terrain_Dataset + CSV con columna discreta
#
# Uso (ejemplo):
# python scripts/evaluate_multitask.py ^
#   --ckpt "A:\Varios\MobileNet\cnn-terreno\models\multitask_two_loaders.pt" ^
#   --data "A:\Varios\MobileNet\cnn-terreno\data" ^
#   --offroad_root "A:\Varios\MobileNet\cnn-terreno\dataset\Off_Road_Terrain_Dataset\Images\Images" ^
#   --offroad_csv  "A:\Varios\MobileNet\cnn-terreno\dataset\Off_Road_Terrain_Dataset\ImageLabels\tsm_2_labels.csv" ^
#   --label_column tsm2_k3 ^
#   --outdir "A:\Varios\MobileNet\cnn-terreno\eval" ^
#   --workers 0

import argparse, os, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image
import pandas as pd

# opcional pero útil para métricas/plots
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# -----------------------------
# Config / utilidades
# -----------------------------

DEFAULT_CLASS_NAMES = ["liso", "grava", "tierra", "obstaculo"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_bottom_center(img: Image.Image):
    """Crop recomendado por StreetSurfaceVis: mitad inferior, centro horizontal."""
    w, h = img.size
    # (x1, y1, x2, y2) = (0.25W, 0.5H, 0.75W, 1.0H)
    return img.crop((0.25 * w, 0.5 * h, 0.75 * w, h))


def pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# -----------------------------
# Datasets
# -----------------------------

class TypeFolder(Dataset):
    """Eval de tipo (liso/grava/tierra/obstaculo) desde carpetas data/{val,test}/clase/imagen."""
    def __init__(self, split_dir: Path, class_names, img_size=224):
        self.items = []  # (path, class_id)
        self.class_to_id = {c: i for i, c in enumerate(class_names)}
        for cname in class_names:
            cdir = split_dir / cname
            if not cdir.exists():
                continue
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.items.append((p, self.class_to_id[cname]))

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = pil_loader(p)
        return self.tf(img), y, str(p)


class OffRoadRough(Dataset):
    """
    Eval de rugosidad: busca imágenes por nombre de archivo dentro de --offroad_root
    y usa las etiquetas discretas de --label_column en el CSV.
    """
    def __init__(self, offroad_root: Path, csv_path: Path, label_col: str, img_size=224):
        offroad_root = Path(offroad_root)
        df = pd.read_csv(csv_path)

        if label_col not in df.columns:
            raise ValueError(f"La columna '{label_col}' no existe en el CSV. Columnas: {list(df.columns)}")

        # Indexa rutas por filename (case-insensitive) recorriendo recursivo offroad_root
        by_name = {}
        for p in offroad_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                by_name[p.name.lower()] = p

        self.items = []  # (path, rough_label)
        y_values = set()

        for _, r in df.iterrows():
            # En los CSVs, la columna 'image' suele tener nombres con extensión (p.ej., 2020-07-28_10-15-00_123.jpg)
            # Asegúrate de que coincide con los archivos en offroad_root
            img_name = str(r.get("image", "")).strip()
            if not img_name:
                continue
            p = by_name.get(img_name.lower(), None)
            if p is None:
                # Si no coincide por nombre exacto, intentamos por stem con extensiones comunes
                stem = Path(img_name).stem.lower()
                found = None
                for ext in IMG_EXTS:
                    cand = by_name.get((stem + ext).lower(), None)
                    if cand is not None:
                        found = cand
                        break
                p = found

            if p is None:
                continue  # no encontrada

            y = r[label_col]
            # Convertimos a int por seguridad (suele venir 0/1/2 ...)
            try:
                y = int(y)
            except Exception:
                # Si viene en float tipo 0.0/1.0/2.0
                y = int(float(y))
            self.items.append((p, y))
            y_values.add(y)

        if not self.items:
            raise RuntimeError(
                "No se encontraron imágenes para evaluar rugosidad. "
                "Verifica que los nombres del CSV coinciden con los archivos de --offroad_root."
            )

        self.k = len(sorted(y_values))  # número de clases efectivas

        self.tf = transforms.Compose([
            transforms.Lambda(crop_bottom_center),      # <- GLOBAL para que sea 'pickleable'
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = pil_loader(p)
        return self.tf(img), y, str(p)


# -----------------------------
# Modelo (igual que el de entrenamiento)
# -----------------------------

class DualHeadNet(nn.Module):
    def __init__(self, num_types=4, rough_k=3, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet50":
            m = models.resnet50(weights=None)
            dim = m.fc.in_features
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
            dim = m.classifier[-1].in_features
            self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
        else:
            raise ValueError("Backbone no soportado")

        self.head_type = nn.Linear(dim, num_types)
        self.head_rough = nn.Linear(dim, rough_k)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return self.head_type(feats), self.head_rough(feats)


# -----------------------------
# Métricas / helpers
# -----------------------------

@torch.no_grad()
def eval_type(model, loader, device, class_names, outdir: Path, split_name: str):
    model.eval()
    y_true, y_pred = [], []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits_t, _ = model(x)
        preds = logits_t.argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()

    report_txt = ""
    if HAVE_SK:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        report_txt = rep
        # guardar CM
        if HAVE_PLT:
            plt.figure(figsize=(6, 5))
            import numpy as np
            im = plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - TYPE ({split_name})")
            plt.colorbar(im)
            tick_marks = range(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
            plt.tight_layout()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            (outdir / f"cm_type_{split_name}.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / f"cm_type_{split_name}.png", bbox_inches="tight", dpi=150)
            plt.close()

        with open(outdir / f"classif_report_type_{split_name}.txt", "w", encoding="utf-8") as f:
            f.write(rep)

    return acc, report_txt


@torch.no_grad()
def eval_rough(model, loader, device, k, outdir: Path, split_name: str):
    model.eval()
    y_true, y_pred = [], []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        _, logits_r = model(x)
        preds = logits_r.argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()

    if HAVE_SK:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
        if HAVE_PLT:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - ROUGH ({split_name})")
            plt.colorbar(im)
            plt.xticks(range(k), [str(i) for i in range(k)])
            plt.yticks(range(k), [str(i) for i in range(k)])
            plt.tight_layout()
            (outdir / f"cm_rough_{split_name}.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / f"cm_rough_{split_name}.png", bbox_inches="tight", dpi=150)
            plt.close()

        with open(outdir / f"classif_report_rough_{split_name}.txt", "w", encoding="utf-8") as f:
            try:
                rep = classification_report(y_true, y_pred, target_names=[f"r{i}" for i in range(k)],
                                            digits=4, zero_division=0)
            except Exception:
                rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
            f.write(rep)

    return acc


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Ruta del checkpoint .pt")
    ap.add_argument("--data", required=True, help="Carpeta data con {val,test}/clases")
    ap.add_argument("--offroad_root", required=True, help="Raíz de imágenes Off_Road_Terrain_Dataset/Images/Images")
    ap.add_argument("--offroad_csv", required=True, help="CSV con etiquetas de rugosidad")
    ap.add_argument("--label_column", default="tsm2_k3", help="Columna discreta para rugosidad (p.ej., tsm2_k2/k3/k4)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0, help="En Windows usa 0 para evitar errores de pickle")
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Carga checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")  # mantener weights_only=False por compatibilidad
    ckpt_args = ckpt.get("args", {})
    class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)
    rough_k = int(ckpt_args.get("rough_k", 3))
    backbone = ckpt_args.get("backbone", "resnet50")

    # Construye modelo y carga pesos
    model = DualHeadNet(num_types=len(class_names), rough_k=rough_k, backbone=backbone)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    # -------- Eval TYPE (data/{val,test}) --------
    data_root = Path(args.data)

    results = {
        "type": {},
        "rough": {}
    }

    for split in ["val", "test"]:
        split_dir = data_root / split
        if split_dir.exists():
            ds = TypeFolder(split_dir, class_names=class_names, img_size=args.img_size)
            if len(ds) == 0:
                print(f"[WARN] split '{split}' no tiene imágenes para tipo.")
                continue
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
            acc, rep = eval_type(model, dl, device, class_names, outdir, split)
            results["type"][split] = {"acc": acc}
            print(f"[TYPE-{split}] acc={acc:.4f}")
        else:
            print(f"[INFO] split '{split}' no existe en data/ para tipo.")

    # -------- Eval ROUGH (Off-Road) --------
    rough_val = OffRoadRough(offroad_root=Path(args.offroad_root),
                             csv_path=Path(args.offroad_csv),
                             label_col=args.label_column,
                             img_size=args.img_size)
    dl_rough_val = DataLoader(rough_val, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    rough_acc_val = eval_rough(model, dl_rough_val, device, rough_val.k, outdir, "val")
    results["rough"]["val"] = {"acc": rough_acc_val, "k": rough_val.k}
    print(f"[ROUGH-val] acc={rough_acc_val:.4f} (k={rough_val.k})")

    # Guarda reporte JSON
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "rough_k": rough_k,
            "results": results
        }, f, indent=2)

    print(f"[DONE] Reporte en: {outdir / 'report.json'}")


if __name__ == "__main__":
    main()



