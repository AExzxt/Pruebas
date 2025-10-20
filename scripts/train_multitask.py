import argparse, os, math, time, json
from pathlib import Path
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# AMP moderno
from torch.amp import GradScaler, autocast

print("[SCRIPT] train_multitask.py :: VERSION = mt-2.0")


# --- Config ---
CLASS_NAMES = ["liso", "grava", "tierra", "obstaculo"]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def pil_loader(path):
    with Image.open(path) as img:
        return img.convert("RGB")

class MultiTaskFolder(Dataset):
    """
    Carga imágenes desde carpetas (tipo de superficie) y,
    si la ruta está en offroad_csv, añade etiqueta de rugosidad.
    """
    def __init__(self, root, split, offroad_csv=None, img_size=224, rough_k=3):
        self.root = Path(root) / split
        self.items = []  # (path, type_id, rough_id_or_-1)

        # índice de rugosidad (abspath -> label)
        self.rough = {}
        if offroad_csv and Path(offroad_csv).exists():
            df = pd.read_csv(offroad_csv)
            # Se espera una columna "path" y una "roughness_label"
            if "path" not in df.columns or "roughness_label" not in df.columns:
                raise ValueError("El CSV debe contener columnas 'path' y 'roughness_label'.")
            for _, r in df.iterrows():
                p = Path(str(r["path"])).resolve()
                try:
                    val = int(r["roughness_label"])
                except Exception:
                    val = -1
                self.rough[str(p).lower()] = val
        self.rough_k = rough_k

        # recorre carpetas de clase
        if not self.root.exists():
            raise FileNotFoundError(f"No existe split '{split}' en {self.root.parent}. Revísalo.")

        for cname in CLASS_NAMES:
            cdir = self.root / cname
            if not cdir.exists():
                continue
            for p in cdir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    type_id = CLASS_TO_ID[cname]
                    rough_id = self.rough.get(str(p.resolve()).lower(), -1)
                    # Sanitiza etiquetas fuera de rango a -1
                    if rough_id is not None and rough_id >= self.rough_k:
                        rough_id = -1
                    self.items.append((p, type_id, rough_id))

        # TFMs
        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, type_id, rough_id = self.items[i]
        img = pil_loader(p)
        return self.tf(img), type_id, rough_id

class DualHeadNet(nn.Module):
    """
    Backbone (ResNet50/EfficientNet-B0) + dos cabezas:
      - head_type: 4 clases (liso/grava/tierra/obstaculo)
      - head_rough: k clases de rugosidad (p.ej. 3)
    """
    def __init__(self, rough_k=3, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            dim = m.fc.in_features
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # hasta pool (B,2048,1,1)
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = m.classifier[-1].in_features
            self.backbone = nn.Sequential(
                m.features, nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError("backbone no soportado")

        self.head_type = nn.Linear(dim, 4)
        self.head_rough = nn.Linear(dim, rough_k)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        logits_type = self.head_type(feats)
        logits_rough = self.head_rough(feats)
        return logits_type, logits_rough

def accuracy(logits, targets, ignore_index=None):
    with torch.no_grad():
        preds = logits.argmax(1)
        if ignore_index is not None:
            mask = targets != ignore_index
            if mask.sum() == 0:
                return float('nan')
            correct = (preds[mask] == targets[mask]).sum().item()
            return correct / mask.sum().item()
        else:
            correct = (preds == targets).sum().item()
            return correct / targets.numel()

def masked_ce(logits, targets, ignore_index=-1, label_smoothing=0.0):
    """
    CrossEntropy sobre subconjunto con etiqueta válida (targets != ignore_index).
    Devuelve (loss_tensor, num_valid).
    """
    mask = (targets != ignore_index)
    if mask.sum() == 0:
        return torch.zeros((), device=logits.device), 0
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return ce(logits[mask], targets[mask]), int(mask.sum().item())

def train_one_epoch(model, loader, opt, scaler, device, label_smoothing_t=0.05, label_smoothing_r=0.05, clip_grad=1.0):
    model.train()
    m_loss = m_acc_t = m_acc_r = 0.0
    n_batches = 0

    for imgs, y_type, y_rough in loader:
        imgs = imgs.to(device, non_blocking=True)
        # Asegurar tensores
        y_type = torch.as_tensor(y_type, device=device, dtype=torch.long)
        y_rough = torch.as_tensor(y_rough, device=device, dtype=torch.long)

        opt.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=(device == "cuda")):
            logits_t, logits_r = model(imgs)

            # CE tipo (tiene siempre etiqueta)
            ce_type = nn.CrossEntropyLoss(label_smoothing=label_smoothing_t)
            Lt = ce_type(logits_t, y_type)

            # CE rugosidad (enmascarada -1)
            Lr, valid_r = masked_ce(logits_r, y_rough, ignore_index=-1, label_smoothing=label_smoothing_r)

            loss = Lt + Lr  # pesos iguales por defecto

        # Anti-NaN/Inf: salta batch
        if torch.isnan(loss) or torch.isinf(loss):
            opt.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        # Desescalar y recortar gradiente
        scaler.unscale_(opt)
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        scaler.step(opt)
        scaler.update()

        # Métricas
        m_loss += loss.item()
        m_acc_t += accuracy(logits_t, y_type)
        m_acc_r += accuracy(logits_r, y_rough, ignore_index=-1)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, float('nan')
    return m_loss/n_batches, m_acc_t/n_batches, m_acc_r/n_batches

@torch.no_grad()
def evaluate(model, loader, device, label_smoothing_t=0.0, label_smoothing_r=0.0):
    model.eval()
    m_loss = m_acc_t = m_acc_r = 0.0
    n_batches = 0

    for imgs, y_type, y_rough in loader:
        imgs = imgs.to(device, non_blocking=True)
        y_type = torch.as_tensor(y_type, device=device, dtype=torch.long)
        y_rough = torch.as_tensor(y_rough, device=device, dtype=torch.long)

        logits_t, logits_r = model(imgs)

        ce_type = nn.CrossEntropyLoss(label_smoothing=label_smoothing_t)
        Lt = ce_type(logits_t, y_type)
        Lr, _ = masked_ce(logits_r, y_rough, ignore_index=-1, label_smoothing=label_smoothing_r)
        loss = Lt + Lr

        m_loss += loss.item()
        m_acc_t += accuracy(logits_t, y_type)
        m_acc_r += accuracy(logits_r, y_rough, ignore_index=-1)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, float('nan')
    return m_loss/n_batches, m_acc_t/n_batches, m_acc_r/n_batches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Carpeta data con train/val(/test)")
    ap.add_argument("--offroad_csv", required=True, help="CSV: offroad_multitask_labels.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rough_k", type=int, default=3)
    ap.add_argument("--backbone", choices=["resnet50","efficientnet_b0"], default="resnet50")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="A:\\Varios\\MobileNet\\cnn-terreno\\models\\multitask.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Datasets / Loaders
    train_ds = MultiTaskFolder(args.data, "train", args.offroad_csv, args.img_size, args.rough_k)
    val_ds   = MultiTaskFolder(args.data, "val",   args.offroad_csv, args.img_size, args.rough_k)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device=="cuda"),
        persistent_workers=(args.workers>0)
    )
    val_dl   = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device=="cuda"),
        persistent_workers=(args.workers>0)
    )

    # Modelo
    model = DualHeadNet(rough_k=args.rough_k, backbone=args.backbone, pretrained=True).to(device)

    # Optimizador
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # AMP scaler moderno
    scaler = GradScaler('cuda', enabled=(device=="cuda"))

    best_score = -1
    history = []
    for epoch in range(1, args.epochs+1):
        t0 = time.time()

        tr_loss, tr_acc_t, tr_acc_r = train_one_epoch(
            model, train_dl, opt, scaler, device,
            label_smoothing_t=0.05, label_smoothing_r=0.05, clip_grad=1.0
        )
        va_loss, va_acc_t, va_acc_r = evaluate(
            model, val_dl, device,
            label_smoothing_t=0.0, label_smoothing_r=0.0
        )
        dt = time.time() - t0

        # score de validación (promedio simple de las dos tareas, ignorando NaN en rough)
        if math.isnan(va_acc_r):
            score = va_acc_t
        else:
            score = (va_acc_t + va_acc_r) / 2.0

        print(f"[{epoch:03d}] {dt:.1f}s  "
              f"train: loss={tr_loss:.4f} type_acc={tr_acc_t:.3f} rough_acc={tr_acc_r:.3f} | "
              f"val: loss={va_loss:.4f} type_acc={va_acc_t:.3f} rough_acc={va_acc_r:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "train_type_acc": float(tr_acc_t),
            "train_rough_acc": float(tr_acc_r) if not math.isnan(tr_acc_r) else None,
            "val_loss": float(va_loss),
            "val_type_acc": float(va_acc_t),
            "val_rough_acc": float(va_acc_r) if not math.isnan(va_acc_r) else None,
            "seconds": float(dt)
        })

        if score > best_score:
            best_score = score
            Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_score": best_score,
                "class_names": CLASS_NAMES
            }, args.out)
            print(f"✔ Guardado mejor modelo en: {args.out}  (score={best_score:.3f})")

    # guarda historial
    hist_json = Path(args.out).with_suffix(".history.json")
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[DONE] Historial: {hist_json}")

if __name__ == "__main__":
    main()



