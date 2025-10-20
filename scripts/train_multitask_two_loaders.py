import argparse, os, time, json, random
from pathlib import Path
from collections import deque

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------------------------
# Config de clases de SUPERFICIE
# ---------------------------
CLASS_NAMES = ["liso", "grava", "tierra", "obstaculo"]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def pil_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")

# ---------------------------
# Dataset de SUPERFICIE (data/{train,val})
# ---------------------------
class SurfaceFolder(Dataset):
    def __init__(self, root, split, img_size=224):
        self.root = Path(root)/split
        self.items = []
        for cname in CLASS_NAMES:
            cdir = self.root/cname
            if not cdir.exists(): 
                continue
            for p in cdir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.items.append( (p, CLASS_TO_ID[cname]) )

        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(pil_rgb(p)), y

# ---------------------------
# Dataset de RUGOSIDAD (Off-Road CSV con rutas absolutas)
# ---------------------------
class OffroadRoughnessDS(Dataset):
    def __init__(self, csv_path, split="train", img_size=224, split_ratio=0.85, seed=42):
        df = pd.read_csv(csv_path)
        # columnas esperadas: path, roughness_label, (opcional) train(bool) del paper StreetSurfaceVis
        if "path" not in df.columns or "roughness_label" not in df.columns:
            raise ValueError("CSV debe contener columnas 'path' y 'roughness_label'")

        # Partición simple por índice si no hay columna 'train'
        if "train" in df.columns:
            if split == "train":
                df = df[df["train"]==True]
            else:
                df = df[df["train"]==False]
        else:
            # split fijo 85/15 por orden aleatorio pero determinista
            rnd = random.Random(seed)
            idx = list(range(len(df)))
            rnd.shuffle(idx)
            cut = int(len(idx)*split_ratio)
            sel = set(idx[:cut]) if split=="train" else set(idx[cut:])
            df = df.iloc[sorted(list(sel))]

        # filtra rutas existentes
        keep = []
        for _,r in df.iterrows():
            p = Path(r["path"])
            if p.exists():
                keep.append( (p, int(r["roughness_label"])) )
        self.items = keep

        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(pil_rgb(p)), y

# ---------------------------
# Red con dos cabezas
# ---------------------------
class DualHeadNet(nn.Module):
    def __init__(self, rough_k=3, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            dim = m.fc.in_features
            self.backbone = nn.Sequential(*(list(m.children())[:-1])) # (B,2048,1,1)
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = m.classifier[-1].in_features
            self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
        else:
            raise ValueError("backbone no soportado")

        self.head_type  = nn.Linear(dim, len(CLASS_NAMES))
        self.head_rough = nn.Linear(dim, rough_k)

    def forward(self, x):
        f = self.backbone(x)
        f = torch.flatten(f, 1)
        return self.head_type(f), self.head_rough(f)

def acc(logits, targets):
    with torch.no_grad():
        pred = logits.argmax(1)
        return (pred==targets).float().mean().item()

# ---------------------------
# Entrenamiento con dos dataloaders
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--offroad_csv", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rough_k", type=int, default=3)
    ap.add_argument("--backbone", choices=["resnet50","efficientnet_b0"], default="resnet50")
    ap.add_argument("--out", default="models/multitask_two_loaders.pt")
    ap.add_argument("--workers", type=int, default=0)  # Windows: 0 suele ser estable
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Datasets/dataloaders
    ds_type_tr = SurfaceFolder(args.data, "train", args.img_size)
    ds_type_va = SurfaceFolder(args.data, "val",   args.img_size)
    dl_type_tr = DataLoader(ds_type_tr, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    dl_type_va = DataLoader(ds_type_va, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    ds_rough_tr = OffroadRoughnessDS(args.offroad_csv, split="train", img_size=args.img_size)
    ds_rough_va = OffroadRoughnessDS(args.offroad_csv, split="val",   img_size=args.img_size)
    dl_rough_tr = DataLoader(ds_rough_tr, batch_size=args.batch, shuffle=True,
                             num_workers=args.workers, pin_memory=True)
    dl_rough_va = DataLoader(ds_rough_va, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    print(f"[INFO] batches -> type(train/val)={len(dl_type_tr)}/{len(dl_type_va)} | "
          f"rough(train/val)={len(dl_rough_tr)}/{len(dl_rough_va)}")

    # Modelo, pérdidas, opt
    model = DualHeadNet(rough_k=args.rough_k, backbone=args.backbone, pretrained=True).to(device)
    loss_type  = nn.CrossEntropyLoss()
    loss_rough = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" if device=="cuda" else "cpu")

    best_score = -1
    hist = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        # iteradores
        it_type  = iter(dl_type_tr)
        it_rough = iter(dl_rough_tr)
        steps = max(len(dl_type_tr), len(dl_rough_tr))

        run_loss = run_acc_t = run_acc_r = 0.0
        for _ in range(steps):
            # --- batch de TIPO
            try:
                xb_t, yb_t = next(it_type)
                xb_t = xb_t.to(device, non_blocking=True)
                yb_t = yb_t.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu"):
                    log_t, _ = model(xb_t)
                    Lt = loss_type(log_t, yb_t)
                scaler.scale(Lt).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += Lt.item()
                run_acc_t += acc(log_t, yb_t)
            except StopIteration:
                pass

            # --- batch de RUGOSIDAD
            try:
                xb_r, yb_r = next(it_rough)
                xb_r = xb_r.to(device, non_blocking=True)
                yb_r = yb_r.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu"):
                    _, log_r = model(xb_r)
                    Lr = loss_rough(log_r, yb_r)
                scaler.scale(Lr).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += Lr.item()
                run_acc_r += acc(log_r, yb_r)
            except StopIteration:
                pass

        # normaliza métricas por pasos efectivos
        denom_t = max(1, len(dl_type_tr))
        denom_r = max(1, len(dl_rough_tr))
        tr_loss = run_loss / (denom_t + denom_r)
        tr_acc_t = run_acc_t / denom_t
        tr_acc_r = run_acc_r / denom_r

        # --------- VALIDACIÓN ---------
        model.eval()
        def eval_loop(dloader, head="type"):
            tot_loss = tot_acc = n = 0
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    log_t, log_r = model(xb)
                    if head=="type":
                        L = loss_type(log_t, yb)
                        A = acc(log_t, yb)
                    else:
                        L = loss_rough(log_r, yb)
                        A = acc(log_r, yb)
                    tot_loss += L.item()
                    tot_acc  += A
                    n += 1
            return tot_loss/max(1,n), tot_acc/max(1,n)

        va_loss_t, va_acc_t = eval_loop(dl_type_va, "type")
        va_loss_r, va_acc_r = eval_loop(dl_rough_va, "rough")
        va_loss = (va_loss_t + va_loss_r)/2.0
        score = (va_acc_t + va_acc_r)/2.0

        dt = time.time() - t0
        print(f"[{epoch:03d}] {dt:.1f}s  "
              f"train: loss={tr_loss:.4f} type_acc={tr_acc_t:.3f} rough_acc={tr_acc_r:.3f} | "
              f"val: loss={va_loss:.4f} type_acc={va_acc_t:.3f} rough_acc={va_acc_r:.3f}")

        hist.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_type_acc": tr_acc_t,
            "train_rough_acc": tr_acc_r,
            "val_loss": va_loss,
            "val_type_acc": va_acc_t,
            "val_rough_acc": va_acc_r,
            "seconds": dt
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

    with open(Path(args.out).with_suffix(".history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print("[DONE] Entrenamiento finalizado.")

if __name__ == "__main__":
    main()


