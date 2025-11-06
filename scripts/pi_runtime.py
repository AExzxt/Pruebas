import argparse
import json
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


# -----------------------------
# Modelo multitarea (tipo + rugosidad)
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
            self.backbone = nn.Sequential(m.features, nn.m(1))
        else:
            raise ValueError("Backbone no soportado: " + str(backbone))

        self.head_type = nn.Linear(dim, num_types)
        self.head_rough = nn.Linear(dim, rough_k)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return self.head_type(feats), self.head_rough(feats)


def load_multitask(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        args = ckpt.get("args", {})
        class_names = ckpt.get("class_names")
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
        args = {}
        class_names = None
    else:
        state_dict = getattr(ckpt, "state_dict", ckpt)
        args = {}
        class_names = None

    # infer dimensiones de cabezas
    wt = state_dict.get("head_type.weight")
    wr = state_dict.get("head_rough.weight")
    num_types = int(wt.shape[0]) if wt is not None else 4
    rough_k = int(wr.shape[0]) if wr is not None else 3
    backbone = args.get("backbone") if isinstance(args, dict) else None
    candidates = [backbone] if backbone else ["resnet50", "efficientnet_b0"]

    last_err = None
    for bb in candidates:
        try:
            net = DualHeadNet(num_types=num_types, rough_k=rough_k, backbone=bb)
            net.load_state_dict(state_dict, strict=True)
            net.to(device)
            net.eval()
            if class_names is None:
                class_names = ["liso", "grava", "tierra", "obstaculo"][:num_types]
            return net, class_names, rough_k, bb
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No se pudo cargar el modelo multitarea: {last_err}")


def softmax_np(x):
    e = np.exp(x - np.max(x))
    s = e / (e.sum() + 1e-9)
    return s


def classify_terrain(frame: np.ndarray, model: nn.Module, device: str, img_size: int = 224,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> dict:
    """Devuelve tipo/roughness y probabilidades usando el modelo multitarea (PyTorch)."""
    x = cv2.resize(frame, (img_size, img_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits_t, logits_r = model(xt)
        lt = logits_t.cpu().numpy()[0]
        lr = logits_r.cpu().numpy()[0]
    pt = softmax_np(lt)
    pr = softmax_np(lr)
    return {
        "type_logits": lt.tolist(),
        "type_proba": pt.tolist(),
        "type_id": int(np.argmax(pt)),
        "type_conf": float(np.max(pt)),
        "rough_logits": lr.tolist(),
        "rough_proba": pr.tolist(),
        "rough_id": int(np.argmax(pr)),
        "rough_conf": float(np.max(pr)),
    }


# -----------------------------
# YOLO (Ultralytics)
# -----------------------------

def load_yolo(weights_path: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar ultralytics. Instala con: pip install ultralytics"
        ) from e
    return YOLO(weights_path)


def detect_potholes(frame: np.ndarray, yolo_model, conf_thres: float = 0.25):
    """Retorna lista de dicts con bbox y conf."""
    results = yolo_model.predict(source=frame, verbose=False, conf=conf_thres, imgsz=640)
    dets = []
    if not results:
        return dets
    r0 = results[0]
    if r0.boxes is None:
        return dets
    for b in r0.boxes:
        xyxy = b.xyxy.cpu().numpy()[0]
        conf = float(b.conf.cpu().numpy()[0])
        cls = int(b.cls.cpu().numpy()[0]) if b.cls is not None else 0
        dets.append({
            "x1": float(xyxy[0]), "y1": float(xyxy[1]),
            "x2": float(xyxy[2]), "y2": float(xyxy[3]),
            "conf": conf, "cls": cls
        })
    return dets


# -----------------------------
# Captura de video (USB / archivo / GStreamer)
# -----------------------------

def open_capture(src: str, width: int = None, height: int = None, fps: int = None, gst_pipeline: str = None):
    cap = None
    if gst_pipeline:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video")
    return cap


def draw_overlay(frame: np.ndarray, terr_label: str, terr_conf: float, rough_id: int, rough_conf: float, boxes: list):
    h, w = frame.shape[:2]
    # panel superior
    header = f"Terreno: {terr_label} ({terr_conf:.2f})  |  Rugosidad: r{rough_id} ({rough_conf:.2f})"
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    # cajas YOLO
    for d in boxes:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]) 
        conf = d.get("conf", 0.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(frame, f"pothole {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, cv2.LINE_AA)
    return frame


def ensure_run_dir(base: Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_checkpoint_files(run_dir: Path, multitask_ckpt: Path, yolo_weights: Path):
    try:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        if multitask_ckpt and Path(multitask_ckpt).exists():
            shutil.copy2(str(multitask_ckpt), str(run_dir / "checkpoints" / Path(multitask_ckpt).name))
        if yolo_weights and Path(yolo_weights).exists():
            shutil.copy2(str(yolo_weights), str(run_dir / "checkpoints" / Path(yolo_weights).name))
    except Exception as e:
        print(f"[warn] No se pudieron copiar checkpoints: {e}")


def append_jsonl(path: Path, obj: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict, header_written_cache: set):
    write_header = False
    if path not in header_written_cache and not path.exists():
        write_header = True
        header_written_cache.add(path)
    with open(path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(','.join(row.keys()) + "\n")
        values = [str(row[k]) for k in row.keys()]
        f.write(','.join(values) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi terreno + rugosidad + baches runtime")
    # modelos
    ap.add_argument("--multitask", default=str(Path("cnn-terreno/deploy/multitask_two_loaders.pt")))
    ap.add_argument("--yolo", default=str(Path("cnn-terreno/deploy/yolo_pothole_best.pt")))
    # captura
    ap.add_argument("--src", default="0", help="0, 1, ruta de archivo o cámara. Para CSI usar --gst")
    ap.add_argument("--gst", default=None, help="Pipeline GStreamer para cámara CSI si se desea")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    # ejecución
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25, help="confidencia mínima YOLO")
    ap.add_argument("--skip", type=int, default=0, help="número de frames a saltar entre inferencias")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--record", action="store_true")
    # salida
    ap.add_argument("--out", default=str(Path("cnn-terreno/pi_runs")))
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    out_base = Path(args.out)
    run_dir = ensure_run_dir(out_base)
    jsonl_path = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"
    header_written = set()

    # Cargar modelos
    print("[info] Cargando modelo multitarea...")
    net, class_names, rough_k, backbone = load_multitask(args.multitask, device)
    print(f"[info] Multitask cargado. backbone={backbone} clases={class_names} rough_k={rough_k}")

    print("[info] Cargando modelo YOLO...")
    yolo = load_yolo(args.yolo)
    print("[info] YOLO cargado.")

    # Checkpoint copia
    copy_checkpoint_files(run_dir, Path(args.multitask), Path(args.yolo))
    with open(run_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump({
            "multitask": args.multitask,
            "yolo": args.yolo,
            "device": device,
            "src": args.src,
            "gst": args.gst,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "conf": args.conf,
        }, f, indent=2, ensure_ascii=False)

    cap = open_capture(args.src, args.width, args.height, args.fps, gst_pipeline=args.gst)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(run_dir / 'output.mp4'), fourcc, float(args.fps), (args.width, args.height))

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.width and args.height and (frame.shape[1] != args.width or frame.shape[0] != args.height):
                frame = cv2.resize(frame, (args.width, args.height))

            do_infer = (args.skip <= 0) or (frame_idx % (args.skip + 1) == 0)
            terr = None
            dets = []
            if do_infer:
                terr = classify_terrain(frame, net, device)
                dets = detect_potholes(frame, yolo, conf_thres=args.conf)

                rec = {
                    "ts": time.time(),
                    "frame": frame_idx,
                    "terrain": {
                        "label": class_names[terr["type_id"]],
                        "conf": terr["type_conf"],
                        "id": terr["type_id"],
                        "proba": terr["type_proba"],
                    },
                    "roughness": {
                        "id": terr["rough_id"],
                        "conf": terr["rough_conf"],
                        "proba": terr["rough_proba"],
                    },
                    "potholes": dets,
                }
                append_jsonl(jsonl_path, rec)
                # CSV flat
                flat = {
                    "ts": rec["ts"],
                    "frame": frame_idx,
                    "terrain_label": rec["terrain"]["label"],
                    "terrain_conf": rec["terrain"]["conf"],
                    "rough_id": rec["roughness"]["id"],
                    "rough_conf": rec["roughness"]["conf"],
                    "potholes_n": len(dets),
                }
                append_csv(csv_path, flat, header_written)

            # overlay
            if terr is not None:
                frame = draw_overlay(frame, class_names[terr["type_id"]], terr["type_conf"], terr["rough_id"], terr["rough_conf"], dets)
            else:
                frame = draw_overlay(frame, "-", 0.0, -1, 0.0, dets)

            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("terreno+YOLO", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"[done] Frames: {frame_idx}  Tiempo: {dt:.1f}s  FPS: {frame_idx/max(1,dt):.2f}")


if __name__ == "__main__":
    main()
