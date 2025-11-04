import argparse
import os
import torch
import torch.nn as nn
from torchvision import models


class DualHeadNet(nn.Module):
    """Backbone (ResNet50 o EfficientNet-B0) + dos cabezas:
    - head_type: num_types clases (tipo de terreno)
    - head_rough: rough_k clases (rugosidad)
    """
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
            raise ValueError("Backbone no soportado: " + str(backbone))

        self.head_type = nn.Linear(dim, num_types)
        self.head_rough = nn.Linear(dim, rough_k)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return self.head_type(feats), self.head_rough(feats)


def load_checkpoint(path):
    # Carga segura si está disponible (torch>=2.5)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    # Formatos soportados:
    # 1) dict con 'model' (state_dict), 'args', 'class_names'
    # 2) state_dict plano
    # 3) objeto con 'state_dict'
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
    return state_dict, args, class_names


def infer_head_dims(state_dict, class_names):
    if class_names is not None:
        num_types = len(class_names)
    else:
        wt = state_dict.get("head_type.weight")
        num_types = wt.shape[0] if wt is not None else 4
    wr = state_dict.get("head_rough.weight")
    rough_k = wr.shape[0] if wr is not None else 3
    return int(num_types), int(rough_k)


def build_model_for_state(state_dict, args, class_names):
    backbone = args.get("backbone") if isinstance(args, dict) else None
    candidates = [backbone] if backbone else ["resnet50", "efficientnet_b0"]

    num_types, rough_k = infer_head_dims(state_dict, class_names)

    last_err = None
    for bb in candidates:
        try:
            model = DualHeadNet(num_types=num_types, rough_k=rough_k, backbone=bb)
            model.load_state_dict(state_dict, strict=True)
            return model, bb, num_types, rough_k
        except Exception as e:
            last_err = e

    for bb in candidates:
        try:
            model = DualHeadNet(num_types=num_types, rough_k=rough_k, backbone=bb)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[aviso] Cargado con strict=False. missing={len(missing)} unexpected={len(unexpected)}")
            return model, bb, num_types, rough_k
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No se pudo reconstruir el modelo desde state_dict. Último error: {last_err}")


def export_onnx(model, onnx_out, img_size=224, opset=12):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits_t, logits_r = self.m(x)
            return logits_t, logits_r

    wrapper = Wrapper(model)

    with torch.no_grad():
        out_t, out_r = wrapper(dummy)
        print("[info] salida TYPE:", tuple(out_t.shape))
        print("[info] salida ROUGH:", tuple(out_r.shape))

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_out,
        input_names=["input"],
        output_names=["logits_type", "logits_rough"],
        opset_version=opset,
        dynamic_axes={
            "input": {0: "batch"},
            "logits_type": {0: "batch"},
            "logits_rough": {0: "batch"}
        }
    )
    print(f"[ok] ONNX exportado en: {onnx_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("cnn-terreno", "models", "multitask_two_loaders.pt"))
    ap.add_argument("--out", default=os.path.join("cnn-terreno", "models", "terreno.onnx"))
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    state_dict, ck_args, class_names = load_checkpoint(args.ckpt)
    model, backbone, num_types, rough_k = build_model_for_state(state_dict, ck_args, class_names)

    img_size = args.img_size
    if img_size is None:
        if isinstance(ck_args, dict) and ck_args.get("img_size"):
            img_size = int(ck_args["img_size"])
        else:
            img_size = 224

    print(f"[info] backbone={backbone} num_types={num_types} rough_k={rough_k} img_size={img_size}")
    export_onnx(model, args.out, img_size=img_size, opset=args.opset)


if __name__ == "__main__":
    main()

