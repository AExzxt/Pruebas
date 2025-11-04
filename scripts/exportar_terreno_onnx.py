# exportar_terreno_onnx.py
import torch
import torch.nn as nn
import os

# ====== AJUSTA ESTO SI LO SABES ======
MODEL_PT = r"A:\Varios\MobileNet\cnn-terreno\models\multitask_two_loaders.pt"
ONNX_OUT = r"A:\Varios\MobileNet\cnn-terreno\models\terreno.onnx"
INPUT_SIZE = 224          # <-- si tu modelo usa otro tamaño (ej. 299), cámbialo
NUM_CLASSES = 4           # <-- pon el número real de clases de terreno
# =====================================

class DummyHead(nn.Module):
    # Útil si el modelo cargado no tiene forward estándar. No se usa si tu modelo ya está completo.
    def __init__(self, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1280, n_classes)  # suposición estilo MobileNetV3
    def forward(self, x):
        if x.ndim == 4:
            x = self.pool(x)
            x = x.flatten(1)
        return self.fc(x)

def try_load_full_model(path):
    try:
        m = torch.load(path, map_location="cpu")
        if isinstance(m, nn.Module):
            return m, "full"
        # a veces viene envuelto en dict {'model': model}
        if isinstance(m, dict) and "model" in m and isinstance(m["model"], nn.Module):
            return m["model"], "full_in_dict"
    except Exception as e:
        print(f"[info] Carga directa falló: {e}")
    return None, None

def main():
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)

    model, mode = try_load_full_model(MODEL_PT)
    if model is None:
        # probablemente sea un state_dict
        print("[aviso] Parece que el .pt contiene un state_dict. Para exportar a ONNX necesitas instanciar la clase del modelo.")
        print("       Opciones:")
        print("         A) Exporta ONNX desde tu script de entrenamiento (donde sí tienes la clase).")
        print("         B) O compárteme el nombre/clase del modelo y su código para agregarlos aquí.")
        return

    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # salida dummy para validar forma (no forzamos nombre de output)
    with torch.no_grad():
        out = model(dummy)
        if isinstance(out, (list, tuple)):
            out0 = out[0]
        else:
            out0 = out
        print("[info] forma de salida:", 
              out0.shape if hasattr(out0, "shape") else type(out0))

    # Exportar ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"[ok] ONNX exportado en: {ONNX_OUT}")

if __name__ == "__main__":
    main()

import torch
m = torch.load(r"A:\Varios\MobileNet\cnn-terreno\models\multitask_two_loaders.pt", map_location="cpu")
sd = m if isinstance(m, dict) and all(isinstance(k, str) for k in m.keys()) else m.get("state_dict", m)
print("N keys:", len(sd))
for k, v in list(sd.items())[:30]:  # muestra primeras 30
    print(k, tuple(v.shape))




