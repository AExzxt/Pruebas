import sys, time, cv2, numpy as np
from tensorflow import keras

CLASSES = ["liso", "grava", "tierra", "obstaculo"]
IMG_SIZE = (128,128)

model = keras.models.load_model("models/final_mnv3.keras")

def predict(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"No se pudo abrir {path}")
    img = cv2.resize(img, IMG_SIZE)
    x = img[..., ::-1] / 255.0  # BGR->RGB
    x = np.expand_dims(x, 0).astype(np.float32)
    t0 = time.time()
    proba = model.predict(x, verbose=0)[0]
    dt = (time.time() - t0)*1000
    cls_id = int(np.argmax(proba))
    return CLASSES[cls_id], float(proba[cls_id]), dt

if __name__ == "__main__":
    path = sys.argv[1]
    cls, conf, ms = predict(path)
    print(f"Predicción: {cls} (conf {conf:.2f}) en {ms:.1f} ms")

