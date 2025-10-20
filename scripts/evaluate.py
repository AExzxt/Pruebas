import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (128,128)
BATCH = 32
CLASSES = ["liso", "grava", "tierra"] # "liso", "grava", "tierra", "obstaculo"

test_ds = image_dataset_from_directory(
    "data/test", labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

model = keras.models.load_model("models/final_mnv3.keras")
y_true = np.concatenate([y for _,y in test_ds], axis=0)
y_pred_proba = model.predict(test_ds)
y_pred = np.argmax(y_pred_proba, axis=1)

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
print(confusion_matrix(y_true, y_pred))

