import os, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (128,128)
BATCH = 32
EPOCHS_HEAD = 15
EPOCHS_FT   = 10
CLASSES = ["liso", "grava", "tierra", "obstaculo"]

train_dir = "data/train"
val_dir   = "data/val"
os.makedirs("models", exist_ok=True)

train_ds = keras.utils.image_dataset_from_directory(
    train_dir, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
norm = layers.Rescaling(1./255)
data_augmentation = keras.Sequential([
    layers.RandomBrightness(factor=0.2),
    layers.RandomContrast(factor=0.2),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomZoom(height_factor=(-0.05,0.0), width_factor=(-0.05,0.0)),
    layers.GaussianNoise(0.01),
])

train_ds = train_ds.map(lambda x,y: (norm(data_augmentation(x, training=True)), y)).prefetch(AUTOTUNE)
val_ds   = val_ds.map(lambda x,y: (norm(x), y)).prefetch(AUTOTUNE)

base = keras.applications.MobileNetV3Small(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE,3))
x = inputs
x = base(x, training=False)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

cb = [
    keras.callbacks.ModelCheckpoint("models/best_head.keras", monitor="val_accuracy",
                                    save_best_only=True, mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cb)

# Fine-tuning: descongela últimas capas
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

cb_ft = [
    keras.callbacks.ModelCheckpoint("models/best_ft.keras", monitor="val_accuracy",
                                    save_best_only=True, mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cb_ft)

model.save("models/final_mnv3.keras")
print("✔ Modelo guardado en models/final_mnv3.keras")

