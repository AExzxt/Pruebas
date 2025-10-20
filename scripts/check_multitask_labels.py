from pathlib import Path
import pandas as pd

DATA = Path(r"A:\Varios\MobileNet\cnn-terreno\data")
CSV  = Path(r"A:\Varios\MobileNet\cnn-terreno\data\offroad_multitask_labels.csv")

df = pd.read_csv(CSV)
if "path" not in df.columns or "roughness_label" not in df.columns:
    raise SystemExit("[ERROR] CSV debe tener columnas 'path' y 'roughness_label'.")

df["norm"] = df["path"].apply(lambda p: str(Path(str(p)).resolve()).lower())
label_map = dict(zip(df["norm"], df["roughness_label"]))

def count_split(split):
    root = DATA / split
    total = 0
    with_label = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp"):
            total += 1
            if str(p.resolve()).lower() in label_map:
                with_label += 1
    print(f"{split}: imgs={total}, con_rugosidad={with_label} ({with_label/total*100:.1f}% si total>0)")

for s in ["train","val","test"]:
    if (DATA/s).exists():
        count_split(s)


