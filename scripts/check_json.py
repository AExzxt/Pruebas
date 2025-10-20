import json

with open("dataset/baches.v5i.coco/train/_annotations.coco.json", encoding="utf-8") as f:
    data = json.load(f)

print("Ejemplo de file_name:", data["images"][0]["file_name"])
print("Total de imágenes registradas:", len(data["images"]))

