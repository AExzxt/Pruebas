import cv2
import sys

# Ajusta esta ruta a una imagen original que tengas en raw/
IMG_PATH = r"raw\ejemplo.jpg"  # <-- CAMBIA a una imagen real
# ROI por defecto (x, y, w, h): AJUSTA a tu cámara
ROI_BOX = (220, 300, 128, 128)

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"No se pudo abrir {IMG_PATH}")
        sys.exit(1)
    x, y, w, h = ROI_BOX
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("ROI preview - presiona cualquier tecla", vis)
    cv2.waitKey(0)
    roi = img[y:y+h, x:x+w]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
