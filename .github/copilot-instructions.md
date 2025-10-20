# Copilot Instructions for AI Agents

## Visión General del Proyecto
Este repositorio implementa una arquitectura de clasificación de imágenes para terrenos y detección de baches, usando redes neuronales convolucionales (CNN) y MobileNetV3. El flujo principal involucra la preparación de datos, entrenamiento, inferencia y evaluación de modelos.

## Estructura Clave
- `data/`: Contiene los conjuntos de datos organizados en carpetas por tipo de terreno (`grava`, `liso`, `obstaculo`, `tierra`) y por partición (`train`, `test`, `val`).
- `dataset/baches.v5i.coco/`: Datos en formato COCO para detección de baches, con anotaciones en JSON.
- `scripts/`: Scripts principales para todo el workflow:
  - `train_mnv3.py`: Entrenamiento de MobileNetV3 para clasificación.
  - `coco_to_classification.py`: Conversión de datos COCO a formato de clasificación.
  - `infer_image.py`: Inferencia sobre imágenes individuales.
  - `evaluate.py`: Evaluación de modelos entrenados.
  - `prepare_roi.py` y `preview_roi.py`: Preparación y visualización de regiones de interés.
- `models/`, `raw/`, `runs/`: Almacenan modelos entrenados, datos sin procesar y resultados de experimentos.

## Flujos de Trabajo Esenciales
- **Entrenamiento:** Ejecuta `python scripts/train_mnv3.py` con los argumentos necesarios para entrenar el modelo. Los datos deben estar preparados en `data/train`.
- **Conversión COCO:** Usa `python scripts/coco_to_classification.py` para transformar anotaciones COCO a formato de clasificación antes de entrenar.
- **Inferencia:** Ejecuta `python scripts/infer_image.py --image <ruta_imagen>` para predecir la clase de una imagen.
- **Evaluación:** Usa `python scripts/evaluate.py` para obtener métricas del modelo sobre el conjunto de validación o test.

## Convenciones y Patrones
- Los scripts asumen rutas relativas desde la raíz del proyecto.
- Los datos deben estar organizados estrictamente por clase y partición.
- Los modelos y resultados se guardan en subcarpetas bajo `models/` y `runs/`.
- El formato COCO se usa solo para detección; para clasificación, se requiere conversión previa.

## Dependencias e Integraciones
- Instala dependencias con `pip install -r requirements.txt` desde la raíz de `cnn-terreno`.
- El entorno virtual recomendado está en `.venv/`.
- El proyecto depende de librerías como TensorFlow, Keras, y posiblemente PyTorch para algunos scripts.

## Ejemplo de Comando de Entrenamiento
```bash
python scripts/train_mnv3.py --epochs 20 --batch_size 32 --data_dir data/train --output_dir models/
```

## Recomendaciones para Agentes
- Verifica la organización de los datos antes de entrenar o inferir.
- Usa los scripts en `scripts/` como entrypoints para tareas principales.
- Documenta cualquier convención nueva en este archivo.

---
¿Hay alguna sección que requiera mayor detalle o información específica sobre flujos, dependencias o convenciones?
