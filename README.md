# Plataforma de percepción de terreno y baches

Repositorio para clasificación de tipo de terreno, detección de baches y control de hardware (IMU, magnetómetro, servos), pensado para correr en PC y Raspberry Pi.

## Estructura del repositorio
- `scripts/`: utilidades para preparación, entrenamiento e inferencia.
  - `train_mnv3.py`: entrena MobileNetV3Small con data augmentation para cuatro clases de terreno y guarda pesos en `models/`.
  - `infer_image.py`: ejecuta inferencia sobre una imagen con el modelo entrenado y muestra clase, confianza y tiempo.
  - `pi_runtime.py`: runtime en Raspberry Pi que combina clasificación multitarea y YOLO para baches, graba resultados y video (ver `deploy/README-PI.md`).
- `deploy/`: recursos de despliegue en Pi.
  - Pesos pre-entrenados (`multitask_two_loaders.pt`, `yolo_pothole_best.pt`).
  - Guía paso a paso para instalar dependencias y lanzar el runtime (`README-PI.md`).
- `interface/`: GUI en Tkinter para controlar cámara, modos Manual/Automático y presets de actuadores; administra hilos para captura y lectura de sensores.
- `sensors/`: drivers sencillos para sensores I²C.
  - `mpu6050.py`: lectura básica de acelerómetro/giroscopio con callbacks a suscriptores.
  - `hmc5883l.py`: magnetómetro con calibración y cálculo de heading.
- `imu/`: implementación avanzada de IMU con calibración persistente, filtro complementario y publicación a suscriptores.
- `actuators/`: controladores para actuadores.
  - `servo_controller.py`: inicialización del PCA9685 y utilidades para mapear ángulos/porcentajes a PWM.
- `main.py`: punto de entrada que integra IMU, magnetómetro, PCA9685 y lanza la GUI.
- `imu_recorder.py`: CLI para calibrar, registrar CSV a tasas configurables y publicar lecturas de IMU vía MQTT.
- `eval/`: reportes de desempeño de modelos (matrices de confusión y métricas).
- `README_hardware.md`: guía de conexión y uso de hardware con la GUI.
- `README_imu.md`: instrucciones para habilitar I²C, calibrar y diagnosticar la MPU6050.
- `requirements.txt` y `requirements-imu.txt`: dependencias para visión/ML y para el stack de IMU, respectivamente.

## Requisitos rápidos
- Vision/ML: `pip install -r requirements.txt` (en PC con GPU o Pi con soporte ARM para PyTorch/Ultralytics).
- Herramientas de IMU: `pip install -r requirements-imu.txt` (en Raspberry Pi habilitar I²C previamente).

## Flujos principales
- **Entrenar clasificación de terreno (PC):**
  ```bash
  python scripts/train_mnv3.py
  ```
  Guarda `models/final_mnv3.keras` con augmentations y fine-tuning de MobileNetV3.

- **Inferir tipo de terreno en una imagen:**
  ```bash
  python scripts/infer_image.py ruta/a/imagen.jpg
  ```
  Devuelve clase, confianza y tiempo de inferencia usando `models/final_mnv3.keras`.

- **Ejecución en Raspberry Pi (clasificación + YOLO):**
  Sigue `deploy/README-PI.md` para instalar dependencias, clonar el repo y lanzar:
  ```bash
  python scripts/pi_runtime.py --src 0 --show --record --out pi_runs
  ```
  Cada ejecución guarda configuraciones, resultados por frame y video con overlay en `pi_runs/<timestamp>/`.

- **Calibrar/registrar IMU desde CLI:**
  ```bash
  python3 imu_recorder.py --calibrate
  python3 imu_recorder.py --rate 100 --csv /home/pi/logs/imu.csv
  ```
  Publica por MQTT con `--mqtt tcp://localhost:1883 --topic imu/data`.

## Documentación complementaria
- `README_hardware.md`: wiring, requisitos y uso de la GUI con IMU + PCA9685.
- `README_imu.md`: guía para habilitar I²C, calibrar y diagnosticar la MPU6050.
- `deploy/README-PI.md`: instrucciones detalladas de despliegue en Raspberry Pi y servicio systemd opcional.
