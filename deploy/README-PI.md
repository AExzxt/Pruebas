Objetivo
========

Este folder contiene los artefactos de despliegue mínimos para Raspberry Pi:
- `multitask_two_loaders.pt` (clasificación tipo+rugosidad, mejor validación)
- `yolo_pothole_best.pt` (detección de baches, run estable)

Requisitos en Raspberry Pi (64-bit recomendado)
-----------------------------------------------

1) Dependencias del sistema

    sudo apt update
    sudo apt install -y python3-pip python3-opencv git

    # Para cámara CSI (opcional)
    sudo apt install -y gstreamer1.0-tools gstreamer1.0-libcamera

2) Clonar repo

    git clone <TU_REPO_GITHUB_URL>.git
    cd MobileNet/cnn-terreno

3) Python packages

    pip install --upgrade pip
    pip install numpy ultralytics
    # PyTorch CPU para aarch64 (Raspberry Pi 4 / 5 64-bit)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

4) Prueba rápida con webcam USB

    python scripts/pi_runtime.py --src 0 --show --record --out pi_runs

5) Cámara CSI (ejemplo 720p@30) usando GStreamer

    python scripts/pi_runtime.py \
      --gst "libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink" \
      --show --record --out pi_runs

Salidas por ejecución
---------------------
- `pi_runs/<timestamp>/checkpoints/`: copias de los pesos usados
- `pi_runs/<timestamp>/run_config.json`: config de la sesión
- `pi_runs/<timestamp>/results.jsonl`: 1 JSON por frame (terreno, rugosidad, bboxes)
- `pi_runs/<timestamp>/results.csv`: resumen plano por frame
- `pi_runs/<timestamp>/output.mp4`: video con overlay (si `--record`)

Servicio systemd (opcional)
---------------------------

1) Crear unidad `/etc/systemd/system/rasp_terrain.service`

    [Unit]
    Description=Terreno+Rugosidad+YOLO Service
    After=network.target

    [Service]
    WorkingDirectory=/home/pi/MobileNet
    ExecStart=/usr/bin/python3 cnn-terreno/scripts/pi_runtime.py --src 0 --out /home/pi/terrain_runs --record
    Restart=always
    User=pi

    [Install]
    WantedBy=multi-user.target

2) Activar

    sudo systemctl daemon-reload
    sudo systemctl enable --now rasp_terrain.service

Notas
-----
- Para mayor FPS, usa `--skip 2` y/o `--width 640 --height 480`.
- Si prefieres ONNX para la cabeza multitarea, primero genera `models/terreno.onnx` con `scripts/exportar_terreno_onnx.py` (en tu PC) y luego integramos onnxruntime en `pi_runtime.py`.

