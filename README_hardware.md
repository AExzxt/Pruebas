# Integración IMU + Magnetómetro + PCA9685 + GUI (Raspberry Pi)

## Estructura
```
main.py
sensors/
  mpu6050.py
  hmc5883l.py
actuators/
  servo_controller.py
interface/
  gui.py
```

## Requisitos
- Raspberry Pi OS, I²C habilitado (`sudo raspi-config` → Interface Options → I2C).
- Python 3.9+.
- Dependencias: `smbus2`, `numpy`, `Pillow`, `opencv-python` (o `opencv-python-headless`), `tkinter` (preinstalado en Pi OS), opcional PyQt5 si prefieres.

Instalación rápida:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk libatlas-base-dev
pip install smbus2 numpy Pillow opencv-python
```

## Uso
1. Conectar:
   - MPU6050 → I2C (0x68).
   - PCA9685 → I2C (0x40 por defecto).
   - Servos MG996R en canal 0 (dirección) y canal 1 (amortiguador).
2. Activar entorno y lanzar GUI:
```bash
cd ~/Pruebas
source /home/aldrin/mobilenet-env/bin/activate
python3 main.py
```
*(Si más adelante agregas un magnetómetro compatible, ajusta `main.py` y la GUI lo mostrará.)*

## Qué hace
- Inicializa I²C, MPU6050 (100 Hz) y PCA9685 a 50 Hz (el magnetómetro es opcional; pasa `mag=None` en `main.py`).
- Inicia hilos para leer IMU y actualizar la GUI sin bloquearla.
- GUI con Tkinter:
  - Lecturas de aceleración, giroscopio y pitch/roll.
  - Sliders y botones rápidos (Izquierda/Centro/Derecha, Suave/Medio/Duro) para los servos.
  - Controles de cámara (Iniciar/Pausar/Reanudar/Detener) y video en tiempo real usando OpenCV.
  - Registro de acciones enviado al PCA9685 y eventos de la cámara.

## Notas
- Manejo de errores I²C: las clases levantan excepciones si el bus falla; revisa cableado y `sudo i2cdetect -y 1`.
- Pulsos para servos: rango 500–2500 µs, ajustable en `ServoMapper`.
- La GUI usa hilos simples y `after` para refrescar cada ~100 ms, evitando congelamientos.
