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
2. Ejecutar GUI:
```bash
python3 main.py
```
*(Si más adelante agregas un magnetómetro compatible, ajusta `main.py` y la GUI lo mostrarán.)*

## Qué hace
- Inicializa I²C, MPU6050 (100 Hz), HMC5883L (heading), PCA9685 a 50 Hz.
- Inicia hilos para leer IMU y magnetómetro.
- GUI con Tkinter:
  - Muestra accel, gyro, pitch/roll, heading.
  - Sliders para servo dirección (-35 a +35°) y amortiguador (0–100%).
  - Video en tiempo real si hay cámara y OpenCV disponible, con redimensionado ligero.

## Notas
- Manejo de errores I²C: las clases levantan excepciones si el bus falla; revisa cableado y `sudo i2cdetect -y 1`.
- Pulsos para servos: rango 500–2500 µs, ajustable en `ServoMapper`.
- La GUI usa hilos simples y `after` para refrescar cada ~100 ms, evitando congelamientos.
