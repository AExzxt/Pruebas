# IMU MPU-6050 para Raspberry Pi

## Habilitar I²C

1. Ejecuta:
   ```bash
   sudo raspi-config
   # Interfacing Options > I2C > Enable
   sudo reboot
   ```
2. Verifica conexión:
   ```bash
   sudo apt install -y i2c-tools
   i2cdetect -y 1
   # Debe aparecer 68
   ```

## Instalación

```bash
sudo apt install python3-pip
pip3 install -r requirements-imu.txt
```

## Uso CLI

- Calibrar y guardar bias:
  ```bash
  python3 imu_recorder.py --calibrate
  ```
- Registrar CSV a 100 Hz:
  ```bash
  python3 imu_recorder.py --rate 100 --csv /home/pi/logs/imu.csv
  ```
- Publicar por MQTT:
  ```bash
  python3 imu_recorder.py --mqtt tcp://localhost:1883 --topic imu/data
  ```

## Ejemplo de integración

```python
from imu.mpu6050 import MPU6050
imu = MPU6050()
imu.initialize()
imu.load_calibration()
imu.start()
last_imu = None
def on_imu(sample):
    global last_imu
    last_imu = sample
imu.subscribe(on_imu)
# En tu pipeline de visión:
# frame = get_frame()
# frame['imu'] = last_imu
```

## Diagnóstico rápido si la Pi no detecta el MPU-6050

1) Comprueba con i2cdetect:

```bash
sudo apt install -y i2c-tools
i2cdetect -y 1
```

Debes ver la dirección `68` (o `69` si AD0=VCC). Si no la ves, revisa cableado y VCC/AD0.

2) Usa el script de diagnóstico incluido:

```bash
python3 cnn-terreno/imu/check_mpu.py --bus 1 --addr 0x68
```

El script intentará inicializar el bus y leer el registro WHO_AM_I (0x75). Si falla, la salida contendrá mensajes útiles para depuración (revisa permisos, i2c habilitado y cableado).

3) Mensajes comunes:
- "No se encontró smbus2 ni smbus": instala `pip3 install smbus2` y habilita I2C.
- "Error leyendo WHO_AM_I": comprueba que AD0 está a GND para 0x68 o a VCC para 0x69 y que i2cdetect lo muestra.


## Validación
- INT_STATUS debe cambiar al mover el sensor.
- pitch/roll ~0° con el módulo plano.
- CSV debe tener encabezado y N filas a la tasa configurada.
- Simula desconexión I²C: verifica reintentos y logs.

## Parámetros avanzados
- --bus, --addr, --accel, --gyro, --dlpf, --rate, --alpha
- Ver ayuda: `python3 imu_recorder.py --help`
