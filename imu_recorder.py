#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from imu.mpu6050 import MPU6050
from imu.utils import setup_logging, handle_signals
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

def main():
    parser = argparse.ArgumentParser(description="Registro de datos MPU-6050 por I2C")
    parser.add_argument('--bus', type=int, default=1, help='Número de bus I2C (default: 1)')
    parser.add_argument('--addr', type=lambda x: int(x,0), default=0x68, help='Dirección I2C (default: 0x68)')
    parser.add_argument('--accel', choices=['±2g','±4g','±8g','±16g'], default='±4g')
    parser.add_argument('--gyro', choices=['±250dps','±500dps','±1000dps','±2000dps'], default='±500dps')
    parser.add_argument('--dlpf', type=int, default=42, help='DLPF Hz (default: 42)')
    parser.add_argument('--rate', type=int, default=100, help='Frecuencia de muestreo Hz (default: 100)')
    parser.add_argument('--calibrate', action='store_true', help='Ejecutar calibración y salir')
    parser.add_argument('--csv', type=str, help='Ruta para guardar CSV')
    parser.add_argument('--mqtt', type=str, help='Broker MQTT (ej: tcp://localhost:1883)')
    parser.add_argument('--topic', type=str, default='imu/data', help='Tópico MQTT')
    parser.add_argument('--alpha', type=float, default=0.02, help='Alpha filtro complementario')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("imu_recorder")

    imu = MPU6050(bus=args.bus, addr=args.addr, accel_range=args.accel, gyro_range=args.gyro, dlpf=args.dlpf)
    imu.ahrs.alpha = args.alpha
    try:
        imu.initialize()
    except Exception as e:
        logger.error("No se pudo inicializar el sensor. ¿I2C habilitado? Ejecuta 'sudo raspi-config' > Interfacing Options > I2C.")
        logger.error(f"Detalle: {e}")
        sys.exit(1)

    if args.calibrate:
        imu.calibrate()
        sys.exit(0)
    imu.load_calibration()

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open('w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ['t_unix','t_mono_ns','ax','ay','az','gx','gy','gz','amag','pitch','roll']
        csv_writer.writerow(header)
        logger.info(f"Escribiendo CSV en {csv_path}")

    mqtt_client = None
    if args.mqtt:
        if mqtt is None:
            logger.error("paho-mqtt no instalado. Instala con 'pip install paho-mqtt'")
            sys.exit(1)
        import re
        m = re.match(r'(tcp|mqtt)://([^:]+):(\d+)', args.mqtt)
        if not m:
            logger.error("Formato MQTT inválido. Usa tcp://host:puerto")
            sys.exit(1)
        host, port = m.group(2), int(m.group(3))
        mqtt_client = mqtt.Client()
        mqtt_client.connect(host, port)
        mqtt_client.loop_start()
        logger.info(f"Publicando en MQTT {args.mqtt} tópico {args.topic}")

    running = True
    def cleanup():
        nonlocal running
        running = False
        if csv_file:
            csv_file.close()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        logger.info("Cerrando imu_recorder.")
    handle_signals(cleanup, logger)

    def on_sample(sample):
        row = [sample[k] for k in ['t_unix','t_mono_ns','ax','ay','az','gx','gy','gz','amag','pitch','roll']]
        if csv_writer:
            csv_writer.writerow(row)
        if mqtt_client:
            mqtt_client.publish(args.topic, str(sample))

    imu.subscribe(on_sample)
    imu.start(rate_hz=args.rate)
    logger.info(f"Adquisición IMU iniciada a {args.rate} Hz. Ctrl+C para salir.")
    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
