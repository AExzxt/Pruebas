#!/usr/bin/env python3
"""Script de diagnóstico rápido para MPU-6050.
Lee WHO_AM_I y toma 10 muestras crudas mostrando valores.
"""
import time
import sys
import argparse
from imu.mpu6050 import MPU6050


def main():
    p = argparse.ArgumentParser(description="Check MPU-6050 presence and quick read")
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--addr', type=lambda x: int(x,0), default=0x68)
    args = p.parse_args()

    imu = MPU6050(bus=args.bus, addr=args.addr)
    try:
        imu.initialize()
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar IMU: {e}")
        sys.exit(2)
    try:
        who = imu.check_device()
        print(f"WHO_AM_I: 0x{who:02X}")
    except Exception as e:
        print(f"[ERROR] WHO_AM_I failed: {e}")
        sys.exit(3)

    print("Leyendo 10 muestras crudas:")
    for i in range(10):
        try:
            raw = imu._read_raw()
            ax,ay,az = raw['accel']
            gx,gy,gz = raw['gyro']
            print(f"{i}: ax={ax}, ay={ay}, az={az}, gx={gx}, gy={gy}, gz={gz}")
        except Exception as e:
            print(f"Lectura fallida en intento {i}: {e}")
        time.sleep(0.05)
    print("Hecho.")

if __name__ == '__main__':
    main()
