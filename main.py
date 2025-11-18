from __future__ import annotations

import logging

from sensors.mpu6050 import MPU6050
from actuators.servo_controller import PCA9685
from interface.gui import App


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Inicializar I2C y dispositivos
    imu = MPU6050(rate_hz=100)
    imu.initialize()

    pca = PCA9685(freq_hz=50.0)
    pca.initialize()

    # Lanzar GUI
    app = App(imu=imu, mag=None, pca=pca, cam_index=0, use_camera=True)
    app.mainloop()


if __name__ == "__main__":
    main()
