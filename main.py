from __future__ import annotations

import logging

from sensors.mpu6050 import MPU6050
from sensors.hmc5883l import HMC5883L
from actuators.servo_controller import PCA9685
from interface.gui import App


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Inicializar I2C y dispositivos
    imu = MPU6050(rate_hz=100)
    imu.initialize()

    mag = HMC5883L()
    mag.initialize()
    # Opcional: calibra hard-iron moviendo el sensor
    # mag.calibrate_hard_iron(15)

    pca = PCA9685(freq_hz=50.0)
    pca.initialize()

    # Lanzar GUI
    app = App(imu=imu, mag=mag, pca=pca, cam_index=0, use_camera=True)
    app.mainloop()


if __name__ == "__main__":
    main()
