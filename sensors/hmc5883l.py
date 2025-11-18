from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:  # pragma: no cover
    from smbus import SMBus  # type: ignore

_log = logging.getLogger(__name__)


@dataclass
class Calibration:
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0


class HMC5883L:
    """Lectura simple del magnetómetro HMC5883L/HW-127 y heading."""

    ADDRESS = 0x1E
    REG_CONFIG_A = 0x00
    REG_CONFIG_B = 0x01
    REG_MODE = 0x02
    REG_DATA_X_MSB = 0x03

    def __init__(self, bus: int = 1, addr: int = 0x1E) -> None:
        self.bus_id = bus
        self.addr = addr
        self._bus: Optional[SMBus] = None
        self.calibration = Calibration()

    def initialize(self) -> None:
        self._bus = SMBus(self.bus_id)
        # Config: 8-sample avg, 15 Hz, normal measurement
        self._write(self.REG_CONFIG_A, 0x70)
        # Gain = 1.3 Ga, configuración estándar
        self._write(self.REG_CONFIG_B, 0x20)
        # Mode = continuous
        self._write(self.REG_MODE, 0x00)
        time.sleep(0.1)
        _log.info("HMC5883L inicializado en I2C bus %s addr 0x%02X", self.bus_id, self.addr)

    def read_raw(self) -> Tuple[int, int, int]:
        data = self._read_block(self.REG_DATA_X_MSB, 6)
        x = self._combine(data[0], data[1])
        z = self._combine(data[2], data[3])
        y = self._combine(data[4], data[5])
        return x, y, z

    def read_calibrated(self) -> Tuple[float, float, float]:
        x, y, z = self.read_raw()
        return (
            x - self.calibration.offset_x,
            y - self.calibration.offset_y,
            z - self.calibration.offset_z,
        )

    def heading_deg(self) -> float:
        bx, by, _ = self.read_calibrated()
        heading = math.degrees(math.atan2(by, bx))
        if heading < 0:
            heading += 360.0
        return heading

    def calibrate_hard_iron(self, duration_s: float = 15.0) -> Calibration:
        """Calibración simple: mover en 8 agitando durante duration_s."""
        _log.info("Calibrando HMC5883L (hard iron). Mueve el sensor en 8 durante %.1f s", duration_s)
        min_x = min_y = min_z = 1e9
        max_x = max_y = max_z = -1e9
        start = time.time()
        while time.time() - start < duration_s:
            x, y, z = self.read_raw()
            min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
            max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
            time.sleep(0.05)
        self.calibration = Calibration(
            offset_x=(max_x + min_x) / 2.0,
            offset_y=(max_y + min_y) / 2.0,
            offset_z=(max_z + min_z) / 2.0,
        )
        _log.info("Calibración hard-iron: %s", self.calibration)
        return self.calibration

    def _write(self, reg: int, val: int) -> None:
        assert self._bus is not None
        self._bus.write_byte_data(self.addr, reg, val)

    def _read_block(self, reg: int, length: int) -> list[int]:
        assert self._bus is not None
        return self._bus.read_i2c_block_data(self.addr, reg, length)

    @staticmethod
    def _combine(msb: int, lsb: int) -> int:
        val = (msb << 8) | lsb
        if val & 0x8000:
            val -= 0x10000
        return val
