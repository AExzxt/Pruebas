from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Dict, Optional

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:  # pragma: no cover
    from smbus import SMBus  # type: ignore

_log = logging.getLogger(__name__)


class MPU6050:
    """Lectura básica de acelerómetro/giroscopio con MPU6050."""

    # Registros clave
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    INT_STATUS = 0x3A
    ACCEL_XOUT_H = 0x3B

    ACCEL_SCALE = {"±2g": 16384.0, "±4g": 8192.0, "±8g": 4096.0, "±16g": 2048.0}
    GYRO_SCALE = {"±250dps": 131.0, "±500dps": 65.5, "±1000dps": 32.8, "±2000dps": 16.4}

    def __init__(
        self,
        bus: int = 1,
        addr: int = 0x68,
        accel_range: str = "±4g",
        gyro_range: str = "±500dps",
        dlpf: int = 42,
        rate_hz: int = 100,
    ) -> None:
        self.bus_id = bus
        self.addr = addr
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.dlpf = dlpf
        self.rate_hz = rate_hz
        self._bus: Optional[SMBus] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[Dict[str, float]], None]] = []
        self._last_ts = time.monotonic()
        self._accel_scale = 9.80665 / self.ACCEL_SCALE[self.accel_range]
        self._gyro_scale = 1.0 / self.GYRO_SCALE[self.gyro_range]

    def initialize(self) -> None:
        self._bus = SMBus(self.bus_id)
        self._write(self.PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        smplrt = max(int(1000 / max(self.rate_hz, 1)) - 1, 0)
        dlpf_cfg = 3 if self.dlpf == 42 else 0
        accel_bits = list(self.ACCEL_SCALE.keys()).index(self.accel_range) << 3
        gyro_bits = list(self.GYRO_SCALE.keys()).index(self.gyro_range) << 3
        self._write(self.SMPLRT_DIV, smplrt)
        self._write(self.CONFIG, dlpf_cfg)
        self._write(self.ACCEL_CONFIG, accel_bits)
        self._write(self.GYRO_CONFIG, gyro_bits)
        _log.info("MPU6050 inicializado en I2C bus %s addr 0x%02X", self.bus_id, self.addr)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def subscribe(self, cb: Callable[[Dict[str, float]], None]) -> None:
        self._callbacks.append(cb)

    def read(self) -> Dict[str, float]:
        data = self._read_block(self.ACCEL_XOUT_H, 14)
        ax, ay, az = (self._combine(data[i], data[i + 1]) for i in (0, 2, 4))
        gx, gy, gz = (self._combine(data[i], data[i + 1]) for i in (8, 10, 12))
        ax *= self._accel_scale
        ay *= self._accel_scale
        az *= self._accel_scale
        gx *= self._gyro_scale
        gy *= self._gyro_scale
        gz *= self._gyro_scale
        amag = math.sqrt(ax * ax + ay * ay + az * az)
        now = time.monotonic()
        dt = now - self._last_ts
        self._last_ts = now
        pitch = math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az)))
        roll = math.degrees(math.atan2(ay, az))
        return {
            "t_unix": time.time(),
            "t_mono_ns": time.monotonic_ns(),
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "amag": amag,
            "pitch": pitch,
            "roll": roll,
            "dt": dt,
        }

    def _loop(self) -> None:
        period = 1.0 / max(self.rate_hz, 1)
        while not self._stop.is_set():
            try:
                sample = self.read()
                for cb in list(self._callbacks):
                    try:
                        cb(sample)
                    except Exception as exc:
                        _log.error("Callback IMU falló: %s", exc)
            except Exception as exc:
                _log.error("Error en lectura IMU: %s", exc)
            time.sleep(period)

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
