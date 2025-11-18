from __future__ import annotations

import logging
import time
from typing import Tuple

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:  # pragma: no cover
    from smbus import SMBus  # type: ignore

_log = logging.getLogger(__name__)


class PCA9685:
    """Controlador PWM de 16 canales a 50 Hz para servos."""

    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus: int = 1, addr: int = 0x40, freq_hz: float = 50.0) -> None:
        self.bus_id = bus
        self.addr = addr
        self.freq_hz = freq_hz
        self._bus: SMBus | None = None

    def initialize(self) -> None:
        self._bus = SMBus(self.bus_id)
        self._write(self.MODE1, 0x00)  # salida normal
        time.sleep(0.005)
        prescale = int(round(25000000.0 / (4096.0 * self.freq_hz) - 1))
        self._write(self.MODE1, 0x10)  # dormir
        self._write(self.PRESCALE, prescale)
        self._write(self.MODE1, 0xA1)  # auto-increment + reiniciar
        _log.info("PCA9685 inicializado en 0x%02X a %.1f Hz (prescale=%d)", self.addr, self.freq_hz, prescale)

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        base = self.LED0_ON_L + 4 * channel
        self._write(base, on & 0xFF)
        self._write(base + 1, on >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    def set_servo_us(self, channel: int, pulse_us: float) -> None:
        # 4096 counts por periodo; periodo = 1/freq
        counts = int(pulse_us * self.freq_hz * 4096.0 / 1_000_000.0)
        counts = max(0, min(4095, counts))
        self.set_pwm(channel, 0, counts)

    def _write(self, reg: int, val: int) -> None:
        assert self._bus is not None
        self._bus.write_byte_data(self.addr, reg, val)


class ServoMapper:
    """Mapea ángulos o porcentajes a pulsos us para MG996R."""

    def __init__(self, min_us: float = 500.0, max_us: float = 2500.0) -> None:
        self.min_us = min_us
        self.max_us = max_us

    def angle_to_us(self, angle_deg: float, span_deg: Tuple[float, float]) -> float:
        """Mapea un ángulo dentro del span (ej. -35 a +35) a pulso us."""
        lo, hi = span_deg
        clamped = max(min(angle_deg, hi), lo)
        norm = (clamped - lo) / (hi - lo) if hi != lo else 0.5
        return self.min_us + norm * (self.max_us - self.min_us)

    def percent_to_us(self, pct: float) -> float:
        clamped = max(0.0, min(100.0, pct))
        norm = clamped / 100.0
        return self.min_us + norm * (self.max_us - self.min_us)
