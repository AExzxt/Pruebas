import numpy as np
from typing import Tuple

class ComplementaryAHRS:
    """
    Filtro complementario para estimar pitch/roll combinando acelerómetro y giroscopio.
    """
    def __init__(self, alpha: float = 0.02, dt: float = 0.01):
        self.alpha = alpha
        self.dt = dt
        self.pitch = 0.0
        self.roll = 0.0

    def update(self, ax: float, ay: float, az: float, gx: float, gy: float) -> Tuple[float, float]:
        # Acelerómetro: pitch/roll instantáneo
        pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi
        roll_acc = np.arctan2(ay, az) * 180 / np.pi
        # Giroscopio: integración
        self.pitch += gx * self.dt
        self.roll += gy * self.dt
        # Complementario
        self.pitch = self.alpha * pitch_acc + (1 - self.alpha) * self.pitch
        self.roll = self.alpha * roll_acc + (1 - self.alpha) * self.roll
        return self.pitch, self.roll
