import time
import json
import threading
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, List
import numpy as np
try:
    from smbus2 import SMBus
except Exception:
    try:
        # fallback a smbus (más antiguo)
        from smbus import SMBus  # type: ignore
    except Exception:
        SMBus = None  # type: ignore
from .filters import ComplementaryAHRS
from .utils import clamp

# Direcciones de registros MPU-6050
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
INT_STATUS = 0x3A
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

ACCEL_SENS = {
    '±2g': 16384.0,
    '±4g': 8192.0,
    '±8g': 4096.0,
    '±16g': 2048.0,
}
GYRO_SENS = {
    '±250dps': 131.0,
    '±500dps': 65.5,
    '±1000dps': 32.8,
    '±2000dps': 16.4,
}

class MPU6050:
    """
    Clase para leer y calibrar el sensor MPU-6050 por I2C.
    """
    def __init__(self, bus: int = 1, addr: int = 0x68,
                 accel_range: str = '±4g', gyro_range: str = '±500dps', dlpf: int = 42):
        self.bus_num = bus
        self.addr = addr
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.dlpf = dlpf
        self.smbus: Optional[SMBus] = None
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        self.ahrs = ComplementaryAHRS(alpha=0.02, dt=0.01)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscribers: List[Callable[[Dict], None]] = []
        self.logger = logging.getLogger("MPU6050")
        self.calib_path = Path.home() / ".imu_calibration.json"

    def initialize(self):
        """Inicializa el sensor: sale de sleep, configura sample rate, DLPF y rangos."""
        try:
            if SMBus is None:
                raise RuntimeError("No se encontró smbus2 ni smbus. Instala 'smbus2' y habilita I2C en la Raspberry Pi.")
            self.smbus = SMBus(self.bus_num)
            # Salir de sleep
            self.smbus.write_byte_data(self.addr, PWR_MGMT_1, 0)
            # Sample rate: 1kHz/(1+div)
            self.smbus.write_byte_data(self.addr, SMPLRT_DIV, 9)  # 100Hz
            # DLPF
            self.smbus.write_byte_data(self.addr, CONFIG, self._dlpf_cfg())
            # Rango acelerómetro
            self.smbus.write_byte_data(self.addr, ACCEL_CONFIG, self._accel_cfg())
            # Rango giroscopio
            self.smbus.write_byte_data(self.addr, GYRO_CONFIG, self._gyro_cfg())
            self.logger.info("MPU6050 inicializado en I2C bus %d addr 0x%02X", self.bus_num, self.addr)
        except Exception as e:
            # Mensaje útil si falla I2C
            msg = f"Error inicializando MPU6050: {e}. Comprueba que I2C esté habilitado (sudo raspi-config -> Interfacing Options -> I2C) y que la dirección sea correcta (AD0 a GND=0x68)."
            self.logger.error(msg)
            raise RuntimeError(msg)

    def _dlpf_cfg(self) -> int:
        """Mapea la frecuencia DLPF (Hz o índice) a la configuración DLPF_CFG (bits 2:0).
        Valores aproximados por datasheet:
         0 -> 260Hz
         1 -> 184Hz
         2 -> 94Hz
         3 -> 44Hz
         4 -> 21Hz
         5 -> 10Hz
         6 -> 5Hz
        """
        # Si el usuario pasó un valor típico en Hz, mapeamos al índice
        try:
            hz = int(self.dlpf)
        except Exception:
            hz = 42
        if hz >= 260:
            return 0
        if hz >= 184:
            return 1
        if hz >= 94:
            return 2
        if hz >= 44:
            return 3
        if hz >= 21:
            return 4
        if hz >= 10:
            return 5
        return 6

    def _accel_cfg(self) -> int:
        # ACCEL_CONFIG: bits 3-4
        ranges = {'±2g': 0, '±4g': 1, '±8g': 2, '±16g': 3}
        return ranges.get(self.accel_range, 1) << 3

    def _gyro_cfg(self) -> int:
        # GYRO_CONFIG: bits 3-4
        ranges = {'±250dps': 0, '±500dps': 1, '±1000dps': 2, '±2000dps': 3}
        return ranges.get(self.gyro_range, 1) << 3

    def calibrate(self, n_samples: int = 500):
        """Calibra el sensor en reposo, guarda bias en ~/.imu_calibration.json."""
        self.logger.info(f"Calibrando... {n_samples} muestras")
        acc = []
        gyro = []
        for _ in range(n_samples):
            raw = self._read_raw()
            acc.append(raw['accel'])
            gyro.append(raw['gyro'])
            time.sleep(0.005)
        self.bias_accel = np.mean(acc, axis=0)
        self.bias_gyro = np.mean(gyro, axis=0)
        calib = {
            'bias_accel': self.bias_accel.tolist(),
            'bias_gyro': self.bias_gyro.tolist(),
            'accel_range': self.accel_range,
            'gyro_range': self.gyro_range,
            'dlpf': self.dlpf,
        }
        with open(self.calib_path, 'w') as f:
            json.dump(calib, f)
        self.logger.info(f"Calibración guardada en {self.calib_path}")

    def load_calibration(self):
        """Carga bias desde ~/.imu_calibration.json si existe."""
        if self.calib_path.exists():
            with open(self.calib_path) as f:
                calib = json.load(f)
            self.bias_accel = np.array(calib.get('bias_accel', [0,0,0]))
            self.bias_gyro = np.array(calib.get('bias_gyro', [0,0,0]))
            self.logger.info(f"Calibración cargada de {self.calib_path}")

    def _read_raw(self) -> Dict:
        """Lee registros crudos accel/gyro."""
        # Implementamos reintentos exponenciales para lecturas I2C ocasionalmente fallidas
        attempts = 0
        backoff = 0.01
        while attempts < 5:
            try:
                if self.smbus is None:
                    raise RuntimeError("SMBus no disponible")
                data = self.smbus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14)
                ax = self._twos(data[0], data[1])
                ay = self._twos(data[2], data[3])
                az = self._twos(data[4], data[5])
                gx = self._twos(data[8], data[9])
                gy = self._twos(data[10], data[11])
                gz = self._twos(data[12], data[13])
                return {
                    'accel': np.array([ax, ay, az], dtype=np.float32),
                    'gyro': np.array([gx, gy, gz], dtype=np.float32)
                }
            except Exception as e:
                attempts += 1
                self.logger.debug(f"Lectura I2C fallida (intento {attempts}): {e}")
                time.sleep(backoff)
                backoff *= 2
        # Si llegamos aquí, lanzar excepción con mensaje útil
        msg = f"No se pudo leer registros del MPU6050 en {self.addr}. Comprueba conexión I2C y que el dispositivo existe."
        self.logger.error(msg)
        raise RuntimeError(msg)

    def _twos(self, h: int, l: int) -> int:
        val = (h << 8) | l
        return val - 65536 if val > 32767 else val

    def read(self) -> Dict:
        """Lee, escala y filtra datos. Devuelve dict con claves estándar."""
        raw = self._read_raw()
        # Escalado
        accel_sens = ACCEL_SENS[self.accel_range]
        gyro_sens = GYRO_SENS[self.gyro_range]
        ax, ay, az = (raw['accel'] - self.bias_accel) / accel_sens * 9.80665
        gx, gy, gz = (raw['gyro'] - self.bias_gyro) / gyro_sens
        amag = float(np.sqrt(ax**2 + ay**2 + az**2))
        pitch, roll = self.ahrs.update(ax, ay, az, gx, gy)
        t_unix = time.time()
        t_mono_ns = time.monotonic_ns()
        return {
            't_unix': t_unix,
            't_mono_ns': t_mono_ns,
            'ax': float(ax), 'ay': float(ay), 'az': float(az),
            'gx': float(gx), 'gy': float(gy), 'gz': float(gz),
            'amag': amag,
            'pitch': pitch,
            'roll': roll,
        }

    def start(self, rate_hz: int = 100):
        """Inicia la adquisición en hilo dedicado."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(rate_hz,), daemon=True)
        self._thread.start()

    def stop(self):
        """Detiene la adquisición."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def subscribe(self, callback: Callable[[Dict], None]):
        """Suscribe un callback para cada muestra."""
        self._subscribers.append(callback)

    def _loop(self, rate_hz: int):
        dt = 1.0 / rate_hz
        next_time = time.monotonic()
        while self._running:
            try:
                sample = self.read()
                for cb in self._subscribers:
                    try:
                        cb(sample)
                    except Exception as ex_cb:
                        self.logger.debug(f"Callback IMU falló: {ex_cb}")
            except Exception as e:
                self.logger.error(f"Error en loop IMU: {e}")
            # Sleep con compensación para reducir jitter
            next_time += dt
            to_sleep = next_time - time.monotonic()
            if to_sleep > 0:
                time.sleep(to_sleep)
            else:
                # si estamos atrasados, avanzamos next_time para evitar drift
                next_time = time.monotonic()

    def check_device(self) -> int:
        """Lee el registro WHO_AM_I (0x75) para verificar la presencia del dispositivo.
        Retorna el valor leído (ej. 0x68) o lanza RuntimeError con mensaje útil."""
        if self.smbus is None:
            raise RuntimeError("SMBus no disponible. Instala smbus2 y habilita I2C.")
        try:
            who = self.smbus.read_byte_data(self.addr, 0x75)
            return who
        except Exception as e:
            msg = f"Error leyendo WHO_AM_I en 0x{self.addr:02X}: {e}. Verifica i2cdetect y cableado."
            self.logger.error(msg)
            raise RuntimeError(msg)
