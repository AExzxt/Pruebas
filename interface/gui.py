from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from sensors.mpu6050 import MPU6050
from sensors.hmc5883l import HMC5883L
from actuators.servo_controller import PCA9685, ServoMapper

_log = logging.getLogger(__name__)


class App(tk.Tk):
    def __init__(
        self,
        imu: MPU6050,
        mag: HMC5883L,
        pca: PCA9685,
        cam_index: int = 0,
        use_camera: bool = True,
    ) -> None:
        super().__init__()
        self.title("IMU + Magnetómetro + PCA9685")
        self.imu = imu
        self.mag = mag
        self.pca = pca
        self.mapper = ServoMapper()
        self.use_camera = use_camera and cv2 is not None
        self.cam_index = cam_index
        self.cam = None
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._create_widgets()
        self._start_threads()

    def _create_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        # Sensores
        frm = ttk.LabelFrame(self, text="Sensores")
        frm.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.lbl_accel = ttk.Label(frm, text="Accel: --")
        self.lbl_gyro = ttk.Label(frm, text="Gyro: --")
        self.lbl_pitch = ttk.Label(frm, text="Pitch/Roll: --")
        self.lbl_heading = ttk.Label(frm, text="Heading: --")
        self.lbl_accel.grid(row=0, column=0, sticky="w")
        self.lbl_gyro.grid(row=1, column=0, sticky="w")
        self.lbl_pitch.grid(row=2, column=0, sticky="w")
        self.lbl_heading.grid(row=3, column=0, sticky="w")

        # Actuadores
        afrm = ttk.LabelFrame(self, text="Actuadores (PCA9685)")
        afrm.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Label(afrm, text="Dirección (-35 a 35°)").grid(row=0, column=0, sticky="w")
        self.slider_dir = ttk.Scale(afrm, from_=-35, to=35, command=self._on_dir_changed)
        self.slider_dir.grid(row=0, column=1, sticky="ew")
        afrm.columnconfigure(1, weight=1)

        ttk.Label(afrm, text="Amortiguador (0-100%)").grid(row=1, column=0, sticky="w")
        self.slider_susp = ttk.Scale(afrm, from_=0, to=100, command=self._on_susp_changed)
        self.slider_susp.grid(row=1, column=1, sticky="ew")

        self.lbl_act = ttk.Label(afrm, text="Último PWM: --")
        self.lbl_act.grid(row=2, column=0, columnspan=2, sticky="w")

        # Cámara
        if self.use_camera:
            self.canvas = tk.Label(self)
            self.canvas.grid(row=2, column=0, padx=5, pady=5)
        else:
            self.canvas = None

    def _start_threads(self) -> None:
        self.imu.subscribe(self._on_imu)
        self.imu.start()
        self._sensor_thread = threading.Thread(target=self._mag_loop, daemon=True)
        self._sensor_thread.start()
        if self.use_camera:
            self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
            self._cam_thread.start()
        self.after(200, self._refresh_gui)

    # -------- sensores
    def _on_imu(self, sample: dict) -> None:
        self._last_imu = sample  # type: ignore[attr-defined]

    def _mag_loop(self) -> None:
        while True:
            try:
                heading = self.mag.heading_deg()
                self._last_heading = heading  # type: ignore[attr-defined]
            except Exception as exc:
                _log.error("Error leyendo magnetómetro: %s", exc)
            time.sleep(0.2)

    # -------- actuadores
    def _on_dir_changed(self, val: str) -> None:
        angle = float(val)
        pulse = self.mapper.angle_to_us(angle, (-35, 35))
        self.pca.set_servo_us(0, pulse)  # canal 0
        self.lbl_act.config(text=f"Dir: {angle:.1f}° -> {pulse:.0f} us")

    def _on_susp_changed(self, val: str) -> None:
        pct = float(val)
        pulse = self.mapper.percent_to_us(pct)
        self.pca.set_servo_us(1, pulse)  # canal 1
        self.lbl_act.config(text=f"Susp: {pct:.0f}% -> {pulse:.0f} us")

    # -------- cámara
    def _cam_loop(self) -> None:
        assert cv2 is not None
        self.cam = cv2.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            _log.error("No se pudo abrir la cámara %s", self.cam_index)
            return
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.01)

    def _refresh_gui(self) -> None:
        imu = getattr(self, "_last_imu", None)
        heading = getattr(self, "_last_heading", None)
        if imu:
            self.lbl_accel.config(text=f"Accel: {imu['ax']:.2f}, {imu['ay']:.2f}, {imu['az']:.2f} m/s²")
            self.lbl_gyro.config(text=f"Gyro: {imu['gx']:.2f}, {imu['gy']:.2f}, {imu['gz']:.2f} °/s")
            self.lbl_pitch.config(text=f"Pitch/Roll: {imu['pitch']:.1f} / {imu['roll']:.1f} °")
        if heading is not None:
            self.lbl_heading.config(text=f"Heading: {heading:.1f} °")

        if self.canvas is not None and not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                self._draw_frame(frame)
            except queue.Empty:
                pass

        self.after(100, self._refresh_gui)

    def _draw_frame(self, frame) -> None:
        assert cv2 is not None
        # Convertir a Tk PhotoImage mediante PIL opcional; usar conversión sencilla
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        # Redimensiona moderado para GUI
        scale = 480 / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        # Convertir a PhotoImage
        try:
            from PIL import Image, ImageTk  # type: ignore
        except Exception:
            return
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.imgtk = imgtk  # evitar GC
        self.canvas.configure(image=imgtk)

    def on_close(self) -> None:
        try:
            self.imu.stop()
            if self.cam:
                self.cam.release()
            if self.use_camera:
                import cv2 as _cv2  # type: ignore
                _cv2.destroyAllWindows()
        finally:
            self.destroy()
