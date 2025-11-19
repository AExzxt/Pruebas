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
from actuators.servo_controller import PCA9685, ServoMapper

_log = logging.getLogger(__name__)


class App(tk.Tk):
    def __init__(
        self,
        imu: MPU6050,
        mag,
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
        self._camera_running = False
        self._camera_paused = False
        self._cam_thread: threading.Thread | None = None
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

        quick = ttk.Frame(afrm)
        quick.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(quick, text="Izquierda", command=lambda: self._apply_dir_preset(-35)).grid(row=0, column=0, padx=2)
        ttk.Button(quick, text="Centro", command=lambda: self._apply_dir_preset(0)).grid(row=0, column=1, padx=2)
        ttk.Button(quick, text="Derecha", command=lambda: self._apply_dir_preset(35)).grid(row=0, column=2, padx=2)
        ttk.Button(quick, text="Suave", command=lambda: self._apply_susp_preset(10)).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(quick, text="Medio", command=lambda: self._apply_susp_preset(50)).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(quick, text="Duro", command=lambda: self._apply_susp_preset(90)).grid(row=1, column=2, padx=2, pady=2)

        self.lbl_act = ttk.Label(afrm, text="Último PWM: --")
        self.lbl_act.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Cámara + controles
        if self.use_camera:
            cam_frame = ttk.LabelFrame(self, text="Cámara")
            cam_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
            btn_row = ttk.Frame(cam_frame)
            btn_row.grid(row=0, column=0, sticky="w")
            ttk.Button(btn_row, text="Iniciar", command=self.start_camera).grid(row=0, column=0, padx=2, pady=2)
            ttk.Button(btn_row, text="Pausar", command=self.pause_camera).grid(row=0, column=1, padx=2, pady=2)
            ttk.Button(btn_row, text="Reanudar", command=self.resume_camera).grid(row=0, column=2, padx=2, pady=2)
            ttk.Button(btn_row, text="Detener", command=self.stop_camera).grid(row=0, column=3, padx=2, pady=2)
            self.canvas = tk.Label(cam_frame)
            self.canvas.grid(row=1, column=0, padx=5, pady=5)
        else:
            self.canvas = None

        # Logs
        log_frame = ttk.LabelFrame(self, text="Registro de acciones")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, height=5, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def _start_threads(self) -> None:
        self.imu.subscribe(self._on_imu)
        self.imu.start()
        if self.mag is not None:
            self._sensor_thread = threading.Thread(target=self._mag_loop, daemon=True)
            self._sensor_thread.start()
        self.after(200, self._refresh_gui)
        if self.use_camera:
            self.start_camera(auto=True)

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
        self._log_action(f"Dirección ajustada a {angle:.1f}° ({pulse:.0f} us)")

    def _on_susp_changed(self, val: str) -> None:
        pct = float(val)
        pulse = self.mapper.percent_to_us(pct)
        self.pca.set_servo_us(1, pulse)  # canal 1
        self.lbl_act.config(text=f"Susp: {pct:.0f}% -> {pulse:.0f} us")
        self._log_action(f"Suspensión ajustada a {pct:.0f}% ({pulse:.0f} us)")

    def _apply_dir_preset(self, angle: float) -> None:
        self.slider_dir.set(angle)
        self._on_dir_changed(str(angle))

    def _apply_susp_preset(self, pct: float) -> None:
        self.slider_susp.set(pct)
        self._on_susp_changed(str(pct))

    # -------- cámara
    def start_camera(self, auto: bool = False) -> None:
        if not self.use_camera:
            return
        if self._camera_running:
            self._camera_paused = False
            if not auto:
                self._log_action("Cámara reanudada")
            return
        self._camera_running = True
        self._camera_paused = False
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()
        if not auto:
            self._log_action("Cámara iniciada")

    def pause_camera(self) -> None:
        if self._camera_running:
            self._camera_paused = True
            self._log_action("Cámara en pausa")

    def resume_camera(self) -> None:
        if self._camera_running and self._camera_paused:
            self._camera_paused = False
            self._log_action("Cámara reanudada")

    def stop_camera(self) -> None:
        if self._camera_running:
            self._camera_running = False
            self._camera_paused = False
            if self._cam_thread and self._cam_thread.is_alive():
                self._cam_thread.join(timeout=1.0)
            if self.cam:
                self.cam.release()
                self.cam = None
            self._log_action("Cámara detenida")

    def _cam_loop(self) -> None:
        assert cv2 is not None
        cam = cv2.VideoCapture(self.cam_index)
        if not cam.isOpened():
            _log.error("No se pudo abrir la cámara %s", self.cam_index)
            self._camera_running = False
            return
        self.cam = cam
        while self._camera_running:
            if self._camera_paused:
                time.sleep(0.1)
                continue
            ret, frame = cam.read()
            if not ret:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.01)
        cam.release()
        self.cam = None

    def _refresh_gui(self) -> None:
        imu = getattr(self, "_last_imu", None)
        heading = getattr(self, "_last_heading", None) if self.mag is not None else None
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

    def _log_action(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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
            if self.use_camera:
                self.stop_camera()
                import cv2 as _cv2  # type: ignore
                _cv2.destroyAllWindows()
        finally:
            self.destroy()
