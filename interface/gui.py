from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import random

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
        self.mode_var = tk.StringVar(value="manual")
        self._manual_widgets: list[tk.Widget] = []
        self._pending_auto_command: tuple[float, float] | None = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._create_widgets()
        self._start_threads()

    def _set_widget_state(self, widget: tk.Widget, enabled: bool) -> None:
        try:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])
        except tk.TclError:
            widget.configure(state="normal" if enabled else "disabled")

    def _set_controls_state(self, enabled: bool) -> None:
        for widget in self._manual_widgets:
            self._set_widget_state(widget, enabled)

    def _on_mode_changed(self) -> None:
        manual = self.mode_var.get() == "manual"
        self._set_controls_state(manual)
        if manual:
            self.auto_status.config(text="Modo actual: Manual")
        else:
            self.auto_status.config(text="Modo actual: Automático (esperando modelo)")
            self._apply_pending_auto_command()

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

        # Modo de control
        mode_frame = ttk.LabelFrame(self, text="Modo de control")
        mode_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Radiobutton(
            mode_frame, text="Manual", variable=self.mode_var, value="manual", command=self._on_mode_changed
        ).grid(row=0, column=0, padx=4, pady=2)
        ttk.Radiobutton(
            mode_frame, text="Automático (modelo)", variable=self.mode_var, value="auto", command=self._on_mode_changed
        ).grid(row=0, column=1, padx=4, pady=2)
        self.auto_status = ttk.Label(mode_frame, text="Modo actual: Manual")
        self.auto_status.grid(row=1, column=0, columnspan=2, sticky="w", padx=4)

        # Actuadores
        afrm = ttk.LabelFrame(self, text="Actuadores (PCA9685)")
        afrm.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Label(afrm, text="Dirección (-35 a 35°)").grid(row=0, column=0, sticky="w")
        self.slider_dir = ttk.Scale(afrm, from_=-35, to=35, command=self._on_dir_changed)
        self.slider_dir.grid(row=0, column=1, sticky="ew")
        afrm.columnconfigure(1, weight=1)

        ttk.Label(afrm, text="Amortiguador (0-100%)").grid(row=1, column=0, sticky="w")
        self.slider_susp = ttk.Scale(afrm, from_=0, to=100, command=self._on_susp_changed)
        self.slider_susp.grid(row=1, column=1, sticky="ew")

        quick = ttk.Frame(afrm)
        quick.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.btn_left = ttk.Button(quick, text="Izquierda", command=lambda: self._apply_dir_preset(-35))
        self.btn_center = ttk.Button(quick, text="Centro", command=lambda: self._apply_dir_preset(0))
        self.btn_right = ttk.Button(quick, text="Derecha", command=lambda: self._apply_dir_preset(35))
        self.btn_soft = ttk.Button(quick, text="Suave", command=lambda: self._apply_susp_preset(10))
        self.btn_mid = ttk.Button(quick, text="Medio", command=lambda: self._apply_susp_preset(50))
        self.btn_hard = ttk.Button(quick, text="Duro", command=lambda: self._apply_susp_preset(90))
        self.btn_left.grid(row=0, column=0, padx=2)
        self.btn_center.grid(row=0, column=1, padx=2)
        self.btn_right.grid(row=0, column=2, padx=2)
        self.btn_soft.grid(row=1, column=0, padx=2, pady=2)
        self.btn_mid.grid(row=1, column=1, padx=2, pady=2)
        self.btn_hard.grid(row=1, column=2, padx=2, pady=2)
        self.btn_demo_auto = ttk.Button(quick, text="Demo auto", command=self._demo_auto_command)
        self.btn_demo_auto.grid(row=2, column=0, columnspan=3, pady=4)

        self.lbl_act = ttk.Label(afrm, text="Último PWM: --")
        self.lbl_act.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Cámara + controles
        if self.use_camera:
            cam_frame = ttk.LabelFrame(self, text="Cámara")
            cam_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
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
        log_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, height=5, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self._manual_widgets = [
            self.slider_dir,
            self.slider_susp,
            self.btn_left,
            self.btn_center,
            self.btn_right,
            self.btn_soft,
            self.btn_mid,
            self.btn_hard,
        ]
        self._on_mode_changed()

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
        if self.mode_var.get() != "manual":
            return
        self._set_direction(float(val), source="Manual")

    def _on_susp_changed(self, val: str) -> None:
        if self.mode_var.get() != "manual":
            return
        self._set_suspension(float(val), source="Manual")

    def _apply_dir_preset(self, angle: float) -> None:
        self.slider_dir.set(angle)
        self._set_direction(angle, source="Manual")

    def _apply_susp_preset(self, pct: float) -> None:
        self.slider_susp.set(pct)
        self._set_suspension(pct, source="Manual")

    def _demo_auto_command(self) -> None:
        steer = random.uniform(-35.0, 35.0)
        susp = random.uniform(0.0, 100.0)
        self.mode_var.set("auto")
        self._on_mode_changed()
        self.submit_auto_command(steer, susp)
        self._log_action("Demo auto: enviando comandos generados aleatoriamente")

    def _set_direction(self, angle: float, source: str = "Manual") -> None:
        clamped = max(-35.0, min(35.0, angle))
        pulse = self.mapper.angle_to_us(clamped, (-35, 35))
        self.pca.set_servo_us(0, pulse)
        self.lbl_act.config(text=f"Dir: {clamped:.1f}° -> {pulse:.0f} us")
        self._log_action(f"[{source}] Dirección {clamped:.1f}° ({pulse:.0f} us)")

    def _set_suspension(self, pct: float, source: str = "Manual") -> None:
        clamped = max(0.0, min(100.0, pct))
        pulse = self.mapper.percent_to_us(clamped)
        self.pca.set_servo_us(1, pulse)
        self.lbl_act.config(text=f"Susp: {clamped:.0f}% -> {pulse:.0f} us")
        self._log_action(f"[{source}] Suspensión {clamped:.0f}% ({pulse:.0f} us)")

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

    def submit_auto_command(self, steer_deg: float, suspension_pct: float) -> None:
        """Permite que otro módulo envíe comandos automáticos desde el modelo."""
        self._pending_auto_command = (steer_deg, suspension_pct)
        self.after(0, self._apply_pending_auto_command)

    def _apply_pending_auto_command(self) -> None:
        if not self._pending_auto_command:
            return
        angle, pct = self._pending_auto_command
        if self.mode_var.get() != "auto":
            return
        self._set_direction(angle, source="Automático")
        self._set_suspension(pct, source="Automático")
        self.auto_status.config(text=f"Modo actual: Automático (Dir {angle:.1f}°, Susp {pct:.0f}%)")

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
