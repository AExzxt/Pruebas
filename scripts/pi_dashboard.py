import os
import signal
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


DEFAULT_ENV_ACTIVATE = "source ~/mobilenet-env/bin/activate && which python"
DEFAULT_RUNTIME_SCRIPT = "/home/aldrin/Pruebas/scripts/pi_runtime.py"
DEFAULT_ANALYZE_SCRIPT = "/home/aldrin/Pruebas/scripts/analyze_run.py"
DEFAULT_MULTITASK = "/home/aldrin/Pruebas/deploy/multitask_two_loaders.pt"
DEFAULT_YOLO = "/home/aldrin/Pruebas/deploy/yolo_pothole_best.pt"
DEFAULT_RUN_DIR = "/home/aldrin/cnn-terreno/pi_runs/latest_or_timestamp"


class CommandRunner:
    def __init__(self, output_callback, finish_callback):
        self.output_callback = output_callback
        self.finish_callback = finish_callback
        self.process = None
        self.thread = None
        self.queue = queue.Queue()
        self._paused = False

    def start(self, command: str):
        self.stop()
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        threading.Thread(target=self._notifier, daemon=True).start()

    def _reader(self):
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.queue.put(line.rstrip())
        self.process.wait()
        self.queue.put(None)

    def _notifier(self):
        while True:
            line = self.queue.get()
            if line is None:
                break
            self.output_callback(line)
        if self.process:
            code = self.process.returncode
        else:
            code = None
        self.finish_callback(code)

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.process = None

    def pause(self):
        if self.process and self.process.poll() is None and os.name == "posix":
            os.kill(self.process.pid, signal.SIGSTOP)
            self._paused = True

    def resume(self):
        if self.process and self.process.poll() is None and os.name == "posix":
            os.kill(self.process.pid, signal.SIGCONT)
            self._paused = False

    @property
    def paused(self):
        return self._paused

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None


class PiDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pi Runtime Dashboard")
        self.geometry("1000x720")
        self.runner = CommandRunner(self._append_output, self._on_command_finish)
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Blue.Horizontal.TProgressbar", troughcolor="#f0f0f0", background="#4A90E2")

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Environment frame
        env_frame = ttk.Labelframe(main_frame, text="1. Entorno virtual", padding=10)
        env_frame.pack(fill=tk.X, pady=5)
        ttk.Button(env_frame, text="Activar mobilenet-env", command=self.activate_env).pack(side=tk.LEFT)
        self.env_label = ttk.Label(env_frame, text="Estado: inactivo")
        self.env_label.pack(side=tk.LEFT, padx=10)

        # Runtime frame
        runtime_frame = ttk.Labelframe(main_frame, text="2. Clasificación en vivo (pi_runtime.py)", padding=10)
        runtime_frame.pack(fill=tk.X, pady=5)

        runtime_inputs = ttk.Frame(runtime_frame)
        runtime_inputs.pack(fill=tk.X, pady=5)

        self.src_var = tk.StringVar(value="0")
        self.width_var = tk.StringVar(value="640")
        self.height_var = tk.StringVar(value="480")
        self.fps_var = tk.StringVar(value="30")
        self.skip_var = tk.StringVar(value="2")
        self.conf_var = tk.StringVar(value="0.35")
        self.show_var = tk.BooleanVar(value=True)
        self.record_var = tk.BooleanVar(value=True)

        self._add_labeled_entry(runtime_inputs, "Fuente (--src / pipeline):", self.src_var, 0, 0, colspan=2)
        self._add_labeled_entry(runtime_inputs, "Width:", self.width_var, 1, 0)
        self._add_labeled_entry(runtime_inputs, "Height:", self.height_var, 1, 1)
        self._add_labeled_entry(runtime_inputs, "FPS:", self.fps_var, 2, 0)
        self._add_labeled_entry(runtime_inputs, "Skip:", self.skip_var, 2, 1)
        self._add_labeled_entry(runtime_inputs, "Conf:", self.conf_var, 3, 0)

        chk_frame = ttk.Frame(runtime_frame)
        chk_frame.pack(fill=tk.X)
        ttk.Checkbutton(chk_frame, text="Mostrar ventana (--show)", variable=self.show_var, command=self._confirm_update).pack(side=tk.LEFT)
        ttk.Checkbutton(chk_frame, text="Grabar MP4 (--record)", variable=self.record_var, command=self._confirm_update).pack(side=tk.LEFT, padx=10)

        btn_frame = ttk.Frame(runtime_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Aplicar configuración", command=self._confirm_update).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Iniciar Clasificación", command=self.run_runtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Detener", command=self.stop_runtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Pausar", command=self.pause_runtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reanudar", command=self.resume_runtime).pack(side=tk.LEFT, padx=5)

        self.runtime_cmd_var = tk.StringVar()
        ttk.Label(runtime_frame, text="Comando actual:").pack(anchor="w")
        ttk.Label(runtime_frame, textvariable=self.runtime_cmd_var, wraplength=900, foreground="#0A84FF").pack(anchor="w")

        # Analyze frame
        analyze_frame = ttk.Labelframe(main_frame, text="3. Analizar run (analyze_run.py)", padding=10)
        analyze_frame.pack(fill=tk.X, pady=5)
        analyze_inputs = ttk.Frame(analyze_frame)
        analyze_inputs.pack(fill=tk.X)
        self.run_path_var = tk.StringVar(value=DEFAULT_RUN_DIR)
        self._add_labeled_entry(analyze_inputs, "Carpeta del run:", self.run_path_var, 0, 0, colspan=2)
        ttk.Button(analyze_inputs, text="Seleccionar carpeta", command=self._select_run_dir).grid(row=0, column=2, padx=5)
        ttk.Button(analyze_frame, text="Analizar carpeta", command=self.run_analysis).pack(anchor="w", pady=5)

        ttk.Label(analyze_frame, text="Gráfico adicional: 'frame_overview.png' muestra FPS vs terreno y detección de baches (colores=terreno, círculos=sin bache, X=bache).").pack(anchor="w")

        # Status + progress
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_frame, style="Blue.Horizontal.TProgressbar", mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=20)

        # Output console
        console_frame = ttk.Labelframe(main_frame, text="Salida de comandos", padding=10)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.console = tk.Text(console_frame, height=12, state=tk.DISABLED, bg="#111", fg="#eee")
        self.console.pack(fill=tk.BOTH, expand=True)

        self._update_runtime_preview()

    def _add_labeled_entry(self, parent, label, var, row, col, colspan=1):
        ttk.Label(parent, text=label).grid(row=row, column=col * 2, sticky="w", padx=5, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=col * 2 + 1, columnspan=colspan, sticky="ew", padx=5, pady=3)
        entry.bind("<FocusOut>", lambda e: self._confirm_update())
        return entry

    def _append_output(self, line: str):
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, line + "\n")
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _on_command_finish(self, code):
        self.progress.stop()
        msg = "Comando finalizado" if code == 0 else f"Comando terminado con código {code}"
        self.status_var.set(msg)

    def _confirm_update(self):
        self._update_runtime_preview()
        self.status_var.set("Configuración actualizada.")

    def _update_runtime_preview(self):
        cmd = self._build_runtime_command(preview=True)
        self.runtime_cmd_var.set(cmd)

    def _build_runtime_command(self, preview=False):
        args = [
            f'python "{DEFAULT_RUNTIME_SCRIPT}"',
            f'--multitask "{DEFAULT_MULTITASK}"',
            f'--yolo "{DEFAULT_YOLO}"',
            f'--src {self.src_var.get().strip()}',
            f'--width {self.width_var.get().strip()}',
            f'--height {self.height_var.get().strip()}',
            f'--fps {self.fps_var.get().strip()}',
            f'--skip {self.skip_var.get().strip()}',
            f'--conf {self.conf_var.get().strip()}',
        ]
        if self.show_var.get():
            args.append("--show")
        if self.record_var.get():
            args.append("--record")
        command = " ".join(args)
        if preview:
            return command
        return f"source ~/mobilenet-env/bin/activate && {command}"

    def run_runtime(self):
        cmd = self._build_runtime_command()
        self.status_var.set("Ejecutando pi_runtime...")
        self.progress.start(8)
        self.runner.start(cmd)

    def stop_runtime(self):
        self.runner.stop()
        self.progress.stop()
        self.status_var.set("Proceso detenido.")

    def pause_runtime(self):
        if os.name != "posix":
            messagebox.showinfo("No disponible", "La pausa solo está disponible en sistemas POSIX.")
            return
        self.runner.pause()
        self.status_var.set("Proceso en pausa.")

    def resume_runtime(self):
        if os.name != "posix":
            return
        self.runner.resume()
        self.status_var.set("Proceso reanudado.")

    def activate_env(self):
        self.status_var.set("Activando entorno...")
        self.progress.start(8)
        threading.Thread(target=self._run_simple_command, args=(DEFAULT_ENV_ACTIVATE, "Entorno activado."), daemon=True).start()

    def _run_simple_command(self, command, success_msg):
        try:
            output = subprocess.check_output(["/bin/bash", "-lc", command], stderr=subprocess.STDOUT, text=True)
            self._append_output(output.strip())
            self.env_label.config(text=f"Estado: {output.strip().splitlines()[-1]}")
            self.status_var.set(success_msg)
        except subprocess.CalledProcessError as e:
            self._append_output(e.output)
            self.status_var.set("Fallo al activar entorno.")
        finally:
            self.progress.stop()

    def _select_run_dir(self):
        directory = filedialog.askdirectory(initialdir="/home/aldrin/cnn-terreno/pi_runs")
        if directory:
            self.run_path_var.set(directory)
            self.status_var.set("Carpeta de análisis seleccionada.")

    def run_analysis(self):
        run_path = self.run_path_var.get().strip()
        cmd = f'source ~/mobilenet-env/bin/activate && python "{DEFAULT_ANALYZE_SCRIPT}" --run "{run_path}"'
        self.status_var.set("Ejecutando analyze_run...")
        self.progress.start(8)
        self.runner.start(cmd)


def main():
    app = PiDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
