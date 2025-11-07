import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def find_latest_run(base: Path) -> Path:
    if not base.exists():
        raise SystemExit(f"No existe carpeta de runs: {base}")
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        raise SystemExit(f"No hay subcarpetas en {base}")
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]


def load_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def perc(value: int, total: int) -> float:
    return (value / total * 100.0) if total else 0.0


def write_table_csv(path: Path, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table_json(path: Path, rows: list):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def build_distribution_rows(counter: Counter, prefix: str, total: int) -> list:
    rows = []
    for label, count in counter.items():
        rows.append({
            "category": prefix,
            "label": str(label),
            "count": count,
            "percentage": round(perc(count, total), 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Resumen mejorado de runs: métricas, tablas limpias y gráficos anotados.")
    ap.add_argument("--run", default=None, help="Carpeta del run (por defecto: último en pi_runs)")
    ap.add_argument("--base", default=str(Path("cnn-terreno") / "pi_runs"), help="Base de runs")
    ap.add_argument("--no-plots", action="store_true", help="No generar gráficos (para entornos sin GUI)")
    args = ap.parse_args()

    base = Path(args.base)
    run_dir = Path(args.run) if args.run else find_latest_run(base)

    jsonl_path = run_dir / "results.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"No se encontró {jsonl_path}")

    n_frames = 0
    ts_min = None
    ts_max = None
    terr_cnt = Counter()
    rough_cnt = Counter()
    potholes_total = 0
    frames_with_pothole = 0
    frames_info: List[Dict[str, Any]] = []

    for rec in load_jsonl(jsonl_path):
        n_frames += 1
        ts = rec.get("ts")
        if isinstance(ts, (int, float)):
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
        terr = rec.get("terrain", {})
        rough = rec.get("roughness", {})
        label = terr.get("label", "?")
        rid = rough.get("id", -1)
        terr_cnt[label] += 1
        rough_cnt[rid] += 1
        potholes = rec.get("potholes", []) or []
        k = len(potholes)
        potholes_total += k
        frame_idx = rec.get("frame", n_frames)
        frames_info.append({
            "frame": frame_idx,
            "ts": ts,
            "terrain": label,
            "rough_id": rid,
            "potholes": k,
        })
        if k > 0:
            frames_with_pothole += 1

    prev_ts = None
    fps_values = []
    for info in frames_info:
        ts = info.get("ts")
        if prev_ts is not None and isinstance(ts, (int, float)) and ts > prev_ts:
            fps = 1.0 / (ts - prev_ts)
        else:
            fps = fps_values[-1] if fps_values else 0.0
        info["fps"] = fps
        fps_values.append(fps)
        if isinstance(ts, (int, float)):
            prev_ts = ts

    frame_table = []
    for info in frames_info:
        frame_table.append({
            "frame": info["frame"],
            "timestamp": info.get("ts", 0.0),
            "fps_est": round(info.get("fps", 0.0), 3),
            "terrain": info["terrain"],
            "rough_id": info["rough_id"],
            "potholes": info["potholes"],
        })
    frame_csv = run_dir / "frame_analysis.csv"
    frame_json = run_dir / "frame_analysis.json"
    write_table_csv(frame_csv, frame_table)
    write_table_json(frame_json, frame_table)

    dt = (ts_max - ts_min) if (ts_min is not None and ts_max is not None) else 0.0
    minutes = dt / 60.0 if dt > 0 else 0.0
    potholes_per_min = potholes_total / minutes if minutes > 0 else 0.0
    frames_with_potholes_pct = perc(frames_with_pothole, n_frames)

    summary = {
        "run_dir": str(run_dir),
        "n_frames": n_frames,
        "duration_sec": dt,
        "duration_min": dt / 60.0 if dt else 0.0,
        "potholes_total": potholes_total,
        "potholes_per_min": potholes_per_min,
        "frames_with_potholes_pct": frames_with_potholes_pct,
        "terrain_counts": dict(terr_cnt),
        "rough_counts": {str(k): v for k, v in rough_cnt.items()},
    }

    out_summary = run_dir / "summary.json"
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Resumen guardado en:", out_summary)
    print("Frames:", n_frames, " Duración (s):", round(dt, 1), " Baches totales:", potholes_total)
    print("Baches/min:", round(potholes_per_min, 2), " Frames con baches (%):", round(frames_with_potholes_pct, 2))

    # Tablas limpias (CSV + JSON)
    table_rows = []
    table_rows.append({
        "category": "global",
        "label": "frames",
        "count": n_frames,
        "percentage": 100.0 if n_frames else 0.0,
    })
    table_rows.append({
        "category": "global",
        "label": "frames_con_baches",
        "count": frames_with_pothole,
        "percentage": round(frames_with_potholes_pct, 2),
    })
    table_rows.append({
        "category": "global",
        "label": "baches_totales",
        "count": potholes_total,
        "percentage": 0.0,
    })
    table_rows.append({
        "category": "global",
        "label": "baches_por_minuto",
        "count": round(potholes_per_min, 3),
        "percentage": 0.0,
    })
    table_rows.extend(build_distribution_rows(terr_cnt, "terreno", sum(terr_cnt.values())))
    table_rows.extend(build_distribution_rows(rough_cnt, "rugosidad", sum(rough_cnt.values())))

    table_csv = run_dir / "summary_table.csv"
    table_json = run_dir / "summary_table.json"
    write_table_csv(table_csv, table_rows)
    write_table_json(table_json, table_rows)
    print("Tabla limpia exportada en:", table_csv, "y", table_json)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import matplotlib.patches as mpatches

            def make_bar(counter: Counter, title: str, filename: str):
                if not counter:
                    return
                labels = list(counter.keys())
                counts = [counter[k] for k in labels]
                total = sum(counts)
                percentages = [perc(c, total) for c in counts]

                plt.figure(figsize=(6, 4))
                bars = plt.bar(labels, counts, color='#1f77b4')
                plt.title(title)
                plt.xlabel('Clase')
                plt.ylabel('Frames')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                for bar, cnt, pct in zip(bars, counts, percentages):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{cnt} ({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                plt.savefig(run_dir / filename, dpi=180)
                plt.close()

            make_bar(terr_cnt, 'Distribución de terreno (frames vs porcentaje)', 'terrain_dist.png')
            make_bar(rough_cnt, 'Distribución de rugosidad (frames vs porcentaje)', 'rough_dist.png')

            # Evolución acumulada de baches (opcional si hay tiempo)
            if potholes_total > 0:
                cumulative = []
                total_so_far = 0
                frame_index = 0
                for rec in load_jsonl(jsonl_path):
                    frame_index += 1
                    total_so_far += len(rec.get('potholes', []) or [])
                    cumulative.append((frame_index, total_so_far))
                if cumulative:
                    xs, ys = zip(*cumulative)
                    plt.figure(figsize=(6, 4))
                    plt.step(xs, ys, where='post', color='#ff7f0e')
                    plt.title('Baches acumulados a lo largo del run')
                    plt.xlabel('Frame')
                    plt.ylabel('Baches acumulados')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(run_dir / 'potholes_cumulative.png', dpi=180)
                    plt.close()

            if frames_info:
                terrain_palette = {
                    'liso': '#4CAF50',
                    'grava': '#A0522D',
                    'tierra': '#D2691E',
                    'obstaculo': '#E53935',
                }
                default_color = '#607D8B'
                fig, ax = plt.subplots(figsize=(8, 4))
                for has_bache in (False, True):
                    xs = []
                    ys = []
                    cols = []
                    for info in frames_info:
                        flag = info['potholes'] > 0
                        if flag != has_bache:
                            continue
                        xs.append(info['frame'])
                        ys.append(info.get('fps', 0.0))
                        cols.append(terrain_palette.get(info['terrain'], default_color))
                    if xs:
                        marker = 'o' if not has_bache else 'X'
                        label = 'Sin bache' if not has_bache else 'Bache detectado'
                        ax.scatter(xs, ys, c=cols, marker=marker, edgecolor='black', linewidths=0.4,
                                   s=50, alpha=0.85, label=label)
                ax.set_title('Visión general: FPS estimado vs clase de terreno')
                ax.set_xlabel('Frame')
                ax.set_ylabel('FPS estimado')
                ax.grid(True, alpha=0.3)
                terrain_handles = [mpatches.Patch(color=terrain_palette.get(name, default_color),
                                                  label=f"Terreno: {name}") for name in terrain_palette]
                marker_handles = [
                    Line2D([0], [0], marker='o', color='w', label='Sin bache', markerfacecolor='white',
                           markeredgecolor='black'),
                    Line2D([0], [0], marker='X', color='w', label='Bache detectado', markerfacecolor='black',
                           markeredgecolor='black')
                ]
                ax.legend(handles=terrain_handles + marker_handles, loc='upper right', fontsize=8)
                plt.tight_layout()
                overview_path = run_dir / 'frame_overview.png'
                plt.savefig(overview_path, dpi=180)
                plt.close()

                legend_path = run_dir / 'frame_overview_legend.txt'
                with open(legend_path, 'w', encoding='utf-8') as f:
                    f.write('Leyenda - frame_overview.png\\n')
                    f.write('- Colores: clase de terreno detectada (liso, grava, tierra, obstaculo).\\n')
                    f.write('- Marcador \"o\": frame sin bache detectado.\\n')
                    f.write('- Marcador \"X\": frame con bache detectado.\\n')
                    f.write('- Eje X: índice de frame (secuencia capturada).\\n')
                    f.write('- Eje Y: FPS estimado (1 / Δt entre frames consecutivos).\\n')

            print("Gráficos generados en:", run_dir)
        except ImportError:
            print("[aviso] matplotlib no disponible; omitiendo gráficos")
        except Exception as e:
            print("[aviso] Error generando gráficos:", e)

    # Sugerencias automáticas simples
    suggestions = []
    if frames_with_potholes_pct < 1.0 and potholes_total > 0:
        suggestions.append("Revisar iluminación/ángulo: pocos frames con baches detectados aunque hubo detecciones.")
    if frames_with_potholes_pct > 20.0:
        suggestions.append("Alto porcentaje de frames con baches; considerar subir --conf o revisar falsos positivos.")
    if perc(terr_cnt.get('liso', 0), n_frames) > 80:
        suggestions.append("Predominan terrenos lisos; valida si el recorrido es representativo o diversifica datos.")
    if not suggestions:
        suggestions.append("Distribución razonable; continuar monitoreo y ajustar parámetros según nuevos runs.")

    suggestions_path = run_dir / "suggestions.txt"
    with open(suggestions_path, 'w', encoding='utf-8') as f:
        for line in suggestions:
            f.write(f"- {line}\n")
    print("Sugerencias guardadas en:", suggestions_path)


if __name__ == "__main__":
    main()
