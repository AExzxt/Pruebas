import argparse
import json
from collections import Counter
from pathlib import Path


def find_latest_run(base: Path) -> Path:
    if not base.exists():
        raise SystemExit(f"No existe carpeta de runs: {base}")
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        raise SystemExit(f"No hay subcarpetas en {base}")
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]


def load_jsonl(path: Path):
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser(description="Resumen de terreno/rugosidad y baches por minuto, con gráficos.")
    ap.add_argument("--run", default=None, help="Carpeta del run (por defecto: último en pi_runs)")
    ap.add_argument("--base", default=str(Path("cnn-terreno")/"pi_runs"), help="Base de runs")
    ap.add_argument("--no-plots", action="store_true", help="No generar gráficos si no se desea")
    args = ap.parse_args()

    base = Path(args.base)
    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run(base)

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
        if k > 0:
            frames_with_pothole += 1

    dt = (ts_max - ts_min) if (ts_min is not None and ts_max is not None) else 0.0
    minutes = dt / 60.0 if dt > 0 else 0.0
    potholes_per_min = potholes_total / minutes if minutes > 0 else 0.0
    frames_with_potholes_pct = (frames_with_pothole / n_frames * 100.0) if n_frames > 0 else 0.0

    summary = {
        "run_dir": str(run_dir),
        "n_frames": n_frames,
        "duration_sec": dt,
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
    print("Frames:", n_frames, " Duración (s):", round(dt, 1))
    print("Baches totales:", potholes_total, "  Baches/min:", round(potholes_per_min, 2))
    print("Frames con baches (%):", round(frames_with_potholes_pct, 2))
    if terr_cnt:
        total = sum(terr_cnt.values())
        print("Terreno (%):", {k: round(v*100/total, 2) for k, v in terr_cnt.items()})
    if rough_cnt:
        totalr = sum(rough_cnt.values())
        print("Rugosidad (%):", {k: round(v*100/totalr, 2) for k, v in rough_cnt.items()})

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            if terr_cnt:
                labels = list(terr_cnt.keys())
                values = [terr_cnt[k] for k in labels]
                plt.figure(figsize=(5, 3))
                plt.bar(labels, values)
                plt.title("Distribución de terreno")
                plt.tight_layout()
                plt.savefig(run_dir / "terrain_dist.png", dpi=160)
                plt.close()
            if rough_cnt:
                labels = list(rough_cnt.keys())
                values = [rough_cnt[k] for k in labels]
                plt.figure(figsize=(5, 3))
                plt.bar([str(l) for l in labels], values)
                plt.title("Distribución de rugosidad")
                plt.tight_layout()
                plt.savefig(run_dir / "rough_dist.png", dpi=160)
                plt.close()
            print("Gráficos guardados en:", run_dir)
        except Exception as e:
            print("[aviso] No se pudieron generar gráficos:", e)


if __name__ == "__main__":
    main()

