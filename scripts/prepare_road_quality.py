# scripts/prepare_road_quality.py
import argparse, os, pickle, io, math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def closest_idx(arr_ns, t_ns):
    # arr_ns: np.ndarray int64 timestamps (ns)
    return int(np.argmin(np.abs(arr_ns - t_ns)))

def std_in_window(times_ns, accel_z, center_ns, win_s=1.0, fs=100.0):
    half = int((win_s/2.0)*fs)
    idx = closest_idx(times_ns, center_ns)
    i0 = max(0, idx-half); i1 = min(len(accel_z), idx+half+1)
    if i1 - i0 < 5:
        return np.nan
    return float(np.std(accel_z[i0:i1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Carpeta Road Quality (contiene runs .pkl)")
    ap.add_argument("--out_img", required=True, help="Carpeta donde guardar imágenes extraídas")
    ap.add_argument("--csv_out", required=True, help="CSV salida: offroad_rq_labels.csv")
    ap.add_argument("--k", type=int, default=3, help="clusters de rugosidad (k-means)")
    args = ap.parse_args()

    root = Path(args.root)
    out_img = Path(args.out_img); out_img.mkdir(parents=True, exist_ok=True)
    rows = []

    pkl_files = list(root.rglob("*.pkl"))
    print(f"[INFO] runs detectados: {len(pkl_files)}")

    all_std = []
    tmp_rows = []  # guardamos (path, std, has_label)

    for i,pk in enumerate(sorted(pkl_files)):
        with open(pk, "rb") as f:
            d = pickle.load(f)

        cam_data = d["camera"]["data"]            # bytes[]
        cam_t = d["camera"]["time"]["abs"]        # int64[]
        imu_z = d["imu"]["accel"]["z"]           # float[]
        imu_t = d["imu"]["time"]["abs"]          # int64[]

        # opcional: etiquetas de anomalía (para obstáculo)
        lab = d.get("labels", [])

        # indexamos intervals
        intervs = []
        for L in lab:
            t0 = int(L["rel_t_start"]*1e9 + d["camera"]["time"]["abs"][0])
            t1 = int(L["rel_t_end"]*1e9 + d["camera"]["time"]["abs"][0])
            intervs.append((t0,t1,L.get("anomaly","no_anomaly")))

        for j,(jpeg, t_ns) in enumerate(zip(cam_data, cam_t)):
            try:
                img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            except Exception:
                continue
            # guardamos imagen
            name = f"run{i:03d}_t{int(t_ns)}.jpg"
            path = out_img / name
            img.save(path, quality=90)

            # rugosidad por std(z) en 1.0 s
            st = std_in_window(np.array(imu_t), np.array(imu_z), int(t_ns), win_s=1.0, fs=100.0)
            if not math.isnan(st):
                all_std.append(st)
                # obstáculo si cae en algún intervalo etiquetado != no_anomaly
                obs = 0
                for (a,b,anom) in intervs:
                    if a <= t_ns <= b and anom != "no_anomaly":
                        obs = 1
                        break
                tmp_rows.append((str(path), st, obs))

    # discretizar rugosidad con KMeans(k)
    if len(all_std) == 0:
        print("[ERROR] no hay std(z) válidos")
        return
    X = np.array(all_std).reshape(-1,1)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X)
    # usamos quantiles globales para centrar (más estable)
    centers = sorted(km.cluster_centers_.reshape(-1))
    thr = []
    if args.k == 3:
        # separar por límites entre centros
        thr = [(centers[0]+centers[1])/2, (centers[1]+centers[2])/2]
    elif args.k == 2:
        thr = [(centers[0]+centers[1])/2]
    else:
        # cortes uniformes por percentiles
        thr = list(np.quantile(all_std, [i/args.k for i in range(1,args.k)]))

    def bucket(v):
        if args.k == 2:
            return 0 if v < thr[0] else 1
        if args.k == 3:
            return 0 if v < thr[0] else (1 if v < thr[1] else 2)
        # genérico
        for i,t in enumerate(thr):
            if v < t: return i
        return len(thr)

    out_rows = []
    for p, st, obs in tmp_rows:
        out_rows.append({"path": p, "roughness_label": bucket(st), "obstacle": int(obs)})

    df = pd.DataFrame(out_rows)
    df.to_csv(args.csv_out, index=False)
    print(f"[DONE] CSV: {args.csv_out}  (rows={len(df)})")

if __name__ == "__main__":
    main()


