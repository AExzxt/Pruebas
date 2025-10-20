import argparse, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csvs", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

dfs = []
for p in args.csvs:
    df = pd.read_csv(p)
    # normalizar columnas mínimas
    if "roughness_label" not in df.columns:
        raise SystemExit(f"{p} no tiene 'roughness_label'")
    if "path" not in df.columns:
        raise SystemExit(f"{p} no tiene 'path'")
    dfs.append(df[["path","roughness_label"]])

out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["path"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(f"[DONE] merged -> {args.out}  rows={len(out)}")



