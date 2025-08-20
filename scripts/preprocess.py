import os, yaml, pandas as pd
with open("configs/trading.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg["processed_path"], exist_ok=True)
infile = os.path.join(cfg["data_path"], "sample.parquet")
df = pd.read_parquet(infile)
df["mid"] = (df["best_bid"] + df["best_ask"]) / 2
df["spread"] = df["best_ask"] - df["best_bid"]
outfile = os.path.join(cfg["processed_path"], "sample.parquet")
df[["ts","mid","spread"]].to_parquet(outfile)
print("Wrote", outfile, "rows=", len(df))
