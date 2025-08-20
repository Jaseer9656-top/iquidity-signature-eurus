# Synthetic EURUSD snapshots for pipeline testing; replace with real OANDA/TrueFX later.
import os, time, random, yaml, pandas as pd
with open("configs/trading.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg["data_path"], exist_ok=True)
rows = []
base = 1.0950
for _ in range(2000):
    base += random.gauss(0, 0.00005)
    bid = round(base - 0.00005, 5)
    ask = round(base + 0.00005, 5)
    rows.append({"ts": int(time.time()*1000), "best_bid": bid, "best_ask": ask})
df = pd.DataFrame(rows)
out = os.path.join(cfg["data_path"], "sample.parquet")
df.to_parquet(out)
print("Wrote", out, "rows=", len(df))
