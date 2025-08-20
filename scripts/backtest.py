import os, yaml, torch, pandas as pd, numpy as np, torch.nn as nn
with open("configs/trading.yaml") as f: cfg = yaml.safe_load(f)
df = pd.read_parquet(os.path.join(cfg["processed_path"], "sample.parquet"))
arr = df[["mid","spread"]].values.astype(float)

class M(nn.Module):
    def __init__(self): super().__init__(); self.c=nn.Conv1d(2,32,3,padding=1); self.p=nn.AdaptiveAvgPool1d(1); self.f=nn.Linear(32,3)
    def forward(self,x): x=x.permute(0,2,1); x=torch.relu(self.c(x)); x=self.p(x).squeeze(-1); return self.f(x)
m = M(); m.load_state_dict(torch.load("models/fastactor.pt", map_location="cpu")); m.eval()

cash = cfg["account_balance"]
for i in range(60, len(arr)-5):
    w = torch.tensor(arr[i-60:i]).unsqueeze(0).float()
    pred = torch.argmax(m(w), dim=-1).item()
    if pred==2:   # long
        entry=arr[i,0]; tp=entry+0.0002; fut=arr[i+1:i+6,0]
        cash += 10 if (fut>=tp).any() else -5
    elif pred==0: # short
        entry=arr[i,0]; tp=entry-0.0002; fut=arr[i+1:i+6,0]
        cash += 10 if (fut<=tp).any() else -5
print("final cash:", cash)
