import os, yaml, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open("configs/trading.yaml") as f: cfg = yaml.safe_load(f)
data = pd.read_parquet(os.path.join(cfg["processed_path"], "sample.parquet"))

class DS(Dataset):
    def __init__(self, df, window=60):
        arr = df[["mid","spread"]].values.astype(np.float32)
        X, y = [], []
        for i in range(window, len(arr)-5):
            X.append(arr[i-window:i])
            fut = arr[i+1:i+6,0].mean() - arr[i,0]
            y.append(2 if fut>0 else (0 if fut<0 else 1))
        self.X, self.y = np.stack(X), np.array(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

ds = DS(data); dl = DataLoader(ds, batch_size=64, shuffle=True)

class M(nn.Module):
    def __init__(self, f=2): 
        super().__init__(); self.conv=nn.Conv1d(f,32,3,padding=1); self.pool=nn.AdaptiveAvgPool1d(1); self.fc=nn.Linear(32,3)
    def forward(self,x): x=x.permute(0,2,1); x=torch.relu(self.conv(x)); x=self.pool(x).squeeze(-1); return self.fc(x)

m=M(); opt=torch.optim.Adam(m.parameters(),1e-3); lossf=nn.CrossEntropyLoss()
for e in range(5):
    tot=0
    for xb,yb in dl:
        logits=m(xb); loss=lossf(logits,yb); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    print("epoch",e,"loss",tot/len(dl))
os.makedirs("models", exist_ok=True); torch.save(m.state_dict(),"models/fastactor.pt"); print("saved models/fastactor.pt")
