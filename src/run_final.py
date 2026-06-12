"""
Final competition training: 2 epochs on all data, same as main's approach.
"""
import sys; sys.path.insert(0, '.')
import os, collections
import numpy as np; import pandas as pd
import torch; import torch.nn as nn; import torch.optim as optim
from tqdm import tqdm
from model import EnsembleAttentionGRU

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Seed (same as main)
import random; random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# Load data (same as main's get_dataloaders)
df = pd.read_parquet('../output/800_stocks_features.parquet')
df['trade_date'] = pd.to_datetime(df['trade_date'])
mask_c = df['ts_code'].str.startswith('300')
df = pd.concat([df[mask_c & (df['trade_date']>='2020-08-24')],
                df[~mask_c & (df['trade_date']>='2016-01-01')]], ignore_index=True)
df = df.sort_values(['ts_code','trade_date']).reset_index(drop=True)

label_cols = ['label_return_1d','label_return_3d','label_return_5d']
feat_cols = [c for c in df.columns if c not in ['ts_code','trade_date']+label_cols]
features_np = df[feat_cols].values.astype(np.float32)
targets_np = df['label_return_5d'].values.astype(np.float32)

# Standardize (same formula as main)
mean=features_np.mean(axis=0); std=features_np.std(axis=0)+1e-8
features_np=(features_np-mean)/std

ft=torch.from_numpy(features_np); tt=torch.from_numpy(targets_np)
codes=df['ts_code'].values
dates_arr=df['trade_date'].dt.strftime('%Y-%m-%d').values
stock_changes=(df['ts_code']!=df['ts_code'].shift(1)).values

# Dataset (same as main's StockSequenceDataset)
class S(torch.utils.data.Dataset):
    def __init__(s,f,t,d,sc,sl,ts,te):
        s.f=f; s.t=t; s.d=d; s.sl=sl; s.idx=[]; cs=0
        for i in range(len(f)):
            if sc[i]: cs=i
            if i-cs+1>=sl:
                dd=s.d[i]
                if dd>=ts and dd<=te: s.idx.append(i)
        s.idx=np.array(s.idx,dtype=np.int32)
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        e=s.idx[i]; st=e-s.sl+1
        return s.f[st:e+1],s.t[e],s.d[e],''

ds=S(ft,tt,dates_arr,stock_changes,15,'2016-01-01','2026-05-29')
dl=torch.utils.data.DataLoader(ds,batch_size=12288,shuffle=True,drop_last=True,pin_memory=True)
print(f"Train: {len(ds)} samples, {len(dl)} batches")

# Model (same as main)
model=EnsembleAttentionGRU(len(feat_cols),hidden_size=64,num_layers=2,dropout=0.2,num_models=3).to(DEVICE)

# HybridLoss (same as main)
class H(nn.Module):
    def __init__(s,a=0.5,m=0.001,mp=1000):
        super().__init__(); s.a=a; s.m=m; s.mp=mp; s.mse=nn.MSELoss()
    def forward(s,p,t,d):
        mse=s.mse(p.squeeze(),t)
        if s.a>=1.0: return mse
        ps=p.squeeze(); dev=p.device
        di=collections.defaultdict(list)
        for i,dd in enumerate(d): di[dd].append(i)
        pw=[]
        for dd,ix in di.items():
            if len(ix)<2: continue
            it=torch.tensor(ix,dtype=torch.long,device=dev)
            pp=ps[it]; tt=t[it]
            pd=pp.unsqueeze(1)-pp.unsqueeze(0); td=tt.unsqueeze(1)-tt.unsqueeze(0)
            mask=torch.triu(torch.abs(td)>s.m,diagonal=1)
            if not mask.any(): continue
            vp=pd[mask]; ys=torch.sign(td[mask])
            if len(vp)>s.mp:
                perm=torch.randperm(len(vp),device=dev)[:s.mp]; vp=vp[perm]; ys=ys[perm]
            pw.append(torch.nn.functional.softplus(-ys*vp).mean())
        if not pw: return mse
        return s.a*mse+(1-s.a)*torch.stack(pw).mean()

criterion=H(a=0.5)
optimizer=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

# Train 3 epochs
for epoch in range(2):
    model.train(); tl=0.0
    for X,y,dates,_ in tqdm(dl,desc=f"E{epoch+1}/2",leave=False):
        X,y=X.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad(); loss=criterion(model(X),y,dates)
        loss.backward(); optimizer.step(); tl+=loss.item()
    print(f"Epoch {epoch+1}/2 | Loss: {tl/len(dl):.4f}")

torch.save(model.state_dict(),'checkpoints/best_ensemble.pth')

# Predict
latest=df['trade_date'].max()
pds=S(ft,tt,dates_arr,stock_changes,15,latest.strftime('%Y-%m-%d'),latest.strftime('%Y-%m-%d'))
model.eval(); scores=[]
with torch.no_grad():
    for X,_,_,_ in torch.utils.data.DataLoader(pds,batch_size=512):
        scores.extend(model(X.to(DEVICE)).cpu().numpy())

pred=pd.DataFrame({'ts_code':[codes[pds.idx[i]] for i in range(len(pds))],'score':scores})
pred=pred.sort_values('score',ascending=False)
pred.to_csv('../output/competition_predictions.csv',index=False)

print(f"\nTop 20 BUY (data: {latest.date()}):")
for i,r in enumerate(pred.head(20).itertuples(),1):
    print(f"  {i:2d}. {r.ts_code}  {r.score:.4f}")
print(f"\nSaved. Total stocks: {len(pred)}")
