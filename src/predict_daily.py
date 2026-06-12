"""Daily prediction: use existing model on latest data. No retraining."""
import sys; sys.path.insert(0, '.')
import numpy as np; import pandas as pd
import torch
from model import EnsembleAttentionGRU

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load latest data
df = pd.read_parquet('../output/800_stocks_features.parquet')
df['trade_date'] = pd.to_datetime(df['trade_date'])

# Board filter (same as main)
mask_c = df['ts_code'].str.startswith('300')
df = pd.concat([df[mask_c & (df['trade_date'] >= '2020-08-24')],
                df[~mask_c & (df['trade_date'] >= '2016-01-01')]], ignore_index=True)
df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

label_cols = ['label_return_1d', 'label_return_3d', 'label_return_5d']
feat_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date'] + label_cols]

fn = df[feat_cols].values.astype(np.float32)
codes = df['ts_code'].values
dates_arr = df['trade_date'].dt.strftime('%Y-%m-%d').values
stock_changes = (df['ts_code'] != df['ts_code'].shift(1)).values

# Standardize: fit on all data
mean = fn.mean(axis=0); std = fn.std(axis=0) + 1e-8
fn = (fn - mean) / std
ft = torch.from_numpy(fn)

# Load model
model = EnsembleAttentionGRU(len(feat_cols), hidden_size=64, num_layers=2,
                              dropout=0.2, num_models=3).to(DEVICE)
model.load_state_dict(torch.load('checkpoints/best_ensemble.pth', map_location=DEVICE))
model.eval()

# Find latest date
latest = df['trade_date'].max()
print(f"Latest data: {latest.date()}")

# Build prediction dataset for latest date
class S:
    def __init__(s, feats, dates, sc, sl, ts, te):
        s.feats=feats; s.dates=dates; s.seq_len=sl; s.idx=[]; cs=0
        for i in range(len(feats)):
            if sc[i]: cs=i
            if i-cs+1>=sl:
                d=s.dates[i]
                if d>=ts and d<=te: s.idx.append(i)
        s.idx=np.array(s.idx, dtype=np.int32)
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        e=s.idx[i]; st=e-s.seq_len+1
        return s.feats[st:e+1], s.dates[e], s.idx[i]

ds = S(ft, dates_arr, stock_changes, 15,
       latest.strftime('%Y-%m-%d'), latest.strftime('%Y-%m-%d'))

scores = []
with torch.no_grad():
    for X, _, _ in torch.utils.data.DataLoader(ds, batch_size=512):
        scores.extend(model(X.to(DEVICE)).cpu().numpy())

pred = pd.DataFrame({
    'ts_code': [codes[ds.idx[i]] for i in range(len(ds))],
    'date': [dates_arr[ds.idx[i]] for i in range(len(ds))],
    'score': scores
}).sort_values('score', ascending=False)

# Add names
basic = pd.read_csv('../data/A股数据/basic.csv', dtype={'ts_code': str})
pred = pred.merge(basic[['ts_code', 'name']], on='ts_code', how='left')

# Save (don't overwrite competition_predictions.csv)
out = f'../output/predictions_{latest.strftime("%m%d")}.csv'
pred.to_csv(out, index=False)

print(f"\nTop 20 for {latest.date()}:")
for i, r in enumerate(pred.head(20).itertuples(), 1):
    print(f"  {i:2d}. {r.ts_code} {r.name}  {r.score:.4f}")

print(f"\nSaved to {out}")
print(f"Total stocks: {len(pred)}")
