"""Predict for latest available date using raw data (no labels needed)."""
import sys; sys.path.insert(0, '.')
import os, glob
import numpy as np; import pandas as pd
import torch
from tqdm import tqdm
from model import EnsembleAttentionGRU

DATA_DIR = '../data/A股数据'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Build universe
universe = set()
for f in sorted(glob.glob(f'{DATA_DIR}/index_weight/*_000300.SH.csv')):
    if os.path.getsize(f) > 1000:
        universe.update(pd.read_csv(f)['con_code'].unique())

csi500 = f'{DATA_DIR}/index_weight/000905.csv'
if os.path.exists(csi500):
    df5 = pd.read_csv(csi500)
    for code in df5['成份券代码Constituent Code'].dropna().unique():
        cs = str(int(code)).zfill(6)
        if cs[0] == '6': universe.add(f'{cs}.SH')
        elif cs[0] in '03': universe.add(f'{cs}.SZ')

# ST filter
st_set = set()
for f in glob.glob(f'{DATA_DIR}/stock_st/*.csv'):
    try: st_set.update(pd.read_csv(f)['ts_code'].dropna().unique())
    except: pass

print(f'Universe: {len(universe)}, ST: {len(st_set)}')

# 2. Load ALL daily data (we need historical for rolling features)
daily_files = sorted(glob.glob(f'{DATA_DIR}/daily/*.csv'))
dfs = []
for f in tqdm(daily_files[-100:], desc='Loading recent data'):  # only need recent for features
    fname = os.path.basename(f)
    df = pd.read_csv(f, dtype={'ts_code': str})
    df = df[df['ts_code'].isin(universe)]
    df = df[~df['ts_code'].isin(st_set)]
    if df.empty: continue
    mf = f'{DATA_DIR}/metric/{fname}'
    if os.path.exists(mf):
        dm = pd.read_csv(mf, dtype={'ts_code': str})
        dm = dm.drop(columns=['trade_date','close'], errors='ignore')
        df = df.merge(dm, on='ts_code', how='left')
    mm = f'{DATA_DIR}/moneyflow/{fname}'
    if os.path.exists(mm):
        dmm = pd.read_csv(mm, dtype={'ts_code': str})
        dmm = dmm.drop(columns=['trade_date'], errors='ignore')
        df = df.merge(dmm, on='ts_code', how='left')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
df = df.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f'Recent data: {df.shape}')

# We also need historical data for rolling computations (ma60, etc.)
# Load more historical data
hist_dfs = []
for f in tqdm(daily_files[-400:-100], desc='Loading history'):
    fname = os.path.basename(f)
    dfh = pd.read_csv(f, dtype={'ts_code': str})
    dfh = dfh[dfh['ts_code'].isin(universe)]
    dfh = dfh[~dfh['ts_code'].isin(st_set)]
    if dfh.empty: continue
    mf = f'{DATA_DIR}/metric/{fname}'
    if os.path.exists(mf):
        dm = pd.read_csv(mf, dtype={'ts_code': str})
        dm = dm.drop(columns=['trade_date','close'], errors='ignore')
        dfh = dfh.merge(dm, on='ts_code', how='left')
    hist_dfs.append(dfh)

df_hist = pd.concat(hist_dfs, ignore_index=True)
df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'].astype(str))
df_all = pd.concat([df_hist, df], ignore_index=True)
df_all = df_all.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f'All data: {df_all.shape}')

# Fill NaN
cols_fill = df_all.columns.drop(['ts_code','trade_date'])
df_all[cols_fill] = df_all.groupby('ts_code')[cols_fill].ffill()
df_all[cols_fill] = df_all[cols_fill].fillna(0)

# 3. Compute features (same as main's data_preprocessing.py)
g = df_all.groupby('ts_code')
df_all['ma5'] = g['close'].transform(lambda x: x.rolling(5).mean())
df_all['ma10'] = g['close'].transform(lambda x: x.rolling(10).mean())
df_all['ma20'] = g['close'].transform(lambda x: x.rolling(20).mean())
df_all['mom5'] = g['close'].transform(lambda x: x.pct_change(5))
df_all['mom10'] = g['close'].transform(lambda x: x.pct_change(10))
df_all['volatility20'] = g['pct_chg'].transform(lambda x: x.rolling(20).std())
df_all['vol_ma5'] = g['vol'].transform(lambda x: x.rolling(5).mean())
ema12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df_all['macd'] = ema12 - ema26
delta = g['close'].transform(lambda x: x.diff())
up = delta.clip(lower=0); down = -delta.clip(upper=0)
ema_up = up.groupby(df_all['ts_code']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
ema_down = down.groupby(df_all['ts_code']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
rs = ema_up / (ema_down + 1e-10)
df_all['rsi14'] = 100 - (100 / (1 + rs))
if 'net_mf_amount' in df_all.columns:
    df_all['net_mf_amount_ma5'] = g['net_mf_amount'].transform(lambda x: x.rolling(5).mean())
    df_all['mf_to_amount_ratio'] = df_all['net_mf_amount'] / (df_all['amount'] + 1e-8)
if 'pe_ttm' in df_all.columns:
    df_all['pe_ttm_ma60'] = g['pe_ttm'].transform(lambda x: x.rolling(60).mean())
    df_all['pe_ttm_deviation'] = (df_all['pe_ttm'] - df_all['pe_ttm_ma60']) / (df_all['pe_ttm_ma60'] + 1e-8)
if 'circ_mv' in df_all.columns:
    df_all['log_circ_mv'] = np.log1p(df_all['circ_mv'])

# 4. Extract features for latest date
latest_date = df_all['trade_date'].max()
print(f'Latest date: {latest_date.date()}')

# Board filter
mask_c = df_all['ts_code'].str.startswith('300')
df_filt = pd.concat([
    df_all[mask_c & (df_all['trade_date'] >= '2020-08-24')],
    df_all[~mask_c & (df_all['trade_date'] >= '2016-01-01')]
], ignore_index=True)
df_filt = df_filt.sort_values(['ts_code','trade_date']).reset_index(drop=True)

label_cols = ['label_return_1d', 'label_return_3d', 'label_return_5d']
feat_cols = [c for c in df_filt.columns if c not in ['ts_code','trade_date'] + label_cols]
feat_cols = [c for c in feat_cols if c in df_filt.columns]
print(f'Features: {len(feat_cols)}')

fn = df_filt[feat_cols].values.astype(np.float32)
mean = fn.mean(axis=0); std = fn.std(axis=0) + 1e-8
fn = (fn - mean) / std
ft = torch.from_numpy(fn)
codes = df_filt['ts_code'].values
dates_arr = df_filt['trade_date'].dt.strftime('%Y-%m-%d').values
stock_changes = (df_filt['ts_code'] != df_filt['ts_code'].shift(1)).values

# 5. Load model
model = EnsembleAttentionGRU(len(feat_cols), hidden_size=64, num_layers=2,
                              dropout=0.2, num_models=3).to(DEVICE)
model.load_state_dict(torch.load('checkpoints/best_ensemble.pth', map_location=DEVICE))
model.eval()

print(f'Features mean/std computed from {len(fn)} rows')

# 6. Get stocks on latest date with 15-day history
class S:
    def __init__(s, f, d, sc, sl, ts, te):
        s.f=f; s.d=d; s.sl=sl; s.idx=[]; cs=0
        for i in range(len(f)):
            if sc[i]: cs=i
            if i-cs+1>=sl:
                dd=s.d[i]
                if dd>=ts and dd<=te: s.idx.append(i)
        s.idx=np.array(s.idx, dtype=np.int32)
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        e=s.idx[i]; st=e-s.sl+1
        return s.f[st:e+1], s.d[e], s.idx[i]

lds = latest_date.strftime('%Y-%m-%d')
ds = S(ft, dates_arr, stock_changes, 15, lds, lds)
print(f'Stocks on {lds}: {len(ds)}')

scores = []
with torch.no_grad():
    for X, _, _ in torch.utils.data.DataLoader(ds, batch_size=512):
        scores.extend(model(X.to(DEVICE)).cpu().numpy())

pred = pd.DataFrame({
    'ts_code': [codes[ds.idx[i]] for i in range(len(ds))],
    'date': lds,
    'score': scores
}).sort_values('score', ascending=False)

basic = pd.read_csv(f'{DATA_DIR}/basic.csv', dtype={'ts_code': str})
pred = pred.merge(basic[['ts_code','name']], on='ts_code', how='left')

out = f'../output/predictions_{latest_date.strftime("%m%d")}.csv'
pred.to_csv(out, index=False)

print(f'\nTop 20 for {lds}:')
for i, r in enumerate(pred.head(20).itertuples(), 1):
    print(f'  {i:2d}. {r.ts_code} {r.name}  {r.score:.4f}')
print(f'\nSaved to {out}')
