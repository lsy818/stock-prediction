import pandas as pd
df = pd.read_csv('D:/study/AI3003-proj/data/A股数据/daily/20260605.csv', dtype={'ts_code': str})
holdings = ['300502','300394','002384','300308','603986','300274','600584','600183','300476']
print("=== Friday June 5 Performance ===")
for code in holdings:
    row = df[df['ts_code'].str.startswith(code)]
    if len(row) > 0:
        c = row['close'].values[0]; p = row['pct_chg'].values[0]
        print(f"  {code}: close={c:>8.2f}  pct_chg={p:>7.2f}%")
# 300308 detail
r = df[df['ts_code'] == '300308.SZ']
if len(r) > 0:
    print(f"\n  300308.SZ 中际旭创 detail:")
    print(f"    open={r['open'].values[0]} high={r['high'].values[0]} low={r['low'].values[0]} close={r['close'].values[0]} pct_chg={r['pct_chg'].values[0]:.2f}%")
