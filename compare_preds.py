import pandas as pd
old = pd.read_csv('D:/study/AI3003-proj/output/competition_predictions.csv')
new = pd.read_csv('D:/study/AI3003-proj/output/predictions_0604.csv')
old = old[['ts_code','name','score']].rename(columns={'score':'s_0522'})
new = new[['ts_code','name','score']].rename(columns={'score':'s_0604'})
old['r_0522'] = old['s_0522'].rank(ascending=False).astype(int)
new['r_0604'] = new['s_0604'].rank(ascending=False).astype(int)
m = old.merge(new, on=['ts_code','name'], how='outer')

# Focus on Top 30 from old
top = m[m['r_0522'].notna()].head(30).copy()
top['chg'] = (top['r_0522'] - top['r_0604']).fillna(0).astype(int)

print(f"{'Code':<12} {'Name':<8} {'R_0522':>6} {'R_0604':>6} {'Chg':>6}  {'Score_0522':>8} {'Score_0604':>8}")
print("-" * 75)
for _, r in top.iterrows():
    chg = int(r['chg'])
    arrow = ' Up' if chg > 0 else (' Dn' if chg < 0 else ' --')
    r5 = r['r_0604'] if pd.notna(r['r_0604']) else '-'
    s6 = r['s_0604'] if pd.notna(r['s_0604']) else 0
    print(f"{r['ts_code']:<12} {str(r['name']):<8} {int(r['r_0522']):>6} {str(r5):>6} {chg:+d}{arrow:<3} {r['s_0522']:>8.4f} {float(s6):>8.4f}")

# New entries in top 20 that weren't before
print("\n--- Dropped from Top 20 ---")
dropped = m[(m['r_0522'] <= 20) & ((m['r_0604'].isna()) | (m['r_0604'] > 20))]
for _, r in dropped.iterrows():
    r6 = int(r['r_0604']) if pd.notna(r['r_0604']) else '-'
    print(f"  {r['ts_code']} {r['name']}  0522 rank={int(r['r_0522'])} -> 0604 rank={r6}")

print("\n--- New in Top 20 ---")
new_top = m[(m['r_0604'] <= 20) & ((m['r_0522'].isna()) | (m['r_0522'] > 20))]
for _, r in new_top.iterrows():
    r5 = int(r['r_0522']) if pd.notna(r['r_0522']) else '-'
    print(f"  {r['ts_code']} {r['name']}  0522 rank={r5} -> 0604 rank={int(r['r_0604'])}")
