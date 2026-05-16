# pyrefly: ignore [missing-import]
import akshare as ak
import pandas as pd

# 获取 平安银行(000001) 从 2015年至今 的日线前复权数据
# symbol: 股票代码, period: 周期, adjust: qfq(前复权)/hfq(后复权)/""(不复权)
print("正在下载数据，请稍候...")
df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20150101", end_date="20260514", adjust="qfq")

# 提取我们在 MVP 中需要的基础列：日期、开、收、高、低、成交量
df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

print("数据获取成功！前5行数据如下：")
print(df.head())

# 保存为 CSV 文件，方便后续读取训练
df.to_csv("000001_daily_qfq.csv")