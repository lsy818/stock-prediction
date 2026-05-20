import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = '../../data'

csi500_path = os.path.join(DATA_DIR, 'index_weight', '000905.csv')
if os.path.exists(csi500_path):
    df_500 = pd.read_csv(csi500_path)
    if '成份券代码Constituent Code' in df_500.columns:
        print("Correct")
    else: print("Wrong")