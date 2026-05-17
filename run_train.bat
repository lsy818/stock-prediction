@echo off
REM Run training after data preparation completes
D:\Anaconda\envs\dl\python.exe -u main.py --mode train --model gru_attention 2>&1 | Tee-Object -FilePath "output\train.log"
D:\Anaconda\envs\dl\python.exe -u main.py --mode eval --model gru_attention
D:\Anaconda\envs\dl\python.exe -u main.py --mode backtest --model gru_attention
