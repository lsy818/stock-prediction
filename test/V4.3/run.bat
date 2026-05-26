@echo off

call conda activate cv

echo "Training model..."
python train.py

echo "Running backtest..."
python backtest.py

echo "Done!"
pause
