"""
Global configuration for the stock prediction project.
"""
import os

# ── Paths ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "A股数据")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
METRIC_DIR = os.path.join(DATA_DIR, "metric")
MONEYFLOW_DIR = os.path.join(DATA_DIR, "moneyflow")
STOCK_ST_DIR = os.path.join(DATA_DIR, "stock_st")
MARKET_DIR = os.path.join(DATA_DIR, "market")
NEWS_DIR = os.path.join(DATA_DIR, "news")
BASIC_PATH = os.path.join(DATA_DIR, "basic.csv")
TRADE_CAL_PATH = os.path.join(DATA_DIR, "trade_cal.csv")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Time Splits ────────────────────────────────────────
TRAIN_START = "2019-01-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-12-31"  # will be capped by latest data

# ── Sliding Window ─────────────────────────────────────
SEQ_LEN = 60          # lookback window in trading days
PRED_HORIZON = 1       # predict T+N return
TARGET_N = 1

# ── Stock Pool ─────────────────────────────────────────
EXCLUDE_MARKETS = ["北交所"]   # exclude BeiJiaoSuo
EXCLUDE_ST = True              # exclude ST / *ST stocks
MIN_LIST_DAYS = 60             # minimum trading history required

# ── Model ──────────────────────────────────────────────
D_MODEL = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.3
OUTPUT_DIM = 1

# ── Training ───────────────────────────────────────────
BATCH_SIZE = 2048
SAMPLE_STEP = 5       # use 1 out of every N training samples for speed
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3

# ── Trading Strategy ───────────────────────────────────
N_HOLD = 20          # number of stocks held
K_TRADE = 2          # number of stocks to rotate daily
INITIAL_CAPITAL = 1_000_000  # 100万

# ── System ─────────────────────────────────────────────
DEVICE = "cuda"
NUM_WORKERS = 0  # Windows: must be 0 to avoid multiprocessing issues
USE_AMP = True        # automatic mixed precision
