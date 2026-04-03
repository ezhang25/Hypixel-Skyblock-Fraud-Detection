"""
config.py - Central configuration for the IRL trade detector.
Set your Hypixel API key here (get one at https://developer.hypixel.net).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
LOG_DIR     = BASE_DIR / "logs"
REAL_DB_PATH = DATA_DIR / "auctions.db"
DEMO_DB_PATH = DATA_DIR / "demo_auctions.db"
REAL_MODEL_DIR = MODEL_DIR
DEMO_MODEL_DIR = MODEL_DIR / "demo"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)
DEMO_MODEL_DIR.mkdir(exist_ok=True)

_STORAGE_CONTEXT = "real"


def set_storage_context(context: str) -> None:
    global _STORAGE_CONTEXT
    if context not in {"real", "demo"}:
        raise ValueError(f"Unknown storage context: {context}")
    _STORAGE_CONTEXT = context


def get_storage_context() -> str:
    return _STORAGE_CONTEXT


def is_demo_context() -> bool:
    return _STORAGE_CONTEXT == "demo"


def get_db_path() -> Path:
    return DEMO_DB_PATH if is_demo_context() else REAL_DB_PATH


def get_model_dir() -> Path:
    model_dir = DEMO_MODEL_DIR if is_demo_context() else REAL_MODEL_DIR
    model_dir.mkdir(exist_ok=True)
    return model_dir


def get_model_paths() -> dict[str, Path]:
    model_dir = get_model_dir()
    return {
        "dir": model_dir,
        "isolation_forest": model_dir / "isolation_forest.pkl",
        "lgbm_classifier": model_dir / "lgbm_classifier.pkl",
        "scaler": model_dir / "scaler.pkl",
        "meta": model_dir / "model_meta.json",
    }


def get_context_cli_args() -> tuple[str, ...]:
    return ("--demo",) if is_demo_context() else ()


def describe_storage_context() -> str:
    return (
        f"context={get_storage_context()} db={get_db_path()} "
        f"models={get_model_dir()}"
    )

# ── Hypixel API ────────────────────────────────────────────────────────────────
# No API key needed — both auction endpoints are fully public.
ENDPOINTS = {
    "auctions_ended": "https://api.hypixel.net/v2/skyblock/auctions_ended",
    "auctions_live":  "https://api.hypixel.net/v2/skyblock/auctions",
}

# Hypixel rate limit: ~300 requests/minute. We stay well under it.
POLL_INTERVAL_SECONDS   = 60   # How often to poll live auctions
REQUEST_DELAY_SECONDS   = 0.5  # Delay between paginated requests

# ── Feature engineering ────────────────────────────────────────────────────────
# Rolling window for computing "normal" price per item
MEDIAN_WINDOW_DAYS      = 7
# Minimum number of historical sales needed to compute a reliable median
MIN_SALES_FOR_MEDIAN    = 5

# ── Anomaly detection thresholds ──────────────────────────────────────────────
# Price must be at least this many times the median to flag as suspicious
PRICE_RATIO_HARD_FLOOR  = 3.0

# Isolation Forest contamination — expected fraction of outliers in training data.
# Start conservative (1-2%), adjust after reviewing flagged cases.
ISOLATION_FOREST_CONTAMINATION = 0.02

# LightGBM threshold — probability above which we flag (0-1).
# Lower = more flags (more false positives), higher = fewer flags (more misses).
LGBM_FLAG_THRESHOLD     = 0.6

# ── Alert tiers ───────────────────────────────────────────────────────────────
# Auctions are tiered by how suspicious they look.
TIER_THRESHOLDS = {
    "HIGH":   0.85,   # Very likely IRL trade — review immediately
    "MEDIUM": 0.65,   # Suspicious — review when possible
    "LOW":    0.45,   # Borderline — log for pattern analysis
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
