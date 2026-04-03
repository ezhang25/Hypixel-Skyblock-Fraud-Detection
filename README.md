# Hypixel Skyblock IRL Trade Detector

Detects suspicious auction house transactions that indicate real-money trading (IRL trading),
where items are sold for prices wildly above fair market value to transfer in-game currency.

## Project Structure

```
skyblock_irl_detector/
├── config.py              # API keys, thresholds, DB path
├── collector.py           # Hypixel API data ingestion (historic + live)
├── database.py            # SQLite schema and query helpers
├── features.py            # Feature engineering pipeline
├── train.py               # Model training (Isolation Forest + LightGBM)
├── detector.py            # Live inference engine
├── dashboard.py           # CLI review dashboard for flagged auctions
├── data/                  # SQLite database files
├── models/                # Saved model artifacts
└── logs/                  # Ingestion and detection logs
```

## Quick Start

```bash
# 1. Install dependencies into the project virtualenv
./.venv/bin/python -m pip install -r requirements.txt

# 2. Collect real Hypixel auction data
./.venv/bin/python collector.py --mode backfill --pages 100

# 3. Train the baseline anomaly detector on the real DB
./.venv/bin/python train.py --stage isolation

# 4. Start the live detector against the real DB/models
./.venv/bin/python detector.py --live

# 5. Review flagged auctions from the real DB
./.venv/bin/python dashboard.py
```

## Notes

- Run the project with `./.venv/bin/python` so the installed dependencies match the interpreter.
- `train.py --stage isolation` and `detector.py` work without LightGBM.
- Real workflows use:
  - DB: `data/auctions.db`
  - models: `models/`
- Demo workflows use:
  - DB: `data/demo_auctions.db`
  - models: `models/demo/`
- On macOS, LightGBM may require OpenMP. If it fails to load, install it with:

```bash
brew install libomp
./.venv/bin/python -m pip install --force-reinstall lightgbm
```

- Once you have at least 30 labeled examples, you can optionally train the supervised model:

```bash
./.venv/bin/python train.py --stage lgbm
```

## Demo Workflow

```bash
# Collect synthetic demo data into the isolated demo DB
./.venv/bin/python collector.py --mode demo

# Train demo-only model artifacts
./.venv/bin/python train.py --demo --stage isolation

# Score the demo DB using the demo model artifacts
./.venv/bin/python detector.py --demo --score-all

# Review demo flags without touching the real DB
./.venv/bin/python dashboard.py --demo --stats
./.venv/bin/python dashboard.py --demo
```

## How it works

1. **Collector** polls `/v2/skyblock/auctions/ended` and `/v2/skyblock/auctions` every 60s
2. **Features** computes price ratios, bid patterns, seller history signals per auction
3. **Isolation Forest** flags statistical outliers unsupervised (no labels needed to start)
4. **LightGBM** classifier takes over once you manually label flagged cases
5. **Dashboard** shows flagged auctions with explainability — why each was flagged

## IRL Trading Signals

- Sale price >> rolling median for that item (price_to_median_ratio)
- Zero or one competing bids on high-value items
- BIN (Buy It Now) rather than auction format
- Same seller/buyer pair appearing repeatedly
- Seller's price significantly above their own historical average
- Unusually fast sale after listing
