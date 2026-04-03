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

## Setup

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Run the project with `./.venv/bin/python` so the installed dependencies match the interpreter.

## Real Data Workflow

### Fresh real run

```bash
# 1. Collect ended auctions into the real DB
./.venv/bin/python collector.py --mode backfill --pages 100

# 2. Train or retrain the real Isolation Forest model
./.venv/bin/python train.py --stage isolation

# 3. Score all auctions currently in the real DB
./.venv/bin/python detector.py --score-all

# 4. Review results
./.venv/bin/python dashboard.py --stats
./.venv/bin/python dashboard.py
```

### Continuous ended-auction collection

```bash
# Poll Hypixel ended auctions every 60 seconds and keep inserting new rows
./.venv/bin/python collector.py --mode live
```

Recommended workflow:
- leave `collector.py --mode live` running in one terminal
- retrain manually when you have enough new data:

```bash
./.venv/bin/python train.py --stage isolation
```

- then rescore the DB with the latest model:

```bash
./.venv/bin/python detector.py --score-all
```

### Continuous live detection

After a model has already been trained:

```bash
./.venv/bin/python detector.py --live
```

`detector.py --live` does not retrain the model. It loads the latest saved model and scores new ended auctions as they arrive.

## Demo Workflow

```bash
# 1. Build synthetic auctions in the isolated demo DB
./.venv/bin/python collector.py --mode demo

# 2. Train demo-only model artifacts
./.venv/bin/python train.py --demo --stage isolation

# 3. Score the demo DB using the demo model artifacts
./.venv/bin/python detector.py --demo --score-all

# 4. Review demo flags without touching the real DB
./.venv/bin/python dashboard.py --demo --stats
./.venv/bin/python dashboard.py --demo
```

## Storage Layout

- Real workflows use:
  - DB: `data/auctions.db`
  - models: `models/`
- Demo workflows use:
  - DB: `data/demo_auctions.db`
  - models: `models/demo/`

## Training Notes

- `train.py` is the only command that retrains models.
- `collector.py` only gathers data.
- `detector.py` only loads existing model files and scores auctions.
- `dashboard.py` only reviews what is already in the database.

### Optional LightGBM

`train.py --stage isolation` and `detector.py` work without LightGBM.

On macOS, LightGBM may require OpenMP:

```bash
brew install libomp
./.venv/bin/python -m pip install --force-reinstall lightgbm
```

Once you have at least 30 labeled examples, you can optionally train the supervised model:

```bash
./.venv/bin/python train.py --stage lgbm
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
