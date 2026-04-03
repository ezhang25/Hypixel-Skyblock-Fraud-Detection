"""
train.py - Train the IRL trade detection models.

Stage 1 (always): Isolation Forest — unsupervised anomaly detection.
  No labels needed. Use this to start flagging from day one.

Stage 2 (once you have labels): LightGBM classifier — supervised detection.
  Trains once you have ≥30 labeled examples (15+ positive, 15+ negative).
  Much more precise than Isolation Forest.

Usage:
  python train.py                    # Train whatever stages are possible
  python train.py --stage isolation  # Only train Isolation Forest
  python train.py --stage lgbm       # Only train LightGBM (needs labels)
  python train.py --eval             # Evaluate existing models on labeled data
"""

import argparse
import json
import logging
import pickle
import time

import config
from runtime import (
    describe_optional_lightgbm_failure,
    log_runtime_environment,
    recommended_command,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log_runtime_environment(
    logging.getLogger("train.bootstrap"),
    "train.py",
    required_modules=("numpy", "pandas", "sklearn"),
)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import RobustScaler

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    LGBM_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    LGBM_AVAILABLE = False
    LGBM_IMPORT_ERROR = exc
    logging.warning(describe_optional_lightgbm_failure(exc))

from config import (
    ISOLATION_FOREST_CONTAMINATION,
    LGBM_FLAG_THRESHOLD,
)
from database import init_db, get_labeled_dataset, db_stats
from features import compute_features_batch, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_VERSION = f"v{int(time.time())}"


def _model_path(name: str):
    return config.get_model_paths()[name]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved: {path}")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Select and order feature columns; fill any missing with column median."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X.values, cols


# ── Stage 1: Isolation Forest ──────────────────────────────────────────────────

def train_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    """
    Train an Isolation Forest on all available auction features.
    This is fully unsupervised — no labels required.
    The contamination parameter is how much of training data we expect
    to be anomalous (fraudulent). Start at 1-2% and tune.
    """
    logger.info(f"Training Isolation Forest on {len(df)} auctions...")
    X, cols = _get_feature_matrix(df)

    scaler = RobustScaler()   # RobustScaler handles price outliers better than StandardScaler
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Score distribution — negate so higher = more anomalous
    raw_scores = -model.score_samples(X_scaled)
    score_min  = float(np.percentile(raw_scores, 5))    # typical normal floor
    score_max  = float(np.percentile(raw_scores, 99))   # near-worst anomaly
    logger.info(
        f"  Score range (negated): [{raw_scores.min():.3f}, {raw_scores.max():.3f}] "
        f"mean={raw_scores.mean():.3f} std={raw_scores.std():.3f}"
    )
    logger.info(f"  Normalisation clamp: [{score_min:.3f}, {score_max:.3f}]")

    predicted_outliers = (model.predict(X_scaled) == -1).sum()
    logger.info(
        f"  Flagged as anomalies: {predicted_outliers} / {len(X)} "
        f"({100*predicted_outliers/len(X):.1f}%)"
    )

    _save(model,  _model_path("isolation_forest"))
    _save(scaler, _model_path("scaler"))

    # Save metadata
    meta = {
        "version":       MODEL_VERSION,
        "trained_at":    int(time.time()),
        "n_train":       len(X),
        "features":      cols,
        "contamination": ISOLATION_FOREST_CONTAMINATION,
        "score_min":     score_min,
        "score_max":     score_max,
        "stage":         "isolation_forest",
    }
    with open(_model_path("meta"), "w") as f:
        json.dump(meta, f, indent=2)

    return model


# ── Stage 2: LightGBM supervised classifier ───────────────────────────────────

def train_lgbm(df: pd.DataFrame, labels: pd.Series) -> object:
    """
    Train a LightGBM binary classifier using manually labeled examples.
    labels: Series indexed by auction_id, values 0 (normal) or 1 (IRL trade).

    The model is trained with heavy class weighting to handle the imbalanced
    dataset (IRL trades are rare even among flagged auctions).
    """
    if not LGBM_AVAILABLE:
        logger.error(
            "LightGBM training is unavailable. %s",
            describe_optional_lightgbm_failure(LGBM_IMPORT_ERROR or RuntimeError("unknown error")),
        )
        return None

    common_ids = df.index.intersection(labels.index)
    if len(common_ids) < 30:
        logger.warning(
            f"Only {len(common_ids)} labeled examples. Need ≥30 for reliable training. "
            "Continue collecting and labeling flagged auctions first."
        )
        return None

    X_df   = df.loc[common_ids]
    y      = labels.loc[common_ids]
    X, cols = _get_feature_matrix(X_df)

    # Load scaler trained in stage 1 (or retrain if missing)
    scaler_path = _model_path("scaler")
    if scaler_path.exists():
        scaler = _load(scaler_path)
        X = scaler.transform(X)
    else:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        _save(scaler, scaler_path)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logger.info(
        f"Training LightGBM on {len(y)} labeled examples: "
        f"{n_pos} IRL trades, {n_neg} normal"
    )

    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,   # handle class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Cross-validation (only if enough examples)
    if len(y) >= 60:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        ap_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
        logger.info(
            f"  CV ROC-AUC:  {auc_scores.mean():.3f} ± {auc_scores.std():.3f}"
        )
        logger.info(
            f"  CV Avg-Prec: {ap_scores.mean():.3f} ± {ap_scores.std():.3f}"
        )

    # Final fit on all labeled data
    model.fit(X, y)

    # Feature importances
    importances = sorted(
        zip(cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top 10 most important features:")
    for feat, imp in importances[:10]:
        logger.info(f"    {feat:<40} {imp:.1f}")

    _save(model, _model_path("lgbm_classifier"))

    # Update metadata
    meta_path = _model_path("meta")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "lgbm_version":      MODEL_VERSION,
        "lgbm_trained_at":   int(time.time()),
        "lgbm_n_labeled":    int(len(y)),
        "lgbm_n_positive":   int(n_pos),
        "lgbm_features":     cols,
        "lgbm_threshold":    LGBM_FLAG_THRESHOLD,
        "stage":             "lgbm",
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, labels: pd.Series) -> None:
    """Print evaluation metrics for any trained models against labeled data."""
    common_ids = df.index.intersection(labels.index)
    if len(common_ids) < 5:
        logger.warning("Not enough labeled data for meaningful evaluation.")
        return

    X_df    = df.loc[common_ids]
    y_true  = labels.loc[common_ids].values
    X, cols = _get_feature_matrix(X_df)

    scaler_path = _model_path("scaler")
    if scaler_path.exists():
        scaler = _load(scaler_path)
        X = scaler.transform(X)

    if_path = _model_path("isolation_forest")
    if if_path.exists():
        logger.info("\n── Isolation Forest evaluation ──")
        if_model = _load(if_path)
        scores   = -if_model.score_samples(X)   # negate: higher = more anomalous
        scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        y_pred   = (if_model.predict(X) == -1).astype(int)
        try:
            logger.info(f"  ROC-AUC:        {roc_auc_score(y_true, scores):.3f}")
            logger.info(f"  Avg Precision:  {average_precision_score(y_true, scores):.3f}")
        except Exception:
            pass
        logger.info("\n" + classification_report(y_true, y_pred, target_names=["normal", "IRL trade"]))

    lgbm_path = _model_path("lgbm_classifier")
    if lgbm_path.exists() and LGBM_AVAILABLE:
        logger.info("\n── LightGBM evaluation ──")
        lgbm_model = _load(lgbm_path)
        probs      = lgbm_model.predict_proba(X)[:, 1]
        y_pred     = (probs >= LGBM_FLAG_THRESHOLD).astype(int)
        try:
            logger.info(f"  ROC-AUC:        {roc_auc_score(y_true, probs):.3f}")
            logger.info(f"  Avg Precision:  {average_precision_score(y_true, probs):.3f}")
        except Exception:
            pass
        logger.info("\n" + classification_report(y_true, y_pred, target_names=["normal", "IRL trade"]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(stage: str = "all", eval_only: bool = False) -> None:
    init_db()
    logger.info("Using storage %s", config.describe_storage_context())
    stats = db_stats()
    logger.info(f"DB: {stats}")

    if stats["total_auctions"] < 100:
        logger.warning(
            "Fewer than 100 auctions in DB. Collect more auctions first with: %s "
            "or use demo data with: %s",
            recommended_command("collector.py", "--mode", "backfill", "--pages", "100"),
            recommended_command("collector.py", "--mode", "demo"),
        )
        return

    # Compute features for all auctions
    df = compute_features_batch()
    if df.empty:
        logger.error("Feature matrix is empty. Check the database.")
        return

    # Load labels if available
    labeled = get_labeled_dataset()
    labels  = pd.Series(
        {r["auction_id"]: r["label"] for r in labeled}
    ) if labeled else pd.Series(dtype=int)

    if eval_only:
        evaluate(df, labels)
        return

    # Train Isolation Forest (always)
    if stage in ("all", "isolation"):
        train_isolation_forest(df)

    # Train LightGBM (only if we have enough labeled examples)
    if stage in ("all", "lgbm"):
        if len(labels) >= 30:
            train_lgbm(df, labels)
        else:
            logger.info(
                f"Skipping LightGBM: only {len(labels)} labels available "
                "(need ≥30). Label more flagged auctions in the dashboard first."
            )

    if labeled:
        evaluate(df, labels)

    logger.info(f"\nModels saved to: {config.get_model_dir()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "isolation", "lgbm"], default="all")
    parser.add_argument("--eval", action="store_true", dest="eval_only")
    parser.add_argument("--demo", action="store_true", help="Use the isolated demo DB and demo model artifacts")
    args = parser.parse_args()
    config.set_storage_context("demo" if args.demo else "real")
    main(stage=args.stage, eval_only=args.eval_only)
