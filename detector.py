"""
detector.py - Live inference engine.

Scores every new auction ingested from the live poll against the trained model
and inserts flagged records into the database for review.

Usage:
  python detector.py --live        # Poll + score continuously
  python detector.py --score-all   # Score all existing auctions in DB (batch)
  python detector.py --backtest    # Score demo/historic data and print results
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
log_runtime_environment(
    logging.getLogger("detector.bootstrap"),
    "detector.py",
    required_modules=("numpy", "sklearn"),
)

import numpy as np
from config import (
    POLL_INTERVAL_SECONDS,
    PRICE_RATIO_HARD_FLOOR,
    LGBM_FLAG_THRESHOLD,
    TIER_THRESHOLDS,
    LOG_DIR,
)
from database import (
    init_db,
    insert_auctions,
    insert_flag,
    get_conn,
    db_stats,
    reset_flagged_queue,
)
from collector import fetch_ended_auctions, _normalise_ended
from features import compute_features_batch, compute_features_single, FEATURE_COLUMNS

try:
    import lightgbm  # noqa — just checking it's available
    LGBM_AVAILABLE = True
    LGBM_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    LGBM_AVAILABLE = False
    LGBM_IMPORT_ERROR = exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "detector.log"),
    ],
)
logger = logging.getLogger("detector")
if not LGBM_AVAILABLE:
    logger.warning(describe_optional_lightgbm_failure(LGBM_IMPORT_ERROR))


def _model_path(name: str):
    return config.get_model_paths()[name]


def _decode_item_json(auction: dict) -> dict:
    blob = auction.get("decoded_item_json")
    if not isinstance(blob, str) or not blob:
        return {}
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _has_pet_skin(auction: dict) -> bool:
    decoded = _decode_item_json(auction)
    pet = decoded.get("pet")
    return isinstance(pet, dict) and isinstance(pet.get("pet_skin"), str) and bool(pet["pet_skin"])


def _should_skip_for_sparse_comparables(auction: dict, features: dict) -> tuple[bool, str | None]:
    is_pet = features.get("is_pet") == 1.0
    has_skin = bool(auction.get("decoded_skin")) or _has_pet_skin(auction)
    has_quality_median = features.get("has_quality_median") == 1.0

    if is_pet and not has_quality_median:
        return True, "pet has no reliable comparable sales yet"

    if has_skin and not has_quality_median:
        return True, "skinned item has no reliable comparable sales yet"

    if features.get("has_item_median") != 1.0 and not has_quality_median:
        return True, "item has no reliable comparable sales yet"

    return False, None


def _passes_strict_evidence_gate(features: dict, scores: dict) -> tuple[bool, str | None]:
    pqmr = features.get("price_to_quality_median_ratio", 1.0)
    pmr = features.get("price_to_median_ratio", 1.0)
    has_quality = features.get("has_quality_median") == 1.0
    has_item = features.get("has_item_median") == 1.0
    repeat_pair = features.get("repeat_pair") == 1.0
    very_fast = features.get("very_fast_sale") == 1.0
    anomaly_score = scores.get("anomaly_score") or 0.0
    fraud_prob = scores.get("fraud_prob")

    if has_quality and pqmr >= 3.0:
        return True, None

    if has_item and pmr >= 8.0 and (very_fast or repeat_pair):
        return True, None

    if fraud_prob is not None and fraud_prob >= 0.90 and has_item and pmr >= PRICE_RATIO_HARD_FLOOR:
        return True, None

    if anomaly_score >= 0.98 and has_quality and pqmr >= 2.0 and repeat_pair:
        return True, None

    return False, "insufficient pricing evidence for a tight fraud flag"


# ── Model loading ──────────────────────────────────────────────────────────────

class ModelBundle:
    """Holds loaded model artifacts and exposes a unified score() method."""

    def __init__(self):
        self.if_model   = None
        self.lgbm_model = None
        self.scaler     = None
        self.meta       = {}
        self.version    = "unloaded"
        self._load()

    def _load(self) -> None:
        if_path = _model_path("isolation_forest")
        if not if_path.exists():
            logger.warning(
                "No Isolation Forest model found. Train one with: %s",
                recommended_command("train.py", "--stage", "isolation", *config.get_context_cli_args()),
            )
            return

        with open(if_path, "rb") as f:
            self.if_model = pickle.load(f)
        logger.info(f"Loaded Isolation Forest from {if_path}")

        scaler_path = _model_path("scaler")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        lgbm_path = _model_path("lgbm_classifier")
        if lgbm_path.exists() and LGBM_AVAILABLE:
            with open(lgbm_path, "rb") as f:
                self.lgbm_model = pickle.load(f)
            logger.info("Loaded LightGBM classifier")
        elif lgbm_path.exists() and not LGBM_AVAILABLE:
            logger.warning(
                "Found a LightGBM model artifact but could not load LightGBM. %s",
                describe_optional_lightgbm_failure(LGBM_IMPORT_ERROR),
            )

        meta_path = _model_path("meta")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
            self.version = self.meta.get("version", "unknown")

        logger.info("Scoring mode: %s", self.mode())

    def is_ready(self) -> bool:
        return self.if_model is not None

    def mode(self) -> str:
        if self.if_model is None:
            return "unavailable"
        if self.lgbm_model is None:
            return "Isolation Forest only"
        return "Isolation Forest + LightGBM"

    def score(self, features: dict) -> dict:
        """
        Returns:
          anomaly_score  — Isolation Forest score, normalised 0-1 (higher = more anomalous)
          fraud_prob     — LightGBM probability 0-1 (None if no supervised model)
          combined_score — blended score used for tiering
        """
        if not self.is_ready():
            return {"anomaly_score": None, "fraud_prob": None, "combined_score": None}

        # Build feature vector in the same column order as training
        cols   = self.meta.get("features", FEATURE_COLUMNS)
        x_vals = np.array([[features.get(c, 0.0) for c in cols]], dtype=float)
        x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler:
            x_vals = self.scaler.transform(x_vals)

        # Isolation Forest: score_samples returns negative values (more negative = more anomalous)
        raw_if_score = float(-self.if_model.score_samples(x_vals)[0])

        # Normalise to [0, 1] using the score range stored in model metadata.
        # score_samples returns negative values; we negate so higher = more anomalous.
        # Range is calibrated from training data distribution.
        score_min = self.meta.get("score_min", 0.35)   # most-normal score (negated)
        score_max = self.meta.get("score_max", 0.80)   # most-anomalous score (negated)
        anomaly_score = float(np.clip(
            (raw_if_score - score_min) / max(score_max - score_min, 1e-6),
            0, 1
        ))

        fraud_prob = None
        if self.lgbm_model is not None:
            fraud_prob = float(self.lgbm_model.predict_proba(x_vals)[0, 1])

        # Blend: if we have both models, weight LightGBM higher
        if fraud_prob is not None:
            combined = 0.3 * anomaly_score + 0.7 * fraud_prob
        else:
            combined = anomaly_score

        return {
            "anomaly_score": anomaly_score,
            "fraud_prob":    fraud_prob,
            "combined_score": combined,
        }


# ── Flagging logic ─────────────────────────────────────────────────────────────

def _assign_tier(score: float) -> str | None:
    """Map a combined score to a tier label. Returns None if below all thresholds."""
    if score >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score >= TIER_THRESHOLDS["LOW"]:
        return "LOW"
    return None


def _build_reasons(features: dict, scores: dict) -> list[str]:
    """Generate a human-readable list of reasons why this auction was flagged."""
    reasons = []

    pqmr = features.get("price_to_quality_median_ratio", -1)
    if features.get("has_quality_median") == 1.0 and pqmr > PRICE_RATIO_HARD_FLOOR:
        reasons.append(
            f"Price is {pqmr:.1f}× the median for similarly upgraded versions of this item"
        )

    pmr = features.get("price_to_median_ratio", -1)
    if pmr > PRICE_RATIO_HARD_FLOOR and features.get("has_quality_median") != 1.0:
        reasons.append(f"Price is {pmr:.1f}× the 7-day median for this item")

    plbin = features.get("price_to_lbin_ratio", -1)
    if plbin > 2:
        reasons.append(f"Price is {plbin:.1f}× the lowest BIN at time of sale")

    if features.get("very_fast_sale") == 1.0:
        ts = features.get("time_to_sell_s", 0)
        reasons.append(f"Sold in {ts:.0f}s (< 2 minutes)")

    if features.get("repeat_pair") == 1.0:
        count = int(features.get("seller_buyer_pair_count_30d", 0))
        reasons.append(f"Seller and buyer have transacted {count}× in 30 days")

    quality_score = features.get("item_quality_score", 0.0)
    if quality_score >= 0.45:
        reasons.append(f"Item has significant upgrade value (quality score {quality_score:.2f})")

    if scores.get("anomaly_score", 0) and scores["anomaly_score"] > 0.8:
        reasons.append(f"High anomaly score ({scores['anomaly_score']:.2f}) from Isolation Forest")

    if scores.get("fraud_prob") is not None and scores["fraud_prob"] > LGBM_FLAG_THRESHOLD:
        reasons.append(f"LightGBM fraud probability: {scores['fraud_prob']:.1%}")

    return reasons or ["Pattern matches known IRL trading signals"]


def _score_features(auction: dict, features: dict, bundle: ModelBundle) -> dict | None:
    """Apply model and evidence gates to an already-computed feature record."""
    # Hard price floor — if price is within normal range, skip entirely
    pmr = features.get("price_to_median_ratio", -1)
    if 0 < pmr < PRICE_RATIO_HARD_FLOOR:
        return None

    skip_sparse, skip_reason = _should_skip_for_sparse_comparables(auction, features)
    if skip_sparse:
        logger.debug(
            "Skipping %s because %s",
            auction.get("auction_id"),
            skip_reason,
        )
        return None

    scores   = bundle.score(features)
    combined = scores.get("combined_score")

    if combined is None:
        return None

    passes_gate, gate_reason = _passes_strict_evidence_gate(features, scores)
    if not passes_gate:
        logger.debug(
            "Skipping %s because %s",
            auction.get("auction_id"),
            gate_reason,
        )
        return None

    tier = _assign_tier(combined)
    if tier is None:
        return None

    reasons = _build_reasons(features, scores)

    return {
        "auction_id":    auction["auction_id"],
        "flagged_at":    int(time.time() * 1000),
        "model_version": bundle.version,
        "anomaly_score": scores.get("anomaly_score"),
        "fraud_prob":    scores.get("fraud_prob"),
        "tier":          tier,
        "reasons":       json.dumps(reasons),
    }


def score_auction(auction: dict, bundle: ModelBundle) -> dict | None:
    """Compute live features for one auction, then score it."""
    try:
        features = compute_features_single(auction)
    except Exception as e:
        logger.debug(f"Feature error for {auction.get('auction_id')}: {e}")
        return None
    return _score_features(auction, features, bundle)


# ── Ingestion modes ────────────────────────────────────────────────────────────

def score_all_existing(bundle: ModelBundle) -> None:
    """Rescore all auctions using the batched feature path used by training."""
    logger.info("Computing batched features for existing auctions...")
    feature_frame = compute_features_batch()
    if feature_frame.empty:
        logger.warning("No features available; leaving the existing flagged queue unchanged.")
        return

    # Load only the fields needed by the evidence gate after feature calculation.
    # Keeping decoded_item_json out of memory prevents swapping on t3.micro.
    with get_conn() as conn:
        rows = {
            row["auction_id"]: dict(row)
            for row in conn.execute(
                "SELECT auction_id, item_name, decoded_skin FROM auctions"
            ).fetchall()
        }

    cleared = reset_flagged_queue()
    logger.info("Cleared %s stale flagged rows before full rescore", cleared)
    logger.info("Scoring %s existing auctions...", len(feature_frame))
    flagged = 0
    for auction_id, feature_row in feature_frame.iterrows():
        auction = rows[auction_id]
        flag = _score_features(auction, feature_row.to_dict(), bundle)
        if flag:
            insert_flag(flag)
            flagged += 1

    logger.info(f"Done. Flagged {flagged} / {len(feature_frame)} auctions.")
    logger.info(f"DB stats: {db_stats()}")


def live_detect(bundle: ModelBundle) -> None:
    """
    Continuously poll for new ended auctions and score them in real time.
    New flags are inserted into the DB for review via the dashboard.
    """
    logger.info("Starting live detector...")
    poll_count   = 0
    total_flagged = 0

    while True:
        try:
            start = time.time()
            raw_auctions = fetch_ended_auctions()
            if not raw_auctions:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            rows = [r for a in raw_auctions if (r := _normalise_ended(a)) is not None]
            new  = insert_auctions(rows)

            flagged_this_round = 0
            for auction in rows:
                flag = score_auction(auction, bundle)
                if flag:
                    insert_flag(flag)
                    flagged_this_round += 1
                    if flag["tier"] == "HIGH":
                        logger.warning(
                            f"[HIGH] {auction['item_name']} — "
                            f"{auction['final_price']:,} coins — "
                            f"score={flag['anomaly_score']:.2f} — "
                            f"{json.loads(flag['reasons'])[0]}"
                        )

            total_flagged += flagged_this_round
            elapsed        = time.time() - start
            poll_count    += 1

            logger.info(
                f"[Poll #{poll_count}] {len(raw_auctions)} auctions, "
                f"{new} new, {flagged_this_round} flagged "
                f"({elapsed:.1f}s) | total flagged: {total_flagged}"
            )

            time.sleep(max(1, POLL_INTERVAL_SECONDS - elapsed))

        except KeyboardInterrupt:
            logger.info("Live detector stopped.")
            break
        except Exception as e:
            logger.error(f"Error in live loop: {e}", exc_info=True)
            time.sleep(30)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",       action="store_true", help="Continuous live polling")
    parser.add_argument("--score-all",  action="store_true", help="Score all existing DB auctions")
    parser.add_argument("--backtest",   action="store_true", help="Score + print summary")
    parser.add_argument("--demo",       action="store_true", help="Use the isolated demo DB and demo model artifacts")
    args = parser.parse_args()
    config.set_storage_context("demo" if args.demo else "real")
    init_db()
    logger.info("Using storage %s", config.describe_storage_context())

    bundle = ModelBundle()
    if not bundle.is_ready():
        logger.error(
            "No model loaded. Train the Isolation Forest first with: %s",
            recommended_command("train.py", "--stage", "isolation", *config.get_context_cli_args()),
        )
        exit(1)

    if args.live:
        live_detect(bundle)
    elif args.score_all or args.backtest:
        score_all_existing(bundle)
    else:
        parser.print_help()
