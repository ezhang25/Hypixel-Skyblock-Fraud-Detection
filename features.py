"""
features.py - Feature engineering for IRL trade detection.

Takes raw auction records and computes signals that distinguish legitimate
high-value sales from artificially inflated IRL trades.

Feature groups:
  1. Price signals       — how far above market value is the price?
  2. Auction mechanics   — bid patterns, BIN vs auction, speed of sale
  3. Seller history      — is this seller's behaviour unusual?
  4. Pair signals        — how often does this exact seller/buyer pair transact?
  5. Item context        — rarity tier, category normalisation
"""

import json
import logging
import re
from statistics import median, stdev

from runtime import log_runtime_environment

log_runtime_environment(
    logging.getLogger("features.bootstrap"),
    "features.py",
    required_modules=("numpy", "pandas"),
)

import numpy as np
import pandas as pd

from config import MEDIAN_WINDOW_DAYS, MIN_SALES_FOR_MEDIAN
from database import (
    get_item_price_history,
    get_seller_history,
    get_pair_frequency,
    get_conn,
)

logger = logging.getLogger(__name__)
PET_LEVEL_RE = re.compile(r"\[Lvl\s+(\d+)\]")
PET_MIN_SALES_FOR_MEDIAN = 2

# Rarity tier weights — rarer items legitimately sell for more variance
TIER_MULTIPLIER = {
    "COMMON":    1.0,
    "UNCOMMON":  1.1,
    "RARE":      1.2,
    "EPIC":      1.3,
    "LEGENDARY": 1.5,
    "MYTHIC":    2.0,
    "DIVINE":    2.0,
    "SPECIAL":   2.5,
    "VERY_SPECIAL": 3.0,
    "UNKNOWN":   1.3,
}


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if len(values) >= MIN_SALES_FOR_MEDIAN else None


def _safe_stdev(values: list[float]) -> float | None:
    return float(stdev(values)) if len(values) >= 2 else None


def _safe_quality_median(values: list[float], is_pet: bool) -> float | None:
    min_sales = PET_MIN_SALES_FOR_MEDIAN if is_pet else MIN_SALES_FOR_MEDIAN
    return float(median(values)) if len(values) >= min_sales else None


def _history_sources_for_auction(auction: dict) -> tuple[str, ...]:
    source = auction.get("source", "ended")
    if source == "demo":
        return ("demo", "ended")
    return ("ended",)


def _quality_signature(auction: dict) -> tuple:
    item_key = auction.get("item_id") or auction.get("decoded_item_id") or auction.get("item_name")
    pet = _pet_metadata(auction)
    if item_key == "PET" or pet.get("pet_type"):
        pet_level = int(pet.get("pet_level") or 0)
        pet_level_bucket = min((pet_level // 10) * 10, 100)
        return (
            "PET",
            pet.get("pet_type") or "UNKNOWN",
            pet.get("pet_tier") or auction.get("tier") or "UNKNOWN",
            pet_level_bucket,
            (pet.get("pet_held_item") or "").strip().lower(),
        )

    return (
        item_key,
        int(auction.get("decoded_stars") or 0),
        1 if auction.get("decoded_recombobulated") else 0,
        int(auction.get("decoded_enchant_count") or 0),
        (auction.get("decoded_reforge") or "").strip().lower(),
    )


def _matches_quality_signature(candidate: dict, signature: tuple) -> bool:
    return _quality_signature(candidate) == signature


def _pet_quality_matches(history: list[dict], auction: dict, auction_id: str) -> list[dict]:
    """Return only exact pet-quality matches; never approximate a pet baseline."""
    exact_signature = _quality_signature(auction)
    return [
        h for h in history
        if h["auction_id"] != auction_id and _matches_quality_signature(h, exact_signature)
    ]


def _safe_numeric(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decode_item_json(auction: dict) -> dict:
    blob = auction.get("decoded_item_json")
    if not isinstance(blob, str) or not blob:
        return {}
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _pet_metadata(auction: dict) -> dict:
    pet = {
        "pet_type": auction.get("decoded_pet_type"),
        "pet_tier": auction.get("decoded_pet_tier"),
        "pet_level": auction.get("decoded_pet_level"),
        "pet_exp": auction.get("decoded_pet_exp"),
        "pet_held_item": auction.get("decoded_pet_held_item"),
        "pet_candy_used": auction.get("decoded_pet_candy_used"),
    }

    decoded_item = _decode_item_json(auction)
    decoded_pet = decoded_item.get("pet")
    if isinstance(decoded_pet, dict):
        for key in pet:
            if pet[key] is None and decoded_pet.get(key) is not None:
                pet[key] = decoded_pet.get(key)

    extra = decoded_item.get("extra_attributes")
    if isinstance(extra, dict):
        raw_pet_info = extra.get("petInfo")
        if isinstance(raw_pet_info, str):
            try:
                pet_info = json.loads(raw_pet_info)
            except json.JSONDecodeError:
                pet_info = {}
            if pet["pet_type"] is None:
                pet["pet_type"] = pet_info.get("type")
            if pet["pet_tier"] is None:
                pet["pet_tier"] = pet_info.get("tier")
            if pet["pet_held_item"] is None:
                pet["pet_held_item"] = pet_info.get("heldItem")
            if pet["pet_candy_used"] is None and isinstance(pet_info.get("candyUsed"), (int, float)):
                pet["pet_candy_used"] = int(pet_info["candyUsed"])
            if pet["pet_exp"] is None and isinstance(pet_info.get("exp"), (int, float)):
                pet["pet_exp"] = float(pet_info["exp"])
            if pet["pet_level"] is None:
                level_data = pet_info.get("level")
                if isinstance(level_data, dict) and isinstance(level_data.get("level"), (int, float)):
                    pet["pet_level"] = int(level_data["level"])
                elif isinstance(pet_info.get("level"), (int, float)):
                    pet["pet_level"] = int(pet_info["level"])

    if pet["pet_level"] is None:
        clean_name = auction.get("decoded_clean_name") or auction.get("item_name") or ""
        match = PET_LEVEL_RE.search(clean_name)
        if match:
            pet["pet_level"] = int(match.group(1))

    return pet


def _decoded_item_quality_features(auction: dict) -> dict[str, float]:
    pet = _pet_metadata(auction)
    stars = _safe_numeric(auction.get("decoded_stars"))
    enchants = _safe_numeric(auction.get("decoded_enchant_count"))
    recombobulated = 1.0 if auction.get("decoded_recombobulated") else 0.0
    hot_potato = _safe_numeric(auction.get("decoded_hot_potato_count"))
    fuming = _safe_numeric(auction.get("decoded_fuming_potato_count"))
    has_reforge = 1.0 if auction.get("decoded_reforge") else 0.0
    has_runes = 1.0 if auction.get("decoded_rune_summary") else 0.0
    has_gemstones = 1.0 if auction.get("decoded_gemstone_summary") else 0.0
    has_attributes = 1.0 if auction.get("decoded_attribute_summary") else 0.0
    has_decoded_item = 1.0 if auction.get("decoded_item_id") else 0.0
    pet_level = _safe_numeric(pet.get("pet_level"))
    pet_exp = _safe_numeric(pet.get("pet_exp"))
    has_pet_held_item = 1.0 if pet.get("pet_held_item") else 0.0
    pet_candy_used = _safe_numeric(pet.get("pet_candy_used"))
    is_pet = 1.0 if (auction.get("item_id") == "PET" or pet.get("pet_type")) else 0.0
    is_max_level_pet = 1.0 if pet_level >= 100 else 0.0

    quality_score = (
        0.20 * recombobulated
        + 0.18 * min(stars / 10.0, 1.0)
        + 0.18 * min(enchants / 10.0, 1.0)
        + 0.12 * has_reforge
        + 0.10 * min(hot_potato / 10.0, 1.0)
        + 0.08 * min(fuming / 5.0, 1.0)
        + 0.05 * has_runes
        + 0.05 * has_gemstones
        + 0.04 * has_attributes
        + 0.10 * min(pet_level / 100.0, 1.0)
        + 0.05 * has_pet_held_item
    )

    return {
        "has_decoded_item_meta": has_decoded_item,
        "decoded_stars": stars,
        "decoded_enchant_count": enchants,
        "decoded_recombobulated": recombobulated,
        "decoded_has_reforge": has_reforge,
        "decoded_hot_potato_count": hot_potato,
        "decoded_fuming_potato_count": fuming,
        "decoded_has_runes": has_runes,
        "decoded_has_gemstones": has_gemstones,
        "decoded_has_attributes": has_attributes,
        "is_pet": is_pet,
        "decoded_pet_level": pet_level,
        "decoded_pet_exp": pet_exp,
        "decoded_has_pet_held_item": has_pet_held_item,
        "decoded_pet_candy_used": pet_candy_used,
        "decoded_is_max_level_pet": is_max_level_pet,
        "item_quality_score": quality_score,
    }


def compute_features_single(auction: dict) -> dict:
    """
    Compute the full feature vector for one auction record.
    Makes DB lookups for context — suitable for live inference.
    Returns a flat dict of feature_name -> float.
    """
    features: dict[str, float] = {}
    auction_id  = auction["auction_id"]
    item_id     = auction.get("item_id") or auction.get("item_name")
    seller_uuid = auction["seller_uuid"]
    buyer_uuid  = auction.get("buyer_uuid") or ""
    final_price = float(auction["final_price"])
    tier        = auction.get("tier", "UNKNOWN")
    tier_weight = TIER_MULTIPLIER.get(tier, 1.3)
    features.update(_decoded_item_quality_features(auction))

    # ── 1. Price signals ───────────────────────────────────────────────────────
    history_sources = _history_sources_for_auction(auction)
    history = (
        get_item_price_history(item_id, days=MEDIAN_WINDOW_DAYS, allowed_sources=history_sources)
        if item_id else []
    )
    prices  = [h["final_price"] for h in history if h["auction_id"] != auction_id]
    if features["is_pet"] == 1.0:
        quality_history = _pet_quality_matches(history, auction, auction_id)
    else:
        quality_signature = _quality_signature(auction)
        quality_history = [
            h for h in history
            if h["auction_id"] != auction_id and _matches_quality_signature(h, quality_signature)
        ]
    quality_prices = [h["final_price"] for h in quality_history]

    item_median   = _safe_median(prices)
    item_stdev    = _safe_stdev(prices)
    quality_median = _safe_quality_median(quality_prices, features["is_pet"] == 1.0)
    quality_stdev = _safe_stdev(quality_prices)

    features["quality_match_count"] = float(len(quality_prices))
    features["has_quality_median"] = 1.0 if quality_median and quality_median > 0 else 0.0
    if quality_median and quality_median > 0:
        features["price_to_quality_median_ratio"] = final_price / quality_median
        features["price_vs_quality_median_log"] = float(np.log1p(final_price / quality_median))
    else:
        features["price_to_quality_median_ratio"] = 1.0
        features["price_vs_quality_median_log"] = 0.0

    if quality_median and quality_stdev and quality_stdev > 0:
        features["quality_price_zscore"] = (final_price - quality_median) / quality_stdev
    else:
        features["quality_price_zscore"] = 0.0

    if item_median and item_median > 0:
        features["price_to_median_ratio"]  = final_price / item_median
        features["price_vs_median_log"]    = float(np.log1p(final_price / item_median))
        features["has_item_median"]        = 1.0
    else:
        # No price history yet — keep values neutral and track that context is missing.
        features["price_to_median_ratio"]  = 1.0
        features["price_vs_median_log"]    = 0.0
        features["has_item_median"]        = 0.0

    if item_median and item_stdev and item_stdev > 0:
        features["price_zscore"] = (final_price - item_median) / item_stdev
    else:
        features["price_zscore"] = 0.0

    # lbin_at_time: compare against known lowest BIN at time of sale
    lbin = auction.get("lbin_at_time")
    if lbin and lbin > 0:
        features["price_to_lbin_ratio"] = final_price / float(lbin)
    else:
        features["price_to_lbin_ratio"] = 1.0

    # ── 2. Auction mechanics ───────────────────────────────────────────────────
    bid_count   = int(auction.get("bid_count", 0))
    is_bin      = int(bool(auction.get("is_bin", False)))
    time_to_sell = auction.get("time_to_sell_s")

    features["bid_count"]      = float(bid_count)
    features["is_bin"]         = float(is_bin)
    features["log_bid_count"]  = float(np.log1p(bid_count))

    # Zero bids on a non-BIN expensive auction is suspicious
    features["zero_bids"]      = 1.0 if bid_count == 0 else 0.0

    # Keep as a neutral mechanic for compatibility with older models.
    features["bin_zero_bid"]   = float(is_bin and bid_count == 0)

    if time_to_sell is not None and time_to_sell >= 0:
        features["time_to_sell_s"]    = float(time_to_sell)
        features["log_time_to_sell"]  = float(np.log1p(time_to_sell))
        # Suspicious if sold in under 2 minutes
        features["very_fast_sale"]    = 1.0 if time_to_sell < 120 else 0.0
    else:
        features["time_to_sell_s"]    = -1.0
        features["log_time_to_sell"]  = -1.0
        features["very_fast_sale"]    = 0.0

    # ── 3. Seller history ──────────────────────────────────────────────────────
    seller_hist  = get_seller_history(seller_uuid, days=30, allowed_sources=history_sources)
    seller_prices = [h["final_price"] for h in seller_hist
                     if h["auction_id"] != auction_id]

    seller_median = _safe_median(seller_prices)
    seller_stdev  = _safe_stdev(seller_prices)

    features["seller_sale_count_30d"] = float(len(seller_prices))

    if seller_median and seller_median > 0:
        features["price_to_seller_avg_ratio"] = final_price / seller_median
        features["has_seller_history"]        = 1.0
    else:
        features["price_to_seller_avg_ratio"] = 1.0
        features["has_seller_history"]        = 0.0

    if seller_median and seller_stdev and seller_stdev > 0:
        features["seller_price_zscore"] = (final_price - seller_median) / seller_stdev
    else:
        features["seller_price_zscore"] = 0.0

    # ── 4. Pair signals ────────────────────────────────────────────────────────
    if buyer_uuid:
        pair_count = get_pair_frequency(seller_uuid, buyer_uuid, days=30)
        features["seller_buyer_pair_count_30d"] = float(pair_count)
        # Pair transacting 3+ times in a month is a strong signal
        features["repeat_pair"]                 = 1.0 if pair_count >= 3 else 0.0
    else:
        features["seller_buyer_pair_count_30d"] = 0.0
        features["repeat_pair"]                 = 0.0

    # ── 5. Item context ────────────────────────────────────────────────────────
    features["tier_weight"]     = tier_weight
    features["log_final_price"] = float(np.log1p(final_price))

    # Composite risk score (heuristic, used alongside model score)
    features["heuristic_score"] = _heuristic_score(features)

    return features


def _heuristic_score(f: dict) -> float:
    """
    Simple rule-based score 0-1. Used as a feature and for hard-floor filtering.
    Not the final detector — just adds interpretable signal to the ML model.
    """
    score = 0.0
    weight_total = 0.0

    def add(weight, condition):
        nonlocal score, weight_total
        weight_total += weight
        if condition:
            score += weight

    # Price far above median
    pmr = f.get("price_to_median_ratio", -1)
    if pmr > 0:
        add(0.35, pmr > 10)
        add(0.15, 3 < pmr <= 10)

    # Very fast sale
    add(0.10, f.get("very_fast_sale", 0) == 1)

    # Same pair repeating
    add(0.15, f.get("repeat_pair", 0) == 1)

    # Seller's price way above their own norm
    psar = f.get("price_to_seller_avg_ratio", -1)
    if psar > 0:
        add(0.05, psar > 5)

    return score / weight_total if weight_total > 0 else 0.0


def compute_features_batch(auction_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Compute features for all (or specified) auctions in the DB.
    Returns a DataFrame with auction_id as index and features as columns.
    Much faster than calling compute_features_single in a loop because
    we bulk-load the price histories in one query.
    """
    sql = "SELECT * FROM auctions"
    params: list = []
    if auction_ids:
        placeholders = ",".join("?" * len(auction_ids))
        sql += f" WHERE auction_id IN ({placeholders})"
        params = auction_ids

    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not rows:
        logger.warning("No auctions found in DB for feature computation.")
        return pd.DataFrame()

    logger.info(f"Computing features for {len(rows)} auctions...")

    # Build features row by row. The live path intentionally uses only completed-sale
    # sources for real auctions, while demo rows can fall back to demo history.
    records = []
    for auction in rows:
        try:
            f = compute_features_single(auction)
            f["auction_id"] = auction["auction_id"]
            records.append(f)
        except Exception as e:
            logger.debug(f"Feature error for {auction['auction_id']}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("auction_id")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} features")
    return df


FEATURE_COLUMNS = [
    "price_to_median_ratio",
    "price_vs_median_log",
    "has_item_median",
    "price_zscore",
    "price_to_quality_median_ratio",
    "price_vs_quality_median_log",
    "has_quality_median",
    "quality_price_zscore",
    "quality_match_count",
    "price_to_lbin_ratio",
    "bid_count",
    "is_bin",
    "log_bid_count",
    "zero_bids",
    "bin_zero_bid",
    "time_to_sell_s",
    "log_time_to_sell",
    "very_fast_sale",
    "seller_sale_count_30d",
    "price_to_seller_avg_ratio",
    "has_seller_history",
    "seller_price_zscore",
    "seller_buyer_pair_count_30d",
    "repeat_pair",
    "tier_weight",
    "has_decoded_item_meta",
    "decoded_stars",
    "decoded_enchant_count",
    "decoded_recombobulated",
    "decoded_has_reforge",
    "decoded_hot_potato_count",
    "decoded_fuming_potato_count",
    "decoded_has_runes",
    "decoded_has_gemstones",
    "decoded_has_attributes",
    "is_pet",
    "decoded_pet_level",
    "decoded_pet_exp",
    "decoded_has_pet_held_item",
    "decoded_pet_candy_used",
    "decoded_is_max_level_pet",
    "item_quality_score",
    "log_final_price",
    "heuristic_score",
]


if __name__ == "__main__":
    from database import init_db
    import logging
    logging.basicConfig(level=logging.INFO)
    init_db()
    df = compute_features_batch()
    if not df.empty:
        print(df.describe())
        print("\nTop suspicious by heuristic score:")
        print(df.nlargest(10, "heuristic_score")[["price_to_median_ratio", "bin_zero_bid",
                                                    "very_fast_sale", "heuristic_score"]])
