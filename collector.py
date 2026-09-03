"""
collector.py - Hypixel auction house data ingestion.

Two modes:
  --mode backfill   Pull pages of recently ended auctions to build training data.
  --mode live       Poll every 60s for new ended auctions (run continuously).

Usage:
  python collector.py --mode backfill --pages 200
  python collector.py --mode live
"""

import argparse
import json
import logging
import time
from datetime import datetime

import config
from runtime import log_runtime_environment
from item_decoder import decode_item_bytes

log_runtime_environment(
    logging.getLogger("collector.bootstrap"),
    "collector.py",
    required_modules=("requests",),
)

import requests

from config import (
    ENDPOINTS,
    POLL_INTERVAL_SECONDS,
    REQUEST_DELAY_SECONDS,
    LOG_DIR,
)
from database import init_db, insert_auctions, db_stats

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "collector.log"),
    ],
)
logger = logging.getLogger("collector")
_ITEM_DECODE_WARNING_LIMIT = 10
_item_decode_warning_count = 0

# ── Helpers ────────────────────────────────────────────────────────────────────


def _log_decode_warning(auction_id: str | None, error: str | None) -> None:
    global _item_decode_warning_count
    if not error or _item_decode_warning_count >= _ITEM_DECODE_WARNING_LIMIT:
        return
    _item_decode_warning_count += 1
    logger.warning(
        "Item metadata decode failed for auction %s: %s",
        auction_id or "unknown",
        error,
    )
    if _item_decode_warning_count == _ITEM_DECODE_WARNING_LIMIT:
        logger.warning("Further item decode warnings will be suppressed for this run.")


def _metadata_columns(decoded_item: dict | None) -> dict:
    if not decoded_item:
        return {
            "decoded_item_json": None,
            "decoded_item_id": None,
            "decoded_clean_name": None,
            "decoded_stars": None,
            "decoded_recombobulated": None,
            "decoded_hot_potato_count": None,
            "decoded_fuming_potato_count": None,
            "decoded_reforge": None,
            "decoded_enchant_count": None,
            "decoded_enchant_summary": None,
            "decoded_rune_summary": None,
            "decoded_gemstone_summary": None,
            "decoded_attribute_summary": None,
            "decoded_dungeon_tier": None,
            "decoded_skin": None,
            "decoded_dye": None,
            "decoded_pet_type": None,
            "decoded_pet_tier": None,
            "decoded_pet_level": None,
            "decoded_pet_exp": None,
            "decoded_pet_held_item": None,
            "decoded_pet_candy_used": None,
            "item_quantity": 1,
            "has_item_quantity": False,
        }

    pet = decoded_item.get("pet") or {}
    return {
        "decoded_item_json": json.dumps(decoded_item, sort_keys=True),
        "decoded_item_id": decoded_item.get("item_id"),
        "decoded_clean_name": decoded_item.get("clean_name"),
        "decoded_stars": decoded_item.get("stars"),
        "decoded_recombobulated": decoded_item.get("recombobulated"),
        "decoded_hot_potato_count": decoded_item.get("hot_potato_count"),
        "decoded_fuming_potato_count": decoded_item.get("fuming_potato_count"),
        "decoded_reforge": decoded_item.get("reforge"),
        "decoded_enchant_count": decoded_item.get("enchant_count"),
        "decoded_enchant_summary": decoded_item.get("enchant_summary"),
        "decoded_rune_summary": decoded_item.get("rune_summary"),
        "decoded_gemstone_summary": decoded_item.get("gemstone_summary"),
        "decoded_attribute_summary": decoded_item.get("attribute_summary"),
        "decoded_dungeon_tier": decoded_item.get("dungeon_tier"),
        "decoded_skin": decoded_item.get("skin"),
        "decoded_dye": decoded_item.get("dye"),
        "decoded_pet_type": pet.get("pet_type"),
        "decoded_pet_tier": pet.get("pet_tier"),
        "decoded_pet_level": pet.get("pet_level"),
        "decoded_pet_exp": pet.get("pet_exp"),
        "decoded_pet_held_item": pet.get("pet_held_item"),
        "decoded_pet_candy_used": pet.get("pet_candy_used"),
        "item_quantity": max(1, int(decoded_item.get("item_count") or 1)),
        "has_item_quantity": True,
    }


def _normalise_ended(raw: dict) -> dict | None:
    """
    Map a raw auction object from the Hypixel API into our DB schema.
    Returns None if the record is incomplete/malformed.
    """
    try:
        auction_id  = raw.get("auction_id") or raw.get("uuid") or raw.get("id") or raw.get("_id")
        seller_uuid = raw.get("seller") or raw.get("auctioneer")
        final_price = raw.get("price") or raw.get("highest_bid_amount", 0)

        if not auction_id or not seller_uuid or not final_price:
            return None

        decoded_item, decode_error = decode_item_bytes(raw.get("item_bytes"))
        if decode_error not in (None, "item_bytes missing"):
            _log_decode_warning(str(auction_id), decode_error)

        item_name = (
            raw.get("item_name")
            or (decoded_item or {}).get("clean_name")
            or raw.get("item_lore")
            or "unknown"
        )

        started_at  = raw.get("start")
        ended_at    = raw.get("end") or raw.get("timestamp")
        time_to_sell = None
        if started_at and ended_at:
            time_to_sell = max(0, int((ended_at - started_at) / 1000))

        metadata = _metadata_columns(decoded_item)
        bids = raw.get("bids", [])
        bid_count = int(bids) if isinstance(bids, int) else len(bids) if isinstance(bids, list) else 0

        return {
            "auction_id":    str(auction_id),
            "item_name":     str(item_name)[:256],
            "item_id":       metadata["decoded_item_id"],
            "item_quantity": metadata["item_quantity"],
            "has_item_quantity": metadata["has_item_quantity"],
            "tier":          raw.get("tier", "UNKNOWN"),
            "category":      raw.get("category", "misc"),
            "seller_uuid":   str(seller_uuid),
            "buyer_uuid":    str(raw.get("buyer", "")) or None,
            "start_price":   int(raw.get("starting_bid", 0)),
            "final_price":   int(final_price),
            "bid_count":     bid_count,
            "is_bin":        bool(raw.get("bin", False)),
            "started_at":    int(started_at) if started_at else None,
            "ended_at":      int(ended_at),
            "time_to_sell_s": time_to_sell,
            "lbin_at_time":  None,   # populated separately if we have the data
            "ingested_at":   int(time.time() * 1000),
            "source":        "ended",
            **metadata,
        }
    except Exception as e:
        logger.debug(f"Failed to normalise auction: {e}")
        return None


# ── API calls ──────────────────────────────────────────────────────────────────

def fetch_ended_auctions() -> list[dict]:
    """
    Fetch the most recently ended auctions from the Hypixel API.
    This endpoint returns all auctions that ended since the last call
    (it's incremental — no pagination needed for live mode).
    """
    try:
        resp = requests.get(
            ENDPOINTS["auctions_ended"],
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.warning(f"API returned success=false: {data.get('cause', 'unknown')}")
            return []
        auctions = data.get("auctions", [])
        if not auctions:
            logger.info("Ended-auctions API returned no new completed auctions this poll.")
        return auctions
    except requests.RequestException as e:
        logger.error(f"Failed to fetch ended auctions due to request error: {e}")
        return []


def fetch_live_auctions_page(page: int = 0) -> tuple[list[dict], int]:
    """
    Fetch one page of currently active auctions.
    Returns (auctions_list, total_pages).
    The live auction endpoint requires an API key and is paginated.
    """
    try:
        resp = requests.get(
            ENDPOINTS["auctions_live"],
            params={"page": page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.warning(f"Live API page {page} returned success=false")
            return [], 0
        auctions = data.get("auctions", [])
        if not auctions:
            logger.info(f"Live API page {page} returned zero auctions.")
        return auctions, data.get("totalPages", 1)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch live auctions page {page} due to request error: {e}")
        return [], 0


# ── Ingestion modes ────────────────────────────────────────────────────────────

def backfill(max_pages: int = 100) -> None:
    """
    Pull multiple pages of the live auction endpoint to build historical data.
    Each page has ~1000 auctions. 100 pages ≈ 100k auctions.
    Note: the live endpoint only shows currently active auctions;
    use ended endpoint for completed sales history.
    """
    logger.info(f"Starting backfill — fetching up to {max_pages} pages of live auctions")
    total_new = 0

    # First pull the ended auctions (best source of completed sales)
    logger.info("Fetching recently ended auctions...")
    raw_ended = fetch_ended_auctions()
    rows = [r for raw in raw_ended if (r := _normalise_ended(raw)) is not None]
    new = insert_auctions(rows)
    total_new += new
    logger.info(f"  Ended auctions: {len(raw_ended)} fetched, {new} new")

    # Then walk live auction pages for active listings
    _, total_pages = fetch_live_auctions_page(0)
    pages_to_fetch = min(max_pages, total_pages)
    logger.info(f"  Live auction pages available: {total_pages}, fetching: {pages_to_fetch}")

    for page in range(pages_to_fetch):
        raw_live, _ = fetch_live_auctions_page(page)
        # Treat live auctions as "ended" once they have a highest bid
        completed = [r for r in raw_live if r.get("highest_bid_amount", 0) > 0]
        rows = []
        for raw in completed:
            normalised = _normalise_ended(raw)
            if normalised:
                normalised["source"] = "live"
                rows.append(normalised)

        new = insert_auctions(rows)
        total_new += new

        if page % 10 == 0:
            logger.info(f"  Page {page}/{pages_to_fetch} — {new} new this page, {total_new} total")

        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(f"Backfill complete. Total new auctions inserted: {total_new}")
    logger.info(f"DB stats: {db_stats()}")


def live_poll() -> None:
    """
    Continuously poll the ended auctions endpoint every POLL_INTERVAL_SECONDS.
    Designed to run forever as a background process.
    """
    logger.info(f"Starting live poll — interval: {POLL_INTERVAL_SECONDS}s")
    logger.info("Press Ctrl+C to stop.")

    poll_count = 0
    while True:
        try:
            start = time.time()
            raw = fetch_ended_auctions()
            rows = [r for raw_a in raw if (r := _normalise_ended(raw_a)) is not None]
            new = insert_auctions(rows)
            elapsed = time.time() - start
            poll_count += 1

            logger.info(
                f"[Poll #{poll_count}] {len(raw)} auctions fetched, "
                f"{new} new — {elapsed:.1f}s"
            )

            if poll_count % 10 == 0:
                logger.info(f"DB stats: {db_stats()}")

            # Sleep for the remainder of the interval
            sleep_time = max(1, POLL_INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Live poll stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in poll loop: {e}", exc_info=True)
            time.sleep(30)   # back off on unexpected errors


# ── Demo mode: generate synthetic data ────────────────────────────────────────

def generate_demo_data(n_normal: int = 2000, n_suspicious: int = 50) -> None:
    """
    Generate synthetic auction data so you can develop the model
    without needing a live API key. Inserts into the same DB.
    """
    import random
    import uuid
    logger.info(f"Generating {n_normal} normal + {n_suspicious} suspicious demo auctions...")

    items = [
        ("HYPERION", "Hyperion", "LEGENDARY", "weapon",    80_000_000),
        ("SCYLLA",   "Scylla",   "LEGENDARY", "weapon",    60_000_000),
        ("REAPER_SCYTHE", "Reaper Scythe", "LEGENDARY", "weapon", 45_000_000),
        ("LIVID_DAGGER", "Livid Dagger", "LEGENDARY", "weapon", 40_000_000),
        ("NECRON_BLADE", "Necron's Blade", "LEGENDARY", "weapon", 50_000_000),
        ("SHADOW_FURY", "Shadow Fury", "LEGENDARY", "weapon", 55_000_000),
        ("MIDAS_STAFF", "Midas Staff", "LEGENDARY", "weapon", 30_000_000),
        ("ASPECT_OF_THE_DRAGONS", "AOTD", "EPIC", "weapon", 5_000_000),
        ("GIANTS_SWORD", "Giant's Sword", "LEGENDARY", "weapon", 20_000_000),
        ("ATOMSPLIT_KATANA", "Atomsplit Katana", "LEGENDARY", "weapon", 70_000_000),
    ]

    sellers = [str(uuid.uuid4()) for _ in range(100)]
    buyers  = [str(uuid.uuid4()) for _ in range(200)]

    now_ms = int(time.time() * 1000)
    rows = []

    # Normal auctions: price within ±40% of baseline, multiple bids
    for _ in range(n_normal):
        item_id, item_name, tier, cat, base_price = random.choice(items)
        price = int(base_price * random.uniform(0.6, 1.4))
        start_ms = now_ms - random.randint(60_000, 7 * 86_400_000)
        end_ms   = start_ms + random.randint(60_000, 2 * 86_400_000)
        rows.append({
            "auction_id":    str(uuid.uuid4()),
            "item_name":     item_name,
            "item_id":       item_id,
            "tier":          tier,
            "category":      cat,
            "seller_uuid":   random.choice(sellers),
            "buyer_uuid":    random.choice(buyers),
            "start_price":   int(price * 0.8),
            "final_price":   price,
            "item_quantity": 1,
            "has_item_quantity": True,
            "bid_count":     random.randint(1, 20),
            "is_bin":        random.random() < 0.4,
            "started_at":    start_ms,
            "ended_at":      end_ms,
            "time_to_sell_s": int((end_ms - start_ms) / 1000),
            "lbin_at_time":  int(base_price * random.uniform(0.9, 1.1)),
            "ingested_at":   now_ms,
            "source":        "demo",
        })

    # Suspicious auctions: price 5-50× median, 0-1 bids, BIN, fast sale
    irl_seller  = str(uuid.uuid4())   # repeat offender
    irl_buyer   = str(uuid.uuid4())   # same buyer each time
    for _ in range(n_suspicious):
        item_id, item_name, tier, cat, base_price = random.choice(items)
        multiplier = random.uniform(5, 50)
        price    = int(base_price * multiplier)
        start_ms = now_ms - random.randint(60_000, 3 * 86_400_000)
        end_ms   = start_ms + random.randint(5_000, 120_000)   # very fast
        rows.append({
            "auction_id":    str(uuid.uuid4()),
            "item_name":     item_name,
            "item_id":       item_id,
            "tier":          tier,
            "category":      cat,
            "seller_uuid":   irl_seller,
            "buyer_uuid":    irl_buyer,
            "start_price":   price,
            "final_price":   price,
            "item_quantity": 1,
            "has_item_quantity": True,
            "bid_count":     random.randint(0, 1),
            "is_bin":        True,
            "started_at":    start_ms,
            "ended_at":      end_ms,
            "time_to_sell_s": int((end_ms - start_ms) / 1000),
            "lbin_at_time":  int(base_price * random.uniform(0.9, 1.1)),
            "ingested_at":   now_ms,
            "source":        "demo",
        })

    random.shuffle(rows)
    inserted = insert_auctions(rows)
    logger.info(f"Demo data: {inserted} auctions inserted. DB stats: {db_stats()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypixel auction house collector")
    parser.add_argument(
        "--mode",
        choices=["backfill", "live", "demo"],
        default="demo",
        help="backfill: pull historical data | live: continuous poll | demo: synthetic data",
    )
    parser.add_argument("--pages", type=int, default=50, help="Pages to fetch in backfill mode")
    args = parser.parse_args()
    config.set_storage_context("demo" if args.mode == "demo" else "real")
    init_db()
    logger.info("Using storage %s", config.describe_storage_context())

    if args.mode == "backfill":
        backfill(max_pages=args.pages)
    elif args.mode == "live":
        live_poll()
    elif args.mode == "demo":
        generate_demo_data()
