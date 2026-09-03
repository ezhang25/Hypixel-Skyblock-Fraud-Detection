"""
database.py - SQLite schema setup and all query helpers.

Tables:
  auctions       — raw auction records (historic + live)
  flagged        — auctions flagged by the detector with scores/reasons
  labels         — manual labels for supervised training (once you review flags)
"""

import sqlite3
import logging
import json
from contextlib import contextmanager
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def _backfill_missing_rarities(conn: sqlite3.Connection) -> int:
    """Repair old UNKNOWN tiers from the decoded NBT lore saved in SQLite."""
    from item_decoder import extract_rarity_from_lore

    rows = conn.execute(
        "SELECT auction_id, decoded_item_json FROM auctions WHERE tier IS NULL OR tier = 'UNKNOWN'"
    ).fetchall()
    updates = []
    for row in rows:
        try:
            decoded = json.loads(row["decoded_item_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        rarity = extract_rarity_from_lore(decoded.get("lore") if isinstance(decoded, dict) else None)
        if rarity:
            updates.append((rarity, row["auction_id"]))
    if updates:
        conn.executemany("UPDATE auctions SET tier = ? WHERE auction_id = ?", updates)
    return len(updates)

AUCTION_METADATA_COLUMNS = {
    "decoded_item_json": "TEXT",
    "decoded_item_id": "TEXT",
    "decoded_clean_name": "TEXT",
    "decoded_stars": "INTEGER",
    "decoded_recombobulated": "BOOLEAN",
    "decoded_hot_potato_count": "INTEGER",
    "decoded_fuming_potato_count": "INTEGER",
    "decoded_reforge": "TEXT",
    "decoded_enchant_count": "INTEGER",
    "decoded_enchant_summary": "TEXT",
    "decoded_rune_summary": "TEXT",
    "decoded_gemstone_summary": "TEXT",
    "decoded_attribute_summary": "TEXT",
    "decoded_dungeon_tier": "TEXT",
    "decoded_skin": "TEXT",
    "decoded_dye": "TEXT",
    "decoded_pet_type": "TEXT",
    "decoded_pet_tier": "TEXT",
    "decoded_pet_level": "INTEGER",
    "decoded_pet_exp": "REAL",
    "decoded_pet_held_item": "TEXT",
    "decoded_pet_candy_used": "INTEGER",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS auctions (
    auction_id      TEXT PRIMARY KEY,
    item_name       TEXT NOT NULL,
    item_id         TEXT,                   -- Hypixel internal item ID (most reliable key)
    tier            TEXT,                   -- COMMON/UNCOMMON/RARE/EPIC/LEGENDARY/SPECIAL
    category        TEXT,                   -- weapon/armor/accessory/etc
    seller_uuid     TEXT NOT NULL,
    buyer_uuid      TEXT,
    start_price     INTEGER NOT NULL,       -- coins
    final_price     INTEGER NOT NULL,       -- coins
    item_quantity   INTEGER NOT NULL DEFAULT 1,
    has_item_quantity BOOLEAN NOT NULL DEFAULT FALSE,
    bid_count       INTEGER DEFAULT 0,
    is_bin          BOOLEAN NOT NULL,       -- Buy It Now vs. auction
    started_at      INTEGER,               -- Unix timestamp ms
    ended_at        INTEGER NOT NULL,      -- Unix timestamp ms
    time_to_sell_s  INTEGER,               -- seconds from start to sale
    lbin_at_time    INTEGER,               -- lowest BIN price at time of sale (if known)
    decoded_item_json TEXT,
    decoded_item_id TEXT,
    decoded_clean_name TEXT,
    decoded_stars INTEGER,
    decoded_recombobulated BOOLEAN,
    decoded_hot_potato_count INTEGER,
    decoded_fuming_potato_count INTEGER,
    decoded_reforge TEXT,
    decoded_enchant_count INTEGER,
    decoded_enchant_summary TEXT,
    decoded_rune_summary TEXT,
    decoded_gemstone_summary TEXT,
    decoded_attribute_summary TEXT,
    decoded_dungeon_tier TEXT,
    decoded_skin TEXT,
    decoded_dye TEXT,
    decoded_pet_type TEXT,
    decoded_pet_tier TEXT,
    decoded_pet_level INTEGER,
    decoded_pet_exp REAL,
    decoded_pet_held_item TEXT,
    decoded_pet_candy_used INTEGER,
    ingested_at     INTEGER NOT NULL,      -- when we stored it
    source          TEXT DEFAULT 'ended'   -- 'ended' | 'live'
);

CREATE INDEX IF NOT EXISTS idx_item_id   ON auctions(item_id);
CREATE INDEX IF NOT EXISTS idx_item_name ON auctions(item_name);
CREATE INDEX IF NOT EXISTS idx_seller    ON auctions(seller_uuid);
CREATE INDEX IF NOT EXISTS idx_ended_at  ON auctions(ended_at);

CREATE TABLE IF NOT EXISTS flagged (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    auction_id      TEXT NOT NULL REFERENCES auctions(auction_id),
    flagged_at      INTEGER NOT NULL,
    model_version   TEXT NOT NULL,
    anomaly_score   REAL,                   -- raw Isolation Forest score
    fraud_prob      REAL,                   -- LightGBM probability (null if no supervised model)
    tier            TEXT,                   -- HIGH / MEDIUM / LOW
    reasons         TEXT,                   -- JSON array of human-readable reasons
    reviewed        BOOLEAN DEFAULT FALSE,
    UNIQUE(auction_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_flagged_tier ON flagged(tier);
CREATE INDEX IF NOT EXISTS idx_flagged_rev  ON flagged(reviewed);

CREATE TABLE IF NOT EXISTS labels (
    auction_id      TEXT PRIMARY KEY REFERENCES auctions(auction_id),
    label           INTEGER NOT NULL,       -- 1 = confirmed IRL trade, 0 = false positive
    labeled_at      INTEGER NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_metrics (
    metric_name     TEXT PRIMARY KEY,
    metric_value    INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
"""


@contextmanager
def get_conn():
    """Context manager that yields a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(config.get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(auctions)").fetchall()
        }
        for column, column_type in AUCTION_METADATA_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE auctions ADD COLUMN {column} {column_type}")
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(auctions)").fetchall()
        }
        if "item_quantity" not in existing:
            conn.execute("ALTER TABLE auctions ADD COLUMN item_quantity INTEGER NOT NULL DEFAULT 1")
        if "has_item_quantity" not in existing:
            conn.execute("ALTER TABLE auctions ADD COLUMN has_item_quantity BOOLEAN NOT NULL DEFAULT FALSE")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decoded_item_id ON auctions(decoded_item_id)")
        repaired_rarities = _backfill_missing_rarities(conn)
        # Seed existing databases once. Afterwards insert_auctions maintains this
        # counter independently from retention cleanup of old auction rows.
        conn.execute("""
            INSERT OR IGNORE INTO ingestion_metrics (metric_name, metric_value, updated_at)
            SELECT 'all_time_auctions_ingested', COUNT(*), CAST(strftime('%s', 'now') AS INTEGER) * 1000
            FROM auctions
        """)
    logger.info(f"Database initialised at {config.get_db_path()}")
    if repaired_rarities:
        logger.info("Repaired rarity tier from NBT lore for %s existing auctions", repaired_rarities)


def insert_auctions(rows: list[dict]) -> int:
    """
    Bulk-insert auction records. Ignores duplicates (auction_id is PK).
    Returns the number of newly inserted rows.
    """
    if not rows:
        return 0

    optional_columns = tuple(AUCTION_METADATA_COLUMNS) + (
        "item_id",
        "buyer_uuid",
        "started_at",
        "time_to_sell_s",
        "lbin_at_time",
    )
    prepared_rows = []
    for row in rows:
        prepared = dict(row)
        for column in optional_columns:
            prepared.setdefault(column, None)
        # Quantity is decoded from the NBT item stack. Older callers and
        # historical records did not have it, so preserve that uncertainty.
        prepared.setdefault("item_quantity", 1)
        prepared.setdefault("has_item_quantity", False)
        prepared_rows.append(prepared)

    sql = """
        INSERT OR IGNORE INTO auctions
            (auction_id, item_name, item_id, tier, category,
             seller_uuid, buyer_uuid, start_price, final_price, item_quantity, has_item_quantity,
             bid_count, is_bin, started_at, ended_at, time_to_sell_s,
             lbin_at_time, decoded_item_json, decoded_item_id,
             decoded_clean_name, decoded_stars, decoded_recombobulated,
             decoded_hot_potato_count, decoded_fuming_potato_count, decoded_reforge,
             decoded_enchant_count, decoded_enchant_summary, decoded_rune_summary,
             decoded_gemstone_summary, decoded_attribute_summary, decoded_dungeon_tier,
             decoded_skin, decoded_dye, decoded_pet_type, decoded_pet_tier,
             decoded_pet_level, decoded_pet_exp, decoded_pet_held_item,
             decoded_pet_candy_used, ingested_at, source)
        VALUES
            (:auction_id, :item_name, :item_id, :tier, :category,
             :seller_uuid, :buyer_uuid, :start_price, :final_price, :item_quantity, :has_item_quantity,
             :bid_count, :is_bin, :started_at, :ended_at, :time_to_sell_s,
             :lbin_at_time, :decoded_item_json, :decoded_item_id,
             :decoded_clean_name, :decoded_stars, :decoded_recombobulated,
             :decoded_hot_potato_count, :decoded_fuming_potato_count, :decoded_reforge,
             :decoded_enchant_count, :decoded_enchant_summary, :decoded_rune_summary,
             :decoded_gemstone_summary, :decoded_attribute_summary, :decoded_dungeon_tier,
             :decoded_skin, :decoded_dye, :decoded_pet_type, :decoded_pet_tier,
             :decoded_pet_level, :decoded_pet_exp, :decoded_pet_held_item,
             :decoded_pet_candy_used, :ingested_at, :source)
    """
    with get_conn() as conn:
        cursor = conn.executemany(sql, prepared_rows)
        inserted = max(cursor.rowcount, 0)
        if inserted:
            conn.execute("""
                INSERT INTO ingestion_metrics (metric_name, metric_value, updated_at)
                VALUES ('all_time_auctions_ingested', ?, CAST(strftime('%s', 'now') AS INTEGER) * 1000)
                ON CONFLICT(metric_name) DO UPDATE SET
                    metric_value = metric_value + excluded.metric_value,
                    updated_at = excluded.updated_at
            """, (inserted,))
        return inserted


def _source_clause(allowed_sources: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    placeholders = ",".join("?" * len(allowed_sources))
    return f"({placeholders})", allowed_sources


def get_item_price_history(
    item_id: str,
    days: int = 7,
    allowed_sources: tuple[str, ...] = ("ended",),
) -> list[dict]:
    """Returns all sales for an item within the last N days."""
    cutoff_ms = int((__import__("time").time() - days * 86400) * 1000)
    source_sql, source_params = _source_clause(allowed_sources)
    sql = """
        SELECT auction_id, item_name, tier, final_price, item_quantity, has_item_quantity,
               ended_at, bid_count, is_bin, source,
               item_id, decoded_item_id, decoded_clean_name, decoded_item_json,
               decoded_stars, decoded_recombobulated, decoded_enchant_count,
               decoded_reforge, decoded_pet_type, decoded_pet_tier,
               decoded_pet_level, decoded_pet_exp, decoded_pet_held_item,
               decoded_pet_candy_used
        FROM auctions
        WHERE item_id = ? AND ended_at >= ? AND source IN
    """ + source_sql + """
        ORDER BY ended_at DESC
    """
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(sql, (item_id, cutoff_ms, *source_params)).fetchall()
        ]


def get_seller_history(
    seller_uuid: str,
    days: int = 30,
    allowed_sources: tuple[str, ...] = ("ended",),
) -> list[dict]:
    """Returns recent sales by a seller for baseline comparison."""
    cutoff_ms = int((__import__("time").time() - days * 86400) * 1000)
    source_sql, source_params = _source_clause(allowed_sources)
    sql = """
        SELECT auction_id, item_name, final_price, item_quantity, has_item_quantity,
               ended_at, bid_count, is_bin, source
        FROM auctions
        WHERE seller_uuid = ? AND ended_at >= ? AND source IN
    """ + source_sql + """
        ORDER BY ended_at DESC
    """
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(sql, (seller_uuid, cutoff_ms, *source_params)).fetchall()
        ]


def get_pair_frequency(seller_uuid: str, buyer_uuid: str, days: int = 30) -> int:
    """How many times has this seller/buyer pair transacted recently?"""
    cutoff_ms = int((__import__("time").time() - days * 86400) * 1000)
    sql = """
        SELECT COUNT(*) FROM auctions
        WHERE seller_uuid = ? AND buyer_uuid = ? AND ended_at >= ?
    """
    with get_conn() as conn:
        return conn.execute(sql, (seller_uuid, buyer_uuid, cutoff_ms)).fetchone()[0]


def insert_flag(record: dict) -> None:
    """Insert or update a flagged auction record unless it was labeled normal."""
    with get_conn() as conn:
        existing_label = conn.execute(
            "SELECT label FROM labels WHERE auction_id = ?",
            (record["auction_id"],),
        ).fetchone()
        if existing_label is not None and existing_label[0] == 0:
            logger.info(
                "Skipping flag insert for %s because it is labeled false positive",
                record["auction_id"],
            )
            return

        existing_flag = conn.execute(
            "SELECT reviewed FROM flagged WHERE auction_id = ? AND model_version = ?",
            (record["auction_id"], record["model_version"]),
        ).fetchone()
        reviewed = bool(existing_flag[0]) if existing_flag is not None else False

        sql = """
            INSERT OR REPLACE INTO flagged
                (auction_id, flagged_at, model_version, anomaly_score,
                 fraud_prob, tier, reasons, reviewed)
            VALUES
                (:auction_id, :flagged_at, :model_version, :anomaly_score,
                 :fraud_prob, :tier, :reasons, :reviewed)
        """
        payload = dict(record)
        payload["reviewed"] = reviewed
        conn.execute(sql, payload)


def reset_flagged_queue() -> int:
    """
    Clear the current flagged queue while preserving labels.
    Useful before a full rescore after retraining so dashboard stats reflect
    the latest model rather than a mix of old and new flag sets.
    """
    with get_conn() as conn:
        conn.execute("DELETE FROM flagged")
        removed = conn.execute("SELECT changes()").fetchone()[0]
    logger.info("Reset flagged queue, removed %s existing flagged rows", removed)
    return removed


def get_unreviewed_flags(tier: str | None = None, limit: int = 50) -> list[dict]:
    """Fetch flagged auctions awaiting review, newest first."""
    where = "WHERE f.reviewed = FALSE"
    params: list = []
    if tier:
        where += " AND f.tier = ?"
        params.append(tier)

    sql = f"""
        SELECT f.id, f.auction_id, f.flagged_at, f.tier,
               f.anomaly_score, f.fraud_prob, f.reasons,
               a.item_name, a.item_id, a.tier AS item_rarity, a.category,
               a.final_price, a.item_quantity, a.has_item_quantity,
               a.seller_uuid, a.buyer_uuid, a.bid_count, a.is_bin,
               a.decoded_clean_name, a.decoded_stars, a.decoded_recombobulated,
               a.decoded_hot_potato_count, a.decoded_fuming_potato_count,
               a.decoded_reforge, a.decoded_dungeon_tier, a.decoded_skin, a.decoded_dye,
               a.decoded_enchant_summary, a.decoded_rune_summary,
               a.decoded_gemstone_summary, a.decoded_attribute_summary,
               a.decoded_item_json, a.decoded_pet_type, a.decoded_pet_tier,
               a.decoded_pet_level, a.decoded_pet_exp, a.decoded_pet_held_item,
               a.decoded_pet_candy_used
        FROM flagged f
        JOIN auctions a ON f.auction_id = a.auction_id
        {where}
        ORDER BY f.flagged_at DESC
        LIMIT ?
    """
    params.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def is_unreviewed_flag(auction_id: str) -> bool:
    """Return whether an auction is still eligible for review."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM flagged WHERE auction_id = ? AND reviewed = FALSE",
            (auction_id,),
        ).fetchone()
    return row is not None


def label_auction(auction_id: str, label: int, notes: str = "") -> None:
    """Mark a flagged auction as confirmed IRL trade (1) or false positive (0)."""
    import time
    sql = """
        INSERT OR REPLACE INTO labels (auction_id, label, labeled_at, notes)
        VALUES (?, ?, ?, ?)
    """
    with get_conn() as conn:
        conn.execute(sql, (auction_id, label, int(time.time() * 1000), notes))
        if label == 0:
            conn.execute("DELETE FROM flagged WHERE auction_id = ?", (auction_id,))
        else:
            conn.execute("UPDATE flagged SET reviewed = TRUE WHERE auction_id = ?", (auction_id,))
    logger.info(f"Labelled {auction_id} as {'IRL TRADE' if label else 'false positive'}")


def get_labeled_dataset() -> list[dict]:
    """Return labeled auctions joined with their features for supervised training."""
    sql = """
        SELECT a.*, l.label
        FROM auctions a
        JOIN labels l ON a.auction_id = l.auction_id
    """
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def db_stats() -> dict:
    """Quick counts for each table — useful for monitoring."""
    with get_conn() as conn:
        retained_auctions = conn.execute("SELECT COUNT(*) FROM auctions").fetchone()[0]
        all_time_row = conn.execute("""
            SELECT metric_value
            FROM ingestion_metrics
            WHERE metric_name = 'all_time_auctions_ingested'
        """).fetchone()
        return {
            "total_auctions":     retained_auctions,
            "all_time_auctions_ingested": all_time_row[0] if all_time_row else retained_auctions,
            "flagged_total":      conn.execute("SELECT COUNT(*) FROM flagged").fetchone()[0],
            "flagged_unreviewed": conn.execute("SELECT COUNT(*) FROM flagged WHERE reviewed=FALSE").fetchone()[0],
            "labeled":            conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
            "confirmed_irl":      conn.execute("SELECT COUNT(*) FROM labels WHERE label=1").fetchone()[0],
        }


if __name__ == "__main__":
    init_db()
    print("DB stats:", db_stats())
