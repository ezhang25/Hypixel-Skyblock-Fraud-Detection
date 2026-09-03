"""
dashboard.py - CLI review dashboard for flagged auctions.

Lets you:
  - Browse flagged auctions by tier
  - View reasons and context for each flag
  - Label them as confirmed IRL trades or false positives
  - See aggregate stats and trends

Usage:
  python dashboard.py              # Interactive review mode
  python dashboard.py --stats      # Print summary stats only
  python dashboard.py --tier HIGH  # Show only HIGH tier flags
"""

import argparse
import json
import logging
import re
from datetime import datetime

import config
from runtime import log_runtime_environment
from database import (
    init_db,
    get_unreviewed_flags,
    label_auction,
    db_stats,
    get_conn,
)

logger = logging.getLogger(__name__)
log_runtime_environment(logger, "dashboard.py")

TIER_COLORS = {
    "HIGH":   "\033[91m",   # red
    "MEDIUM": "\033[93m",   # yellow
    "LOW":    "\033[96m",   # cyan
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
PET_LEVEL_RE = re.compile(r"\[Lvl\s+(\d+)\]", re.IGNORECASE)


def _fmt_coins(n: int) -> str:
    """Format coin amounts readably: 1234567 → 1.23M"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _decode_item_json(flag: dict) -> dict:
    blob = flag.get("decoded_item_json")
    if not isinstance(blob, str) or not blob:
        return {}
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _pet_summary(flag: dict) -> dict:
    pet = {
        "pet_type": flag.get("decoded_pet_type"),
        "pet_tier": flag.get("decoded_pet_tier"),
        "pet_level": flag.get("decoded_pet_level"),
        "pet_held_item": flag.get("decoded_pet_held_item"),
        "pet_candy_used": flag.get("decoded_pet_candy_used"),
    }

    decoded_item = _decode_item_json(flag)
    decoded_pet = decoded_item.get("pet")
    if isinstance(decoded_pet, dict):
        for key in pet:
            if pet[key] is None and decoded_pet.get(key) is not None:
                pet[key] = decoded_pet.get(key)

    if pet["pet_level"] is None:
        clean_name = flag.get("decoded_clean_name") or flag.get("item_name") or ""
        match = PET_LEVEL_RE.search(clean_name)
        if match:
            pet["pet_level"] = int(match.group(1))

    return pet


def _format_pet_type(pet_type: str | None, fallback_name: str) -> str:
    if not pet_type:
        return fallback_name
    return pet_type.replace("_", " ").title()


def print_stats() -> None:
    stats = db_stats()
    print(f"\n{BOLD}══ IRL Trade Detector — Database Stats ══{RESET}")
    print(f"  All-time auctions ingested: {stats['all_time_auctions_ingested']:>8,}")
    print(f"  Auctions currently retained: {stats['total_auctions']:>7,}")
    print(f"  Total flags raised:       {stats['flagged_total']:>10,}")
    print(f"  Awaiting review:          {stats['flagged_unreviewed']:>10,}")
    print(f"  Manually labeled:         {stats['labeled']:>10,}")
    print(f"  Confirmed IRL trades:     {stats['confirmed_irl']:>10,}  "
          f"{GREEN}{'▲' if stats['confirmed_irl'] else ''}{RESET}")

    # Tier breakdown
    with get_conn() as conn:
        for tier in ("HIGH", "MEDIUM", "LOW"):
            c = conn.execute(
                "SELECT COUNT(*) FROM flagged WHERE tier=? AND reviewed=FALSE",
                (tier,)
            ).fetchone()[0]
            color = TIER_COLORS.get(tier, "")
            print(f"  {color}[{tier:<6}]{RESET} unreviewed:      {c:>10,}")
    print()


def print_flag(flag: dict, index: int, total: int) -> None:
    tier    = flag["tier"]
    color   = TIER_COLORS.get(tier, "")
    reasons = json.loads(flag.get("reasons", "[]"))
    pet = _pet_summary(flag)
    is_pet = flag.get("item_id") == "PET" or bool(pet.get("pet_type"))

    print(f"\n{BOLD}──────────────────────────────────────────────────────{RESET}")
    print(f"  {BOLD}[{index}/{total}]{RESET}  Auction: {DIM}{flag['auction_id']}{RESET}")
    print(f"  {color}{BOLD}TIER: {tier}{RESET}  "
          f"Anomaly score: {flag.get('anomaly_score', 0):.3f}  "
          + (f"Fraud prob: {flag.get('fraud_prob', 0):.1%}" if flag.get("fraud_prob") else ""))
    print(f"  {BOLD}Item:{RESET}    {flag['item_name']}")
    if flag.get("decoded_clean_name") and flag["decoded_clean_name"] != flag["item_name"]:
        print(f"  {BOLD}Decoded:{RESET} {flag['decoded_clean_name']}")
    if is_pet:
        pet_bits = []
        pet_tier = pet.get("pet_tier") or flag.get("tier")
        pet_name = _format_pet_type(pet.get("pet_type"), flag.get("decoded_clean_name") or flag["item_name"])
        pet_label = f"{pet_tier} {pet_name}".strip() if pet_tier else pet_name
        if pet.get("pet_level") is not None:
            pet_label += f" (Lvl {int(pet['pet_level'])})"
        pet_bits.append(pet_label)
        if pet.get("pet_held_item"):
            pet_bits.append(f"Held item: {pet['pet_held_item']}")
        if pet.get("pet_candy_used") is not None:
            pet_bits.append(f"Candy used: {int(pet['pet_candy_used'])}")
        print(f"  {BOLD}Pet:{RESET}     " + " | ".join(pet_bits))
    quantity = int(flag.get("item_quantity") or 1)
    total_price = float(flag["final_price"])
    if quantity > 1:
        print(f"  {BOLD}Price:{RESET}   {_fmt_coins(total_price)} coins total × {quantity:,} ({_fmt_coins(total_price / quantity)} each)")
    else:
        print(f"  {BOLD}Price:{RESET}   {_fmt_coins(total_price)} coins")
    detail_bits = []
    if flag.get("decoded_reforge"):
        detail_bits.append(f"Reforge: {flag['decoded_reforge']}")
    if flag.get("decoded_stars") is not None:
        detail_bits.append(f"Stars: {flag['decoded_stars']}")
    if detail_bits:
        print(f"  {BOLD}Traits:{RESET}  " + " | ".join(detail_bits))
    if flag.get("decoded_enchant_summary"):
        print(f"  {BOLD}Enchants:{RESET} {flag['decoded_enchant_summary']}")
    if flag.get("decoded_rune_summary"):
        print(f"  {BOLD}Runes:{RESET}   {flag['decoded_rune_summary']}")
    if flag.get("decoded_gemstone_summary"):
        print(f"  {BOLD}Gems:{RESET}    {flag['decoded_gemstone_summary']}")
    if flag.get("decoded_attribute_summary"):
        print(f"  {BOLD}Attrs:{RESET}   {flag['decoded_attribute_summary']}")
    print(f"  {BOLD}Seller:{RESET}  {flag['seller_uuid']}")
    if flag.get("buyer_uuid"):
        print(f"  {BOLD}Buyer:{RESET}   {flag['buyer_uuid']}")
    print(f"  {BOLD}BIN:{RESET}     {'Yes' if flag['is_bin'] else 'No'}  "
          f"  {BOLD}Bids:{RESET} {flag['bid_count']}")
    print(f"  {BOLD}Flagged:{RESET} {_fmt_ts(flag['flagged_at'])}")
    print(f"\n  {BOLD}Reasons:{RESET}")
    for r in reasons:
        print(f"    • {r}")


def interactive_review(tier: str | None = None) -> None:
    """Walk through unreviewed flags one at a time, letting user label each."""
    flags = get_unreviewed_flags(tier=tier, limit=100)
    if not flags:
        print(f"\n{GREEN}No unreviewed flags{' at tier ' + tier if tier else ''}.{RESET}")
        return

    print(f"\n{BOLD}Found {len(flags)} unreviewed flag(s).{RESET}")
    print("Commands:  [y] confirm IRL trade  [n] false positive  [s] skip  [q] quit\n")

    reviewed = 0
    for i, flag in enumerate(flags, 1):
        print_flag(flag, i, len(flags))

        while True:
            try:
                cmd = input(f"\n  {BOLD}Label [{flag['auction_id'][:8]}…]:{RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if cmd == "y":
                notes = input("  Notes (optional): ").strip()
                label_auction(flag["auction_id"], label=1, notes=notes)
                print(f"  {RED}✓ Confirmed IRL trade.{RESET}")
                reviewed += 1
                break
            elif cmd == "n":
                notes = input("  Notes (optional): ").strip()
                label_auction(flag["auction_id"], label=0, notes=notes)
                print(f"  {GREEN}✓ Marked as false positive.{RESET}")
                reviewed += 1
                break
            elif cmd == "s":
                print("  Skipped.")
                break
            elif cmd == "q":
                print(f"\nReviewed {reviewed} flag(s) this session.")
                return
            else:
                print("  Enter y / n / s / q")

    print(f"\n{GREEN}Session complete. Reviewed {reviewed} / {len(flags)} flags.{RESET}")
    print("Tip: once you have ≥30 labels, run  python train.py --stage lgbm  to train the supervised model.")


def print_recent_high() -> None:
    """Print the most recent HIGH-tier flags without interactive prompts."""
    flags = get_unreviewed_flags(tier="HIGH", limit=20)
    if not flags:
        print("No HIGH-tier flags pending review.")
        return
    for i, flag in enumerate(flags, 1):
        print_flag(flag, i, len(flags))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats",     action="store_true", help="Print stats and exit")
    parser.add_argument("--tier",      choices=["HIGH", "MEDIUM", "LOW"], help="Filter by tier")
    parser.add_argument("--high-only", action="store_true", help="Show HIGH flags non-interactively")
    parser.add_argument("--demo",      action="store_true", help="Use the isolated demo DB and demo model artifacts")
    args = parser.parse_args()
    config.set_storage_context("demo" if args.demo else "real")
    init_db()
    logger.info("Using storage %s", config.describe_storage_context())

    print_stats()

    if args.stats:
        pass
    elif args.high_only:
        print_recent_high()
    else:
        interactive_review(tier=args.tier)
