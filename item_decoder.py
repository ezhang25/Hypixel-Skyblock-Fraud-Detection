"""
item_decoder.py - Decode Hypixel item_bytes into structured item metadata.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import re
from typing import Any

import nbtlib


FORMAT_CODE_RE = re.compile(r"\u00A7.")


def strip_minecraft_formatting(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = FORMAT_CODE_RE.sub("", text).strip()
    return cleaned or None


def extract_item_bytes_data(item_bytes_field: Any) -> str | None:
    if isinstance(item_bytes_field, str):
        return item_bytes_field
    if isinstance(item_bytes_field, dict):
        data = item_bytes_field.get("data")
        return data if isinstance(data, str) else None
    return None


def _unwrap_nbt(value: Any) -> Any:
    if hasattr(value, "unpack"):
        return _unwrap_nbt(value.unpack())
    if isinstance(value, dict):
        return {str(k): _unwrap_nbt(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap_nbt(v) for v in value]
    return value


def _get_first_stack(raw_nbt: bytes) -> dict[str, Any] | None:
    file = nbtlib.File.parse(io.BytesIO(raw_nbt))
    root = _unwrap_nbt(file)
    items = root.get("i")
    if isinstance(items, list) and items:
        first = items[0]
        return first if isinstance(first, dict) else None
    return None


def _summary_from_mapping(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    parts = [f"{key}:{value[key]}" for key in sorted(value)]
    return ", ".join(parts) or None


def _extract_gemstone_data(extra: dict[str, Any]) -> dict[str, Any] | None:
    direct = extra.get("gems")
    if isinstance(direct, dict) and direct:
        return direct

    gemstone_keys = {
        key: value
        for key, value in extra.items()
        if "gem" in key.lower() and value not in (None, "", {}, [])
    }
    return gemstone_keys or None


def _extract_attribute_data(extra: dict[str, Any]) -> dict[str, Any] | None:
    direct = extra.get("attributes")
    if isinstance(direct, dict) and direct:
        return direct

    attribute_keys = {
        key: value
        for key, value in extra.items()
        if "attribute" in key.lower() and value not in (None, "", {}, [])
    }
    return attribute_keys or None


def _extract_stars(extra: dict[str, Any]) -> int | None:
    star_keys = ("dungeon_item_level", "upgrade_level", "dungeonize_level")
    star_values = []
    for key in star_keys:
        value = extra.get(key)
        if isinstance(value, (int, float)):
            star_values.append(int(value))
    return max(star_values) if star_values else None


def _extract_potato_counts(extra: dict[str, Any]) -> tuple[int | None, int | None]:
    raw_total = extra.get("hot_potato_count")
    if not isinstance(raw_total, (int, float)):
        return None, None
    total = max(0, int(raw_total))
    return min(total, 10), max(total - 10, 0)


def _extract_pet_data(extra: dict[str, Any]) -> dict[str, Any] | None:
    raw_pet_info = extra.get("petInfo")
    if not isinstance(raw_pet_info, str) or not raw_pet_info:
        return None

    try:
        pet_info = json.loads(raw_pet_info)
    except json.JSONDecodeError:
        return None

    level_data = pet_info.get("level")
    pet_level = None
    if isinstance(level_data, dict) and isinstance(level_data.get("level"), (int, float)):
        pet_level = int(level_data["level"])
    elif isinstance(pet_info.get("level"), (int, float)):
        pet_level = int(pet_info["level"])

    pet_exp = pet_info.get("exp")
    pet_exp = float(pet_exp) if isinstance(pet_exp, (int, float)) else None

    candy_used = pet_info.get("candyUsed")
    candy_used = int(candy_used) if isinstance(candy_used, (int, float)) else None

    return {
        "pet_type": pet_info.get("type") if isinstance(pet_info.get("type"), str) else None,
        "pet_tier": pet_info.get("tier") if isinstance(pet_info.get("tier"), str) else None,
        "pet_level": pet_level,
        "pet_exp": pet_exp,
        "pet_held_item": pet_info.get("heldItem") if isinstance(pet_info.get("heldItem"), str) else None,
        "pet_candy_used": candy_used,
        "pet_skin": pet_info.get("skin") if isinstance(pet_info.get("skin"), str) else None,
    }


def decode_item_bytes(item_bytes_field: Any) -> tuple[dict[str, Any] | None, str | None]:
    encoded = extract_item_bytes_data(item_bytes_field)
    if not encoded:
        return None, "item_bytes missing"

    try:
        raw_nbt = gzip.decompress(base64.b64decode(encoded))
        stack = _get_first_stack(raw_nbt)
        if not stack:
            return None, "decoded NBT contained no item stack"

        tag = stack.get("tag", {}) if isinstance(stack.get("tag"), dict) else {}
        display = tag.get("display", {}) if isinstance(tag.get("display"), dict) else {}
        extra = tag.get("ExtraAttributes", {}) if isinstance(tag.get("ExtraAttributes"), dict) else {}

        enchantments = extra.get("enchantments", {})
        enchantments = enchantments if isinstance(enchantments, dict) else {}

        runes = extra.get("runes", {})
        runes = runes if isinstance(runes, dict) else {}

        gemstones = _extract_gemstone_data(extra)
        attributes = _extract_attribute_data(extra)
        hot_potato_count, fuming_potato_count = _extract_potato_counts(extra)
        pet_data = _extract_pet_data(extra)

        display_name = display.get("Name")
        clean_name = strip_minecraft_formatting(display_name)
        item_id = extra.get("id") if isinstance(extra.get("id"), str) else None

        metadata = {
            "item_id": item_id,
            "clean_name": clean_name,
            "display_name": display_name,
            "lore": display.get("Lore") if isinstance(display.get("Lore"), list) else None,
            "enchantments": enchantments or None,
            "enchant_count": len(enchantments) if enchantments else 0,
            "enchant_summary": _summary_from_mapping(enchantments),
            "stars": _extract_stars(extra),
            "recombobulated": bool(extra.get("rarity_upgrades", 0)) if "rarity_upgrades" in extra else None,
            "hot_potato_count": hot_potato_count,
            "fuming_potato_count": fuming_potato_count,
            "reforge": extra.get("modifier") if isinstance(extra.get("modifier"), str) else None,
            "runes": runes or None,
            "rune_summary": _summary_from_mapping(runes),
            "gemstones": gemstones,
            "gemstone_summary": _summary_from_mapping(gemstones),
            "attributes": attributes,
            "attribute_summary": _summary_from_mapping(attributes),
            "dungeon_tier": (
                str(extra.get("dungeon_item_level"))
                if isinstance(extra.get("dungeon_item_level"), (int, float))
                else None
            ),
            "skin": extra.get("skin") if isinstance(extra.get("skin"), str) else None,
            "dye": (
                extra.get("dye_item")
                if isinstance(extra.get("dye_item"), str)
                else extra.get("dye") if isinstance(extra.get("dye"), str) else None
            ),
            "item_uuid": extra.get("uuid") if isinstance(extra.get("uuid"), str) else None,
            "pet": pet_data,
            "extra_attributes": extra or None,
        }

        return metadata, None
    except Exception as exc:
        return None, str(exc)
