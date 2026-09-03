"""Private FastAPI web dashboard for reviewing detector results.

The server binds to localhost by default. Access it from a local browser using
an SSH tunnel instead of exposing the database review interface publicly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import (
    db_stats,
    get_unreviewed_flags,
    init_db,
    is_unreviewed_flag,
    label_auction,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"

app = FastAPI(title="Skyblock Fraud Detector", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LabelRequest(BaseModel):
    label: Literal[0, 1]
    notes: str = Field(default="", max_length=1_000)


def _serialise_flag(flag: dict) -> dict:
    """Make the database flag record ready for JSON and UI consumption."""
    result = dict(flag)
    try:
        result["reasons"] = json.loads(result.get("reasons") or "[]")
    except json.JSONDecodeError:
        result["reasons"] = []
    result["reviewed"] = bool(result.get("reviewed", False))
    return result


@app.on_event("startup")
def initialise_database() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict:
    return db_stats()


@app.get("/api/flags")
def flags(
    tier: Literal["HIGH", "MEDIUM", "LOW"] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    return {"flags": [_serialise_flag(flag) for flag in get_unreviewed_flags(tier, limit)]}


@app.post("/api/flags/{auction_id}/label")
def label_flag(auction_id: str, payload: LabelRequest) -> dict[str, str]:
    if not is_unreviewed_flag(auction_id):
        raise HTTPException(status_code=404, detail="Unreviewed flag not found")

    label_auction(auction_id, label=payload.label, notes=payload.notes.strip())
    outcome = "confirmed IRL trade" if payload.label else "marked false positive"
    return {"message": f"Auction {outcome}."}


def main() -> None:
    host = os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_DASHBOARD_PORT", "8000"))
    uvicorn.run("web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
