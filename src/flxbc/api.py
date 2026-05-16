from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
except ImportError as exc:  # pragma: no cover - runtime dependency installed by make setup
    raise RuntimeError("FastAPI is not installed. Run `make setup`.") from exc

from flxbc.ledger import Ledger


def _ledger() -> Ledger:
    return Ledger(Path(os.getenv("FLXBC_DB", "data/flxbc.db")))


def _require_api_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("FLXBC_API_TOKEN")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API token",
        )


app = FastAPI(
    title="FLxBC Local API",
    version="0.1.0",
    description="Read-only API over the local SQLite ledger for the FLxBC Mac demo.",
    dependencies=[Depends(_require_api_token)],
)


@app.get("/health")
def health() -> dict[str, str]:
    ledger = _ledger()
    return {"status": "ok", "db": str(ledger.path)}


@app.get("/runs")
def runs() -> list[dict]:
    return _ledger().list_runs()


@app.get("/nodes")
def nodes() -> list[dict]:
    return _ledger().list_nodes()


@app.get("/rounds")
def rounds(run_id: Annotated[str | None, Query()] = None) -> list[dict]:
    return _sanitize_rows(_ledger().get_rounds(run_id))


@app.get("/contributions")
def contributions(run_id: Annotated[str | None, Query()] = None) -> list[dict]:
    return _ledger().get_contributions(run_id)


@app.get("/misbehavior")
def misbehavior(run_id: Annotated[str | None, Query()] = None) -> list[dict]:
    return _ledger().get_misbehavior(run_id)


@app.get("/experiments")
def experiments(run_id: Annotated[str | None, Query()] = None) -> list[dict]:
    return _ledger().get_experiments(run_id)


@app.get("/audit-blocks")
def audit_blocks(run_id: Annotated[str | None, Query()] = None) -> list[dict]:
    return _sanitize_rows(_ledger().get_audit_blocks(run_id))


@app.get("/audit-status")
def audit_status(run_id: Annotated[str | None, Query()] = None) -> dict:
    return _ledger().verify_audit_chain(run_id)


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    return [_sanitize_value(row) for row in rows]


def _sanitize_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_artifact_value(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _sanitize_artifact_value(key: str, value):
    if key == "artifact_uri" and isinstance(value, str):
        return Path(value).name
    return _sanitize_value(value)
