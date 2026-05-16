from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

GENESIS_BLOCK_HASH = "0" * 64


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class Ledger:
    def __init__(self, path: str | Path = "data/flxbc.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'running',
                    config_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    reputation REAL NOT NULL DEFAULT 0.8,
                    points REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS rounds (
                    run_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    metrics_hash TEXT NOT NULL,
                    participants_hash TEXT NOT NULL,
                    participants_json TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, round_id)
                );

                CREATE TABLE IF NOT EXISTS contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    samples INTEGER NOT NULL,
                    contribution REAL NOT NULL,
                    reputation REAL NOT NULL,
                    points REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS misbehavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    penalty REAL NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS audit_blocks (
                    run_id TEXT NOT NULL,
                    block_height INTEGER NOT NULL,
                    round_id INTEGER NOT NULL,
                    previous_block_hash TEXT NOT NULL,
                    block_hash TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, block_height),
                    UNIQUE (run_id, round_id)
                );
                """
            )

    def create_run(
        self,
        run_id: str,
        *,
        mode: str,
        dataset: str,
        strategy: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            for table in (
                "rounds",
                "contributions",
                "misbehavior",
                "experiments",
                "audit_blocks",
            ):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, mode, dataset, strategy, status, config_json)
                VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    mode,
                    dataset,
                    strategy,
                    json.dumps(config or {}, sort_keys=True, default=str),
                ),
            )

    def finish_run(self, run_id: str, *, status: str = "completed") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def upsert_node(self, node_id: str, *, display_name: str, reputation: float = 0.8) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes (node_id, display_name, reputation)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET display_name=excluded.display_name
                """,
                (node_id, display_name, reputation),
            )

    def update_node_score(self, node_id: str, *, reputation: float, points_delta: float) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE nodes
                SET reputation = ?, points = points + ?
                WHERE node_id = ?
                """,
                (reputation, points_delta, node_id),
            )

    def record_round(
        self,
        *,
        run_id: str,
        round_id: int,
        metrics: dict[str, Any],
        artifact_uri: str,
        participants: list[str],
        tx_hash: str,
        model_hash: str | None = None,
        metrics_hash: str | None = None,
        participants_hash: str | None = None,
    ) -> dict[str, str]:
        hashes = {
            "model_hash": model_hash
            or stable_hash({"artifact_uri": artifact_uri, "round_id": round_id}),
            "metrics_hash": metrics_hash or stable_hash(metrics),
            "participants_hash": participants_hash or stable_hash(participants),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rounds (
                    run_id, round_id, metrics_json, artifact_uri, model_hash,
                    metrics_hash, participants_hash, participants_json, tx_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_id,
                    json.dumps(metrics, sort_keys=True),
                    artifact_uri,
                    hashes["model_hash"],
                    hashes["metrics_hash"],
                    hashes["participants_hash"],
                    json.dumps(participants, sort_keys=True),
                    tx_hash,
                ),
            )
        return hashes

    def record_contribution(
        self,
        *,
        run_id: str,
        round_id: int,
        node_id: str,
        samples: int,
        contribution: float,
        reputation: float,
        points: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO contributions
                    (run_id, round_id, node_id, samples, contribution, reputation, points)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, round_id, node_id, samples, contribution, reputation, points),
            )
        self.update_node_score(node_id, reputation=reputation, points_delta=points)

    def record_misbehavior(
        self,
        *,
        run_id: str,
        round_id: int,
        node_id: str,
        kind: str,
        penalty: float,
        detail: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO misbehavior (run_id, round_id, node_id, kind, penalty, detail)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, round_id, node_id, kind, penalty, detail),
            )

    def record_experiment_result(self, *, run_id: str, label: str, metrics: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO experiments (run_id, label, metrics_json) VALUES (?, ?, ?)",
                (run_id, label, json.dumps(metrics, sort_keys=True)),
            )

    def record_audit_block(
        self,
        *,
        run_id: str,
        round_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connect() as conn:
            previous = conn.execute(
                """
                SELECT block_height, block_hash
                FROM audit_blocks
                WHERE run_id = ?
                ORDER BY block_height DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            block_height = int(previous["block_height"]) + 1 if previous else 1
            previous_block_hash = previous["block_hash"] if previous else GENESIS_BLOCK_HASH
            tx_hash = "mockchain:" + stable_hash(
                {
                    "run_id": run_id,
                    "round_id": round_id,
                    "payload": payload,
                }
            )[:32]
            block_hash = _audit_block_hash(
                block_height=block_height,
                previous_block_hash=previous_block_hash,
                tx_hash=tx_hash,
                payload=payload,
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_blocks (
                    run_id, block_height, round_id, previous_block_hash,
                    block_hash, tx_hash, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    block_height,
                    round_id,
                    previous_block_hash,
                    block_hash,
                    tx_hash,
                    json.dumps(payload, sort_keys=True),
                ),
            )
        return {
            "run_id": run_id,
            "block_height": block_height,
            "round_id": round_id,
            "previous_block_hash": previous_block_hash,
            "block_hash": block_hash,
            "tx_hash": tx_hash,
            "payload": payload,
        }

    def list_runs(self) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM runs ORDER BY started_at DESC")

    def list_nodes(self) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM nodes ORDER BY points DESC, node_id ASC")

    def get_rounds(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._fetch_all("SELECT * FROM rounds ORDER BY created_at DESC, round_id DESC")
        return self._fetch_all(
            "SELECT * FROM rounds WHERE run_id = ? ORDER BY round_id ASC", (run_id,)
        )

    def get_contributions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._fetch_all("SELECT * FROM contributions ORDER BY created_at DESC")
        return self._fetch_all(
            "SELECT * FROM contributions WHERE run_id = ? ORDER BY round_id ASC, node_id ASC",
            (run_id,),
        )

    def get_misbehavior(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._fetch_all("SELECT * FROM misbehavior ORDER BY created_at DESC")
        return self._fetch_all(
            "SELECT * FROM misbehavior WHERE run_id = ? ORDER BY round_id ASC, node_id ASC",
            (run_id,),
        )

    def get_experiments(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._fetch_all("SELECT * FROM experiments ORDER BY created_at DESC")
        return self._fetch_all(
            "SELECT * FROM experiments WHERE run_id = ? ORDER BY id ASC", (run_id,)
        )

    def get_audit_blocks(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._fetch_all(
                "SELECT * FROM audit_blocks ORDER BY created_at DESC, block_height DESC"
            )
        return self._fetch_all(
            "SELECT * FROM audit_blocks WHERE run_id = ? ORDER BY block_height ASC",
            (run_id,),
        )

    def verify_audit_chain(self, run_id: str | None = None) -> dict[str, Any]:
        blocks = self.get_audit_blocks(run_id)
        if run_id is None:
            blocks = sorted(blocks, key=lambda block: (block["run_id"], block["block_height"]))

        previous_by_run: dict[str, str] = {}
        for block in blocks:
            block_run_id = block["run_id"]
            expected_previous = previous_by_run.get(block_run_id, GENESIS_BLOCK_HASH)
            expected_tx_hash = "mockchain:" + stable_hash(
                {
                    "run_id": block_run_id,
                    "round_id": block["round_id"],
                    "payload": block["payload"],
                }
            )[:32]
            expected_block_hash = _audit_block_hash(
                block_height=block["block_height"],
                previous_block_hash=expected_previous,
                tx_hash=expected_tx_hash,
                payload=block["payload"],
            )
            if (
                block["previous_block_hash"] != expected_previous
                or block["tx_hash"] != expected_tx_hash
                or block["block_hash"] != expected_block_hash
            ):
                return {
                    "valid": False,
                    "blocks": len(blocks),
                    "head_block_hash": None,
                    "broken_at": block["block_height"],
                }
            previous_by_run[block_run_id] = block["block_hash"]

        head = blocks[-1]["block_hash"] if blocks else None
        return {
            "valid": True,
            "blocks": len(blocks),
            "head_block_hash": head,
            "broken_at": None,
        }

    def _fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_decode_row(dict(row)) for row in rows]


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in list(row):
        if key.endswith("_json"):
            decoded_key = key[: -len("_json")]
            row[decoded_key] = json.loads(row.pop(key))
    return row


def _audit_block_hash(
    *,
    block_height: int,
    previous_block_hash: str,
    tx_hash: str,
    payload: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "block_height": block_height,
            "previous_block_hash": previous_block_hash,
            "tx_hash": tx_hash,
            "payload": payload,
        }
    )
