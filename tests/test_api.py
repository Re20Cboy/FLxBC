from fastapi.testclient import TestClient

from flxbc.api import app
from flxbc.ledger import Ledger


def test_api_exposes_mock_audit_chain_status(tmp_path, monkeypatch):
    db_path = tmp_path / "flxbc.db"
    monkeypatch.setenv("FLXBC_DB", str(db_path))
    ledger = Ledger(db_path)
    ledger.create_run("api-run", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")
    ledger.record_audit_block(
        run_id="api-run",
        round_id=1,
        payload={
            "artifact_uri": "artifact",
            "metrics_hash": "metrics",
            "model_hash": "model",
            "participants_hash": "participants",
            "strategy_hash": "strategy",
        },
    )

    client = TestClient(app)
    status = client.get("/audit-status", params={"run_id": "api-run"})
    blocks = client.get("/audit-blocks", params={"run_id": "api-run"})

    assert status.status_code == 200
    assert status.json()["valid"] is True
    assert status.json()["blocks"] == 1
    assert blocks.status_code == 200
    assert blocks.json()[0]["tx_hash"].startswith("mockchain:")
