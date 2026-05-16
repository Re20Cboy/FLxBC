from fastapi.testclient import TestClient

from flxbc.api import app
from flxbc.ledger import Ledger


def test_api_token_is_required_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FLXBC_DB", str(tmp_path / "flxbc.db"))
    monkeypatch.setenv("FLXBC_API_TOKEN", "secret-token")
    Ledger(tmp_path / "flxbc.db")
    client = TestClient(app)

    assert client.get("/runs").status_code == 401
    assert client.get("/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/runs", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_api_sanitizes_artifact_paths_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("FLXBC_DB", str(tmp_path / "flxbc.db"))
    monkeypatch.delenv("FLXBC_API_TOKEN", raising=False)
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("api-privacy", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")
    ledger.record_round(
        run_id="api-privacy",
        round_id=1,
        metrics={
            "accuracy": 0.75,
            "artifact_uri": "/private/full/path/round_1_parameters.npz.enc",
        },
        artifact_uri="/private/full/path/round_1_parameters.npz.enc",
        participants=["hospital_1"],
        tx_hash="mockchain:privacy",
        model_hash="model",
    )
    ledger.record_audit_block(
        run_id="api-privacy",
        round_id=1,
        payload={
            "artifact_uri": "/private/full/path/round_1_parameters.npz.enc",
            "model_hash": "model",
        },
    )

    client = TestClient(app)
    rounds = client.get("/rounds", params={"run_id": "api-privacy"}).json()
    blocks = client.get("/audit-blocks", params={"run_id": "api-privacy"}).json()

    assert rounds[0]["artifact_uri"] == "round_1_parameters.npz.enc"
    assert rounds[0]["metrics"]["artifact_uri"] == "round_1_parameters.npz.enc"
    assert blocks[0]["payload"]["artifact_uri"] == "round_1_parameters.npz.enc"
