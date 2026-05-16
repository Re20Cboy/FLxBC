import base64
import zipfile
from pathlib import Path

import numpy as np
import pytest

from flxbc.config import RunConfig
from flxbc.crypto import (
    decrypt_artifact_file,
    encrypt_artifact_file,
    load_artifact_key_from_env,
)
from flxbc.privacy import hash_file
from flxbc.simulation import run_federated_demo


def _demo_key() -> str:
    return base64.urlsafe_b64encode(b"1" * 32).decode("ascii")


def test_artifact_file_encryption_round_trips_and_blocks_direct_np_load(tmp_path: Path):
    plain_path = tmp_path / "model.npz"
    encrypted_path = tmp_path / "model.npz.enc"
    np.savez_compressed(plain_path, layer=np.array([1.0, 2.0], dtype=np.float32))

    result = encrypt_artifact_file(
        plain_path,
        encrypted_path,
        key=b"1" * 32,
        associated_data=b"run-1:1",
    )

    assert encrypted_path.exists()
    assert not plain_path.exists()
    assert result.ciphertext_hash == hash_file(encrypted_path)
    assert result.nonce_hash
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        np.load(encrypted_path)

    decrypted = decrypt_artifact_file(
        encrypted_path,
        key=b"1" * 32,
        nonce=result.nonce,
        associated_data=b"run-1:1",
    )
    round_trip_path = tmp_path / "round-trip.npz"
    round_trip_path.write_bytes(decrypted)
    with np.load(round_trip_path) as data:
        np.testing.assert_allclose(data["layer"], np.array([1.0, 2.0], dtype=np.float32))


def test_load_artifact_key_requires_env(monkeypatch):
    monkeypatch.delenv("FLXBC_ARTIFACT_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FLXBC_ARTIFACT_KEY"):
        load_artifact_key_from_env()


def test_encrypted_artifact_run_records_ciphertext_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLXBC_ARTIFACT_KEY", _demo_key())
    config = RunConfig(
        run_id="encrypted-artifact",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=1,
        seed=9,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        device="numpy",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        artifact_encryption=True,
    )

    result = run_federated_demo(config)

    round_row = result.ledger.get_rounds("encrypted-artifact")[0]
    artifact_path = Path(round_row["artifact_uri"])
    assert artifact_path.suffix == ".enc"
    assert round_row["model_hash"] == hash_file(artifact_path)
    assert round_row["metrics"]["artifact_encryption"] is True
    assert round_row["metrics"]["artifact_encryption_mode"] == "aes-gcm"
    assert round_row["metrics"]["artifact_ciphertext_hash"] == round_row["model_hash"]
    assert round_row["metrics"]["artifact_nonce_hash"]
    assert not artifact_path.with_suffix("").exists()

    audit_payload = result.ledger.get_audit_blocks("encrypted-artifact")[0]["payload"]
    assert audit_payload["artifact_ciphertext_hash"] == round_row["model_hash"]
    assert audit_payload["artifact_encryption_mode"] == "aes-gcm"
