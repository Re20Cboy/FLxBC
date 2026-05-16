from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def demo_client_secret(
    run_id: str,
    node_id: str,
    *,
    master_secret: str | None = None,
) -> bytes:
    master = master_secret or os.getenv("FLXBC_CLIENT_AUTH_SECRET")
    if not master:
        raise RuntimeError(
            "FLXBC_CLIENT_AUTH_SECRET env var is required. "
            "Set it to a random string for local demos."
        )
    return hmac.new(
        master.encode("utf-8"),
        f"{run_id}:{node_id}".encode(),
        hashlib.sha256,
    ).digest()


def sign_hmac_payload(secret: bytes, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_payload(payload)
    return hmac.new(secret, encoded, hashlib.sha256).hexdigest()


def verify_hmac_payload(secret: bytes, payload: Mapping[str, Any], signature: str) -> bool:
    expected = sign_hmac_payload(secret, payload)
    return hmac.compare_digest(expected, signature)


@dataclass(frozen=True, slots=True)
class EncryptedArtifact:
    path: Path
    nonce: bytes
    ciphertext_hash: str
    nonce_hash: str
    plaintext_hash: str
    mode: str = "aes-gcm"


def load_artifact_key_from_env() -> bytes:
    encoded = os.getenv("FLXBC_ARTIFACT_KEY")
    if not encoded:
        raise RuntimeError("FLXBC_ARTIFACT_KEY must be set when artifact encryption is enabled")
    try:
        import base64

        key = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception as exc:
        raise RuntimeError("FLXBC_ARTIFACT_KEY must be a urlsafe base64 encoded key") from exc
    if len(key) != 32:
        raise RuntimeError("FLXBC_ARTIFACT_KEY must decode to 32 bytes for AES-256-GCM")
    return key


def encrypt_artifact_file(
    plain_path: str | Path,
    encrypted_path: str | Path,
    *,
    key: bytes,
    associated_data: bytes,
) -> EncryptedArtifact:
    plain = Path(plain_path)
    encrypted = Path(encrypted_path)
    plaintext = plain.read_bytes()
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    encrypted.write_bytes(ciphertext)
    plain.unlink()
    return EncryptedArtifact(
        path=encrypted,
        nonce=nonce,
        ciphertext_hash=hashlib.sha256(ciphertext).hexdigest(),
        nonce_hash=hashlib.sha256(nonce).hexdigest(),
        plaintext_hash=hashlib.sha256(plaintext).hexdigest(),
    )


def decrypt_artifact_file(
    encrypted_path: str | Path,
    *,
    key: bytes,
    nonce: bytes,
    associated_data: bytes,
) -> bytes:
    ciphertext = Path(encrypted_path).read_bytes()
    return AESGCM(key).decrypt(nonce, ciphertext, associated_data)


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
