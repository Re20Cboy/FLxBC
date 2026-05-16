from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flxbc.config import PrivacyMode
from flxbc.crypto import verify_hmac_payload
from flxbc.strategy import ParameterDict


@dataclass(frozen=True, slots=True)
class ParameterEnvelope:
    run_id: str
    round_id: int
    node_id: str
    payload_hash: str
    payload_bytes: bytes
    timestamp: float
    nonce: str
    privacy_mode: PrivacyMode
    signature: str = ""


@dataclass(frozen=True, slots=True)
class PrivacyUpdateResult:
    parameters: ParameterDict
    before_clip_norm: float
    after_clip_norm: float
    noise_std: float


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_parameters(parameters: ParameterDict) -> str:
    return hash_bytes(serialize_parameters(parameters))


def apply_update_privacy(
    local_parameters: ParameterDict,
    global_parameters: ParameterDict,
    *,
    clipping_norm: float,
    dp_noise_multiplier: float,
    rng: np.random.Generator,
) -> PrivacyUpdateResult:
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    if dp_noise_multiplier < 0:
        raise ValueError("dp_noise_multiplier cannot be negative")

    before_clip_norm = _floating_update_norm(local_parameters, global_parameters)
    scale = min(1.0, clipping_norm / before_clip_norm) if before_clip_norm > 0 else 1.0
    after_clip_norm = before_clip_norm * scale
    noise_std = clipping_norm * dp_noise_multiplier

    private_parameters: ParameterDict = {}
    for name, local_value in local_parameters.items():
        if not np.issubdtype(local_value.dtype, np.floating):
            private_parameters[name] = local_value.copy()
            continue
        global_value = global_parameters[name].astype(np.float64)
        clipped_delta = (local_value.astype(np.float64) - global_value) * scale
        if noise_std > 0:
            clipped_delta = clipped_delta + rng.normal(0.0, noise_std, size=clipped_delta.shape)
        private_parameters[name] = (global_value + clipped_delta).astype(local_value.dtype)

    return PrivacyUpdateResult(
        parameters=private_parameters,
        before_clip_norm=before_clip_norm,
        after_clip_norm=after_clip_norm,
        noise_std=noise_std,
    )


def serialize_parameters(parameters: ParameterDict) -> bytes:
    manifest: list[dict[str, object]] = []
    payload_parts: list[bytes] = []
    for name in sorted(parameters):
        array = np.ascontiguousarray(parameters[name])
        raw = array.tobytes(order="C")
        manifest.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": len(raw),
            }
        )
        payload_parts.append(raw)

    header = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(header).to_bytes(8, "big") + header + b"".join(payload_parts)


def deserialize_parameters(payload: bytes) -> ParameterDict:
    if len(payload) < 8:
        raise ValueError("parameter payload is missing its manifest header")
    header_len = int.from_bytes(payload[:8], "big")
    header_start = 8
    header_end = header_start + header_len
    if header_end > len(payload):
        raise ValueError("parameter payload manifest is truncated")

    manifest = json.loads(payload[header_start:header_end].decode("utf-8"))
    offset = header_end
    parameters: ParameterDict = {}
    for item in manifest:
        name = str(item["name"])
        dtype = np.dtype(str(item["dtype"]))
        shape = tuple(int(value) for value in item["shape"])
        nbytes = int(item["nbytes"])
        end = offset + nbytes
        if end > len(payload):
            raise ValueError(f"parameter payload for `{name}` is truncated")
        array = np.frombuffer(payload[offset:end], dtype=dtype).copy().reshape(shape)
        parameters[name] = array
        offset = end

    if offset != len(payload):
        raise ValueError("parameter payload has trailing bytes")
    return parameters


def create_parameter_envelope(
    parameters: ParameterDict,
    *,
    run_id: str,
    round_id: int,
    node_id: str,
    privacy_mode: PrivacyMode,
    timestamp: float | None = None,
    nonce: str | None = None,
    signature: str = "",
) -> ParameterEnvelope:
    payload_bytes = serialize_parameters(parameters)
    return ParameterEnvelope(
        run_id=run_id,
        round_id=round_id,
        node_id=node_id,
        payload_hash=hash_bytes(payload_bytes),
        payload_bytes=payload_bytes,
        timestamp=time.time() if timestamp is None else timestamp,
        nonce=secrets.token_hex(16) if nonce is None else nonce,
        privacy_mode=privacy_mode,
        signature=signature,
    )


def validate_parameter_envelope(
    envelope: ParameterEnvelope,
    *,
    expected_run_id: str,
    expected_round_id: int,
    expected_node_id: str,
    seen_nonces: set[tuple[str, int, str, str]],
    now: float | None = None,
    replay_window_seconds: int = 300,
    signature_secret: bytes | None = None,
) -> tuple[bool, str | None]:
    if envelope.run_id != expected_run_id:
        return False, "run-mismatch"
    if envelope.round_id != expected_round_id:
        return False, "round-mismatch"
    if envelope.node_id != expected_node_id:
        return False, "node-mismatch"
    if abs((time.time() if now is None else now) - envelope.timestamp) > replay_window_seconds:
        return False, "expired"
    if hash_bytes(envelope.payload_bytes) != envelope.payload_hash:
        return False, "hash-mismatch"
    if signature_secret is not None and not verify_hmac_payload(
        signature_secret,
        envelope_signature_fields(envelope),
        envelope.signature,
    ):
        return False, "invalid-signature"

    nonce_key = (envelope.run_id, envelope.round_id, envelope.node_id, envelope.nonce)
    if nonce_key in seen_nonces:
        return False, "replay"
    seen_nonces.add(nonce_key)
    return True, None


def envelope_signature_fields(envelope: ParameterEnvelope) -> dict[str, object]:
    return {
        "run_id": envelope.run_id,
        "round_id": envelope.round_id,
        "node_id": envelope.node_id,
        "payload_hash": envelope.payload_hash,
        "timestamp": envelope.timestamp,
        "nonce": envelope.nonce,
        "privacy_mode": envelope.privacy_mode,
    }


def _floating_update_norm(left: ParameterDict, right: ParameterDict) -> float:
    total = 0.0
    for name, left_value in left.items():
        if not np.issubdtype(left_value.dtype, np.floating):
            continue
        diff = left_value.astype(np.float64) - right[name].astype(np.float64)
        total += float(np.sum(diff * diff))
    return float(np.sqrt(total))
