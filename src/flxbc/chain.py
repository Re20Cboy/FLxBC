from __future__ import annotations

import json
import os
from pathlib import Path


def maybe_commit_round(
    *,
    run_id: str,
    round_id: int,
    model_hash: str,
    metrics_hash: str,
    participants_hash: str,
    strategy_hash: str,
    artifact_uri: str,
) -> str | None:
    if os.getenv("FLXBC_CHAIN_ENABLED") != "1":
        return None
    deployment_path = Path(os.getenv("FLXBC_CHAIN_DEPLOYMENT", "artifacts/chain/deployment.json"))
    if not deployment_path.exists():
        return None
    try:
        from web3 import Web3
    except ImportError:
        return None

    deployment = json.loads(deployment_path.read_text())
    rpc_url = os.getenv("FLXBC_CHAIN_RPC", "http://127.0.0.1:8545")
    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        return None
    account = web3.eth.accounts[0]
    contract_info = deployment["TrainingLedger"]
    contract = web3.eth.contract(
        address=contract_info["address"],
        abi=contract_info["abi"],
    )
    tx_hash = contract.functions.commitRound(
        run_id,
        round_id,
        _bytes32(model_hash),
        _bytes32(metrics_hash),
        _bytes32(participants_hash),
        _bytes32(strategy_hash),
        artifact_uri,
    ).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.transactionHash.hex()


def _bytes32(value: str) -> bytes:
    clean = value[2:] if value.startswith("0x") else value
    clean = clean[:64].ljust(64, "0")
    return bytes.fromhex(clean)
