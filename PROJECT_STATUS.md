# FLxBC Project Status

## Current Demo Capability

- Synthetic federated learning demo runs locally without Docker.
- SQLite ledger records runs, rounds, nodes, contributions, and misbehavior events.
- Default mock audit chain records one simulated block per training round.
- Audit blocks link to the previous block and can be verified for tampering.
- FastAPI exposes the local ledger and audit-chain status.
- Streamlit dashboard shows training progress, train/validation/test metrics, node contribution,
  audit hashes, privacy status, and mock blocks.
- `make demo-check` provides a compact reproducible smoke demo with audit-chain verification.
- `make quick-experiment` provides a short comparison run for demonstration.
- A small PneumoniaMNIST run has verified the real MedMNIST + Torch backend path.
- Hardhat contracts remain available as an optional smart-contract adapter.

## Recommended Completion Target

The project should complete as a lightweight local research/demo system, not a full
production blockchain or clinical FL platform. The target is a reliable end-to-end demo:

1. Run synthetic FL quickly.
2. Optionally run MedMNIST/Torch when heavier dependencies are installed.
3. Record each round into a verifiable mock blockchain audit trail.
4. Show training, contribution, misbehavior, and audit status in the dashboard.
5. Keep Solidity contracts as evidence of future migration feasibility.

## Remaining Gaps

- Dashboard polish is in place for audit-chain summaries and related experiment runs.
- Privacy is partially implemented as local demo plumbing, not a completed production
  privacy feature. The FL simulation now wraps client updates in validated parameter
  envelopes, records accepted/rejected envelope counts, hashes model artifacts by file
  content, verifies demo HMAC signatures for `client_auth=hmac-demo`, and can apply DP
  clipping/noise for `privacy_mode=dp` or `full-demo`. Artifact encryption is available
  through AES-GCM when `FLXBC_ARTIFACT_KEY` is set. Secure aggregation demo mode applies
  zero-sum masks when all clients participate and records an explicit fallback when
  clients are missing. Transport TLS/mTLS is still pending for future multi-host use.
- The FastAPI surface can require `FLXBC_API_TOKEN` and now sanitizes artifact paths in
  `/rounds` and `/audit-blocks` responses.
- Training metrics now include train/validation/test loss, accuracy, Macro-F1, AUC, and
  train-validation generalization gaps. Backward-compatible `accuracy`, `macro_f1`, `loss`,
  and `auc` aliases point to test metrics.
- Each federated round records download/upload parameter bytes, total round communication
  bytes, and cumulative communication bytes for later communication-efficiency analysis.
- Each federated round records wall-clock timing for the round, local training,
  aggregation, and evaluation.
- Target metrics now record time-to-target round, seconds, and communication bytes.
- Per-client evaluation summaries record mean/std/min/max for loss, accuracy, Macro-F1,
  and AUC to expose client drift and non-IID fairness risk.
- Optional early stopping is available through `--early-stopping`. It monitors validation
  metrics with `patience`, `min_delta`, `min_rounds`, and optional target accuracy/Macro-F1,
  then records `best_round`, `best_metrics`, and `stop_reason`.
- Common Makefile entrypoints (`make run`, `make run-real-2d`, `make run-real-3d`,
  `make demo`, `make experiment`) now use validation-driven early stopping by default.
- `make run-real-2d` intentionally avoids a default single-metric target accuracy stop because
  small imbalanced PneumoniaMNIST validation samples can hit 0.85 accuracy before the model
  has a convincing test/Macro-F1 profile. It uses a larger default sample budget,
  validation-loss patience, and adaptive round extension instead.
- `make run-real-3d` covers NoduleMNIST3D through the 3D CNN path with smaller default
  samples and a lower adaptive-round cap because the 3D path is heavier.
- Dashboard quality charts now emphasize test accuracy and test Macro-F1; AUC remains in
  metrics tables for binary ranking diagnostics but is no longer plotted in the main chart.
- `docs/privacy-threat-model.md` now defines the protected assets, attacker model,
  non-goals, and minimum privacy loop needed before describing the system as
  privacy-enhanced.
- Privacy-related CLI/config fields are available for requested modes:
  `privacy_mode`, `client_auth`, `transport_security`, `artifact_encryption`,
  `secure_aggregation`, `replay_window_seconds`, `dp_noise_multiplier`, and
  `clipping_norm`.
- Hardhat is pinned to Node.js 22 for optional local demos.
- `npm audit` still reports vulnerabilities in the Hardhat 2 dependency tree. npm's
  available fix is a breaking Hardhat 3 migration, so this remains documented optional
  local-only infrastructure rather than default project runtime.

## Next Best Work

1. Add explicit TLS/mTLS deployment guidance for future multi-process or multi-host use.
2. Expand dashboard/API privacy reporting around artifact hashes and secure aggregation
   fallback rates.
3. Add a dashboard/report export that summarizes final vs best metrics, target-reaching
   cost, and privacy/audit configuration in one table.
4. Decide whether a future release should migrate the optional adapter to Hardhat 3.
5. Add a final paper/report section that explains the mock-chain design boundary.
6. Package or archive a clean demo dataset/cache if the project must run offline.
