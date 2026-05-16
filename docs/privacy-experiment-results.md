# FLxBC Privacy Experiment Results

本文档记录当前隐私增强实现的可运行状态和后续实验应观察的指标。当前结果来自本地 synthetic demo，不代表临床数据表现。

## 当前已实现隐私能力

- 参数更新 envelope：每个客户端 update 带 run id、round id、node id、payload hash、timestamp 和 nonce。
- Replay-aware nonce tracking：同一 run/round/node 的重复 nonce 会被拒绝。
- Demo HMAC client auth：`client_auth=hmac-demo` 时，客户端 update envelope 必须通过 HMAC 验签。
- DP clipping/noise：`privacy_mode=dp` 或 `full-demo` 时执行 L2 clipping；`dp_noise_multiplier > 0` 时加入 Gaussian noise。
- Content model hash：round 的 `model_hash` 绑定磁盘 artifact 文件内容。
- AES-GCM artifact encryption：`artifact_encryption=True` 且设置 `FLXBC_ARTIFACT_KEY` 时只保留 `.npz.enc`。
- Secure aggregation demo：全员参与时走 zero-sum mask 聚合；缺失客户端时显式 fallback。
- API privacy controls：设置 `FLXBC_API_TOKEN` 时要求 Bearer token；API 默认脱敏 artifact path。

## 建议实验矩阵

| Label | Command focus | Expected privacy behavior |
| --- | --- | --- |
| baseline | `--privacy-mode none` | 无 DP、无 HMAC、明文 artifact |
| dp-only | `--privacy-mode dp --dp-noise-multiplier 0.25 --clipping-norm 0.75` | update norm 被 clipping，metrics 记录 DP 状态 |
| hmac-demo | `--client-auth hmac-demo` | 无效 envelope signature 被拒绝 |
| encrypted-artifact | `--privacy-mode encrypted --artifact-encryption` | 磁盘 artifact 为 `.npz.enc` |
| secure-aggregation | `--secure-aggregation --no-failures` | `secure_aggregation_status=applied` |
| full-demo | `--privacy-mode full-demo --client-auth hmac-demo --artifact-encryption --secure-aggregation` | HMAC、DP、加密 artifact、secure aggregation demo 同时开启 |

## 每次实验应记录

- final accuracy
- macro-F1
- rounds
- participants
- privacy mode
- accepted clients
- rejected clients
- DP enabled
- clipping norm
- DP noise multiplier
- mean update norm before clipping
- mean update norm after clipping
- artifact encryption mode
- artifact ciphertext hash
- artifact nonce hash
- secure aggregation status
- secure aggregation fallback reason
- audit chain validity

## 当前解释边界

这些结果用于比较隐私机制对本地 demo 的影响。它们不能直接推导到真实医院多机部署，因为当前客户端和聚合端仍在同一 Python 进程中运行。多机部署前还需要 TLS/mTLS、证书管理、长期密钥轮换、真实网络重放防护和运维审计。

