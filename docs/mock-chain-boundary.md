# FLxBC Mock Chain Boundary

FLxBC 的默认审计链是 SQLite 中的 mock audit chain。它用于本地实验可复现、训练轮次完整性验证和 dashboard 展示，不等价于生产区块链。

## 默认 mock chain 做什么

- 每个有效训练轮次写入一个 audit block。
- block 记录 previous block hash，形成按 run 独立的哈希链。
- payload 记录 model hash、metrics hash、participants hash、strategy hash、privacy hash 和 artifact 元数据。
- `Ledger.verify_audit_chain()` 可以发现 payload、tx hash、previous hash 或 block hash 被篡改的情况。
- FastAPI 和 dashboard 可以展示链头 hash、block 数量和异常位置。

## 默认 mock chain 不做什么

- 不提供跨机构共识。
- 不提供抗审查或去中心化存储。
- 不提供生产级链上身份、权限治理或 gas 经济模型。
- 不保护本机管理员权限攻击者。
- 不替代医疗合规审计。

## Optional Hardhat adapter

Hardhat 合约位于 `contracts/`，用于展示未来迁移到智能合约审计层的可行性。它是 local-only optional adapter：

- 默认 Python demo 不依赖 Hardhat。
- README 要求使用 Node.js 22 运行 Hardhat 命令。
- 当前 Hardhat 2 依赖树仍有 `npm audit` 风险；在决定迁移 Hardhat 3 前，不应把它描述为生产运行时。

## 报告建议表述

建议在论文或项目报告中使用以下边界描述：

> 本项目采用 SQLite mock audit chain 验证训练轮次记录的完整性，并保留 Solidity 合约作为未来链上迁移的接口原型。当前实现重点是联邦学习审计流程和隐私增强机制的本地可复现实验，不声称提供生产区块链共识或临床合规能力。

