# FLxBC 后续改进推进计划 5-13

**Goal:** 将 FLxBC 从本地研究/demo 原型推进为更贴近真实医疗联邦学习场景的可复现实验系统，重点补齐隐私保护、参数安全、审计可信度和交付质量。

**Architecture:** 继续保持“轻量本地演示”为主边界，不直接扩展成生产级临床平台。新增隐私与安全能力应以可插拔模块接入现有 simulation、training、ledger、api、dashboard 流程，确保默认 demo 仍能快速运行，同时可开启隐私增强模式进行对比实验。

**Tech Stack:** Python 3.12, NumPy, PyTorch/MedMNIST optional backend, FastAPI, Streamlit, SQLite WAL mock audit chain, optional Hardhat/Solidity local chain.

---

## 0. 2026-05-13 进展更新

本计划中的第一阶段隐私闭环已推进到可运行 demo 状态：

- 已补 `privacy_mode`、client HMAC demo、参数 envelope、防重放 nonce、DP clipping/noise、artifact 内容哈希、AES-GCM artifact 加密、secure aggregation demo/fallback 状态。
- 已补 FastAPI token 认证与 artifact path 脱敏。
- 已补 dashboard 隐私状态、DP/update norm、安全聚合状态展示。
- 已补 `docs/privacy-threat-model.md`、`docs/mock-chain-boundary.md`、`docs/privacy-experiment-results.md`。
- 已新增 `docs/training-metrics-and-stopping-plan-5-13.md`，推进训练指标体系与 early stopping。
- 已补每轮参数 download/upload bytes、round communication bytes、累计通信量。
- 已补 time-to-target、每轮耗时、per-client 指标，并将常用 Makefile 启动命令切换到 early stopping。
- 已补 dashboard 主指标高亮、AUC 主图降噪、adaptive rounds、2D/3D 真实数据一键入口。

仍未完成或需要下一阶段推进：

- 真实多进程/多主机传输层 TLS/mTLS 与证书生命周期。
- dashboard/report export：final vs best、目标达标成本、隐私配置、链上审计摘要的一键导出。
- per-client 评估与公平性/异质性指标。
- 生产级 secure aggregation 与正式差分隐私会计。
- Hardhat 3 迁移或继续作为隔离的可选本地链适配器。

## 1. 当前项目判断

FLxBC 当前已经具备本地端到端演示能力：

- 联邦学习模拟：`FedAvg`、`FedProx`、`BC-CA-FedProx`。
- 数据路径：synthetic smoke demo、PneumoniaMNIST、NoduleMNIST3D。
- 训练后端：NumPy prototype 与 optional PyTorch CNN。
- 审计路径：SQLite ledger、mock audit chain、可选 Hardhat 合约。
- 展示路径：FastAPI 只读接口与 Streamlit dashboard。

当前主要限制：

- 训练仍是单进程模拟，客户端参数以内存对象直接进入聚合器。
- 没有真实参数传输层，也没有 TLS/mTLS、请求签名、防重放等通信安全机制。
- 没有参数更新加密，也没有安全聚合。
- `dp_noise_multiplier` 和 `clipping_norm` 只存在于配置，尚未进入训练或聚合路径。
- 全局参数 artifact 以 `.npz` 明文落盘。
- `model_hash` 当前哈希的是 artifact URI 和 round id，不是参数文件内容。
- API 与 dashboard 默认本地只读，但没有认证、角色控制或脱敏策略。
- Hardhat 依赖树仍有 `npm audit` 漏洞，且本机 Node v25.8.2 与项目声明的 Node 22 不一致。
- `ruff check .` 当前有 2 个格式问题。

## 2. 优先级

### P0: 隐私与真实场景补强

P0 是下一阶段主线，目标是建立一个最小可演示但逻辑完整的隐私闭环：

- 明确威胁模型与隐私边界。
- 引入参数更新 envelope，让客户端更新不再作为裸参数对象直接传递。
- 对客户端上传加入身份认证、签名、nonce 和时间戳。
- 对参数更新和落盘 artifact 加密。
- 实现 DP clipping/noise，并记录隐私配置和实验指标。
- 实现安全聚合 demo，使服务端可验证流程但不直接依赖单个明文客户端更新。
- 将审计哈希改为基于 artifact 内容和隐私配置，而不是仅基于 URI。

### P1: 研究可信度与实验完整性

- 增加隐私增强前后对比实验。
- 增加恶意更新、重放提交、artifact 篡改等攻击/异常测试。
- 明确 BC-CA-FedProx 中 blockchain、contribution、audit、reputation 的职责边界。
- 产出适合论文/报告的固定结果表、运行摘要和 mock-chain 边界说明。

### P2: 工程交付质量

- 修复 lint。
- 固定 Node 22 使用方式。
- 处理或隔离 Hardhat 2 依赖风险。
- 增加合约测试。
- 清理运行产物，形成干净交付包。
- 加入 CI 检查。

## 3. 文件责任规划

建议新增或修改以下文件：

- Create: `docs/privacy-threat-model.md`
  - 说明真实医疗 FL 场景中的资产、攻击者、信任边界、隐私目标和非目标。
- Create: `src/flxbc/privacy.py`
  - 提供参数裁剪、DP 噪声、内容哈希、参数 envelope 数据结构、隐私配置摘要。
- Create: `src/flxbc/secure_aggregation.py`
  - 提供本地 demo 级安全聚合协议接口与 mask-cancellation 实现。
- Create: `src/flxbc/crypto.py`
  - 提供 AES-GCM artifact 加密、HMAC/签名 envelope、nonce 检查所需工具函数。
- Modify: `src/flxbc/config.py`
  - 增加 `privacy_mode`、`transport_security`、`artifact_encryption`、`secure_aggregation`、`client_auth` 等配置项。
- Modify: `src/flxbc/training.py`
  - 在本地训练结果生成后接入 clipping 和 DP 噪声。
- Modify: `src/flxbc/simulation.py`
  - 将本地训练结果包装为 envelope，再进入验证、解密、聚合、审计流程。
- Modify: `src/flxbc/strategy.py`
  - 让聚合函数支持安全聚合输出或已经验证的 update envelope。
- Modify: `src/flxbc/ledger.py`
  - 增加 artifact 内容哈希、隐私配置哈希、client auth hash、secure aggregation 状态记录。
- Modify: `src/flxbc/api.py`
  - 加入基础认证、敏感字段过滤、隐私状态接口。
- Modify: `dashboard/app.py`
  - 展示隐私模式、DP 参数、artifact 加密状态、安全聚合状态和审计完整性。
- Modify: `src/flxbc/cli.py`
  - 暴露隐私相关 CLI 参数。
- Create: `tests/test_privacy.py`
  - 覆盖 clipping、DP 噪声、envelope 内容哈希、artifact 加密。
- Create: `tests/test_secure_aggregation.py`
  - 覆盖 mask 聚合正确性和缺失客户端处理。
- Create: `tests/test_simulation_privacy.py`
  - 覆盖开启隐私模式后的完整 demo smoke run。

## 4. 实施里程碑

### Milestone 1: 工程基线修复

**Files:**

- Modify: `src/flxbc/simulation.py`
- Modify: `package.json`
- Modify: `README.md`
- Optionally create: `.nvmrc`

- [ ] 修复 `ruff check .` 当前失败项。
  - 移除 `src/flxbc/simulation.py` 中无占位符的 f-string。
  - 拆分 `src/flxbc/simulation.py` 中超过 100 字符的输出行。
- [ ] 增加 Node 22 使用说明。
  - 若新增 `.nvmrc`，内容为 `22`。
  - README 中说明 Hardhat 只作为 local-only optional adapter。
- [ ] 重新验证：
  - `uv run --extra app --extra dev pytest -q`
  - `uv run --extra dev ruff check .`
  - `npm run compile`

**Acceptance:**

- Python 测试全通过。
- Ruff 全通过。
- Hardhat compile 可运行；若仍有 Node 版本警告，README 必须明确处理方式。

### Milestone 2: 威胁模型与隐私边界

**Files:**

- Create: `docs/privacy-threat-model.md`
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`

- [ ] 写清楚系统保护对象：
  - 原始医疗影像数据。
  - 客户端模型参数或梯度更新。
  - 医院真实身份。
  - artifact 文件。
  - 审计链元数据。
- [ ] 写清楚攻击者模型：
  - 半诚实聚合端。
  - 恶意客户端。
  - 被动网络窃听者。
  - 本地 artifact 读取者。
  - dashboard/API 未授权访问者。
- [ ] 写清楚本项目非目标：
  - 不承诺生产级临床合规。
  - 不承诺生产级区块链部署。
  - 不在本地 demo 中处理真实患者数据。
- [ ] 写清楚最小隐私闭环：
  - 客户端上传 envelope。
  - 上传认证与防重放。
  - DP clipping/noise。
  - 可选安全聚合。
  - artifact 内容哈希与加密落盘。

**Acceptance:**

- 后续实现者能根据文档判断每个隐私功能为什么存在、保护什么、不保护什么。

### Milestone 3: 隐私配置与 CLI 接入

**Files:**

- Modify: `src/flxbc/config.py`
- Modify: `src/flxbc/cli.py`
- Modify: `README.md`
- Modify: `tests/test_packaging.py`

- [ ] 增加配置字段：
  - `privacy_mode`: `none`, `dp`, `encrypted`, `secure-aggregation`, `full-demo`
  - `client_auth`: `none`, `hmac-demo`
  - `artifact_encryption`: `False` by default
  - `secure_aggregation`: `False` by default
  - `replay_window_seconds`: default `300`
- [ ] CLI 增加对应参数：
  - `--privacy-mode`
  - `--dp-noise-multiplier`
  - `--clipping-norm`
  - `--artifact-encryption`
  - `--secure-aggregation`
  - `--client-auth`
- [ ] README 增加隐私 demo 示例：
  - synthetic privacy run。
  - DP only run。
  - full demo privacy run。

**Acceptance:**

- 默认命令行为保持兼容。
- 开启隐私参数后，`config.json` 能完整记录隐私设置。

### Milestone 4: 参数 Envelope、内容哈希与上传验证

**Files:**

- Create: `src/flxbc/privacy.py`
- Modify: `src/flxbc/simulation.py`
- Modify: `src/flxbc/ledger.py`
- Create: `tests/test_privacy.py`
- Modify: `tests/test_ledger.py`

- [ ] 定义 `ParameterEnvelope`：
  - `run_id`
  - `round_id`
  - `node_id`
  - `payload_hash`
  - `payload_bytes`
  - `timestamp`
  - `nonce`
  - `privacy_mode`
  - `signature`
- [ ] 使用稳定二进制序列化计算参数内容哈希。
  - 参数 artifact 的 `model_hash` 必须来自文件内容或参数 payload 内容。
  - 审计 payload 中保留 `artifact_uri`，但不能以 URI 作为模型完整性依据。
- [ ] 在 simulation 中加入 envelope 验证流程。
  - 缺少 hash、nonce 或 node id 不匹配时拒绝聚合。
  - 重复 nonce 记录为 misbehavior。
- [ ] ledger 增加 envelope 验证结果记录。
  - 每轮记录 accepted/rejected client count。
  - misbehavior 中记录 `replay`, `invalid-signature`, `hash-mismatch`。

**Acceptance:**

- 篡改参数 payload 会导致 hash 校验失败。
- 重放同一 envelope 会被拒绝并写入 misbehavior。
- audit block 中的 model hash 与实际参数内容绑定。

### Milestone 5: 客户端认证与传输安全 Demo

**Files:**

- Create: `src/flxbc/crypto.py`
- Modify: `src/flxbc/privacy.py`
- Modify: `src/flxbc/simulation.py`
- Create: `tests/test_crypto.py`
- Modify: `README.md`

- [ ] 实现 demo 级 HMAC envelope 签名。
  - 每个模拟医院使用独立 secret。
  - secret 只用于本地 demo，不写入 ledger。
  - signature 覆盖 run id、round id、node id、payload hash、timestamp、nonce。
- [ ] 实现 timestamp 和 nonce 防重放。
  - 超过 `replay_window_seconds` 的 envelope 拒绝。
  - 同一 node 在同一 run/round 重复 nonce 拒绝。
- [ ] 文档中明确传输安全边界。
  - 当前没有真实网络服务时，HMAC 和 envelope 验证模拟真实客户端提交。
  - 若后续拆成多进程/多机器，必须在 HTTP/gRPC 层启用 TLS 或 mTLS。

**Acceptance:**

- 错误 secret 的 signature 无法通过。
- 改动 payload 后原 signature 无法通过。
- 文档明确 TLS/mTLS 属于多机器部署时的必选项。

### Milestone 6: DP Clipping 与 Noise 实装

**Files:**

- Modify: `src/flxbc/training.py`
- Modify: `src/flxbc/privacy.py`
- Modify: `src/flxbc/simulation.py`
- Modify: `src/flxbc/config.py`
- Create: `tests/test_privacy.py`
- Create: `tests/test_simulation_privacy.py`

- [ ] 在客户端本地更新上执行 L2 clipping。
  - clipping 作用于 local parameters 与 global parameters 的差值。
  - clipping 后再恢复成可上传参数。
- [ ] 当 `dp_noise_multiplier > 0` 时加入 Gaussian noise。
  - noise std 使用 `clipping_norm * dp_noise_multiplier`。
  - 使用 run seed、round id、node offset 生成可复现实验噪声。
- [ ] metrics 中记录 DP 状态。
  - `dp_enabled`
  - `clipping_norm`
  - `dp_noise_multiplier`
  - `mean_update_norm_before_clip`
  - `mean_update_norm_after_clip`
- [ ] dashboard 展示 DP 配置和 update norm 变化。

**Acceptance:**

- `dp_noise_multiplier=0` 时行为与当前非 DP 模式尽量一致。
- 开启 DP 后，测试能证明 update norm 被 clipping 限制。
- 相同 seed 下 demo 可复现。

### Milestone 7: 安全聚合 Demo

**Files:**

- Create: `src/flxbc/secure_aggregation.py`
- Modify: `src/flxbc/simulation.py`
- Modify: `src/flxbc/strategy.py`
- Create: `tests/test_secure_aggregation.py`
- Create: `tests/test_simulation_privacy.py`

- [ ] 实现本地 mask-cancellation 协议。
  - 每个客户端 update 加 mask。
  - 聚合端只处理 masked updates。
  - mask 总和为零，最终聚合结果与明文聚合一致。
- [ ] 支持 dropout 策略。
  - demo 版本可在有 dropout 时回退到普通加密 envelope 聚合。
  - 回退必须写入 metrics 和 audit payload。
- [ ] ledger 记录 secure aggregation 状态。
  - `secure_aggregation_enabled`
  - `secure_aggregation_status`
  - `secure_aggregation_fallback_reason`
- [ ] dashboard 显示安全聚合状态。

**Acceptance:**

- 无 dropout 时，secure aggregation 输出与明文聚合数值一致。
- 有 dropout 时，系统不静默失败，必须记录回退原因。
- 测试覆盖普通聚合、mask 聚合和回退路径。

### Milestone 8: Artifact 加密落盘

**Files:**

- Modify: `src/flxbc/crypto.py`
- Modify: `src/flxbc/simulation.py`
- Modify: `src/flxbc/ledger.py`
- Create: `tests/test_artifact_encryption.py`
- Modify: `README.md`

- [ ] 增加 AES-GCM artifact 加密。
  - 明文 `.npz` 只在内存或临时路径存在。
  - 加密后落盘扩展名建议为 `.npz.enc`。
  - ledger 记录 ciphertext hash、nonce hash、encryption mode。
- [ ] 提供本地 demo key 来源。
  - 支持 `FLXBC_ARTIFACT_KEY` 环境变量。
  - 若未提供 key 且开启 encryption，命令应失败并给出明确错误。
- [ ] dashboard/API 默认不暴露解密 key 或完整本地路径。

**Acceptance:**

- 开启加密后 artifact 不能被 `np.load` 直接读取。
- 解密后内容 hash 与 ledger 记录一致。
- 未设置 key 时不会降级为明文。

### Milestone 9: API 与 Dashboard 隐私化

**Files:**

- Modify: `src/flxbc/api.py`
- Modify: `dashboard/app.py`
- Modify: `src/flxbc/ledger.py`
- Create: `tests/test_api_privacy.py`

- [ ] API 增加 demo token 认证。
  - 支持 `FLXBC_API_TOKEN`。
  - 未设置 token 时维持本地开发便利，但 README 必须说明只适合 localhost demo。
- [ ] API 增加字段过滤。
  - 默认不返回完整 artifact path。
  - 默认不返回未脱敏 node display name。
- [ ] dashboard 增加隐私状态卡片。
  - Privacy mode。
  - DP enabled。
  - Artifact encrypted。
  - Secure aggregation status。
  - Audit content hash valid。
- [ ] 增加节点 pseudonym 展示。
  - 默认显示 `hospital_1` 等模拟 ID。
  - 若未来接入真实机构名，必须单独保存映射，不写入公开 ledger。

**Acceptance:**

- 设置 `FLXBC_API_TOKEN` 后，未携带 token 的 API 请求返回 401。
- dashboard 可以展示隐私状态，但不泄露 secret、key、完整 artifact 明文路径。

### Milestone 10: 实验、报告与交付

**Files:**

- Modify: `src/flxbc/simulation.py`
- Modify: `src/flxbc/presentation.py`
- Modify: `dashboard/app.py`
- Create: `docs/mock-chain-boundary.md`
- Create: `docs/privacy-experiment-results.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `README.md`

- [ ] 增加固定实验矩阵。
  - baseline: no privacy。
  - DP only。
  - encrypted artifact。
  - secure aggregation demo。
  - full-demo privacy mode。
- [ ] 每次实验输出统一 summary。
  - final accuracy。
  - macro-F1。
  - rounds。
  - participants。
  - privacy mode。
  - average update norm。
  - audit status。
- [ ] 写 `docs/mock-chain-boundary.md`。
  - 解释 mock chain 只用于本地可验证审计，不等价于生产区块链。
  - 解释 Hardhat adapter 是迁移可行性证据，不是默认运行时。
- [ ] 写 `docs/privacy-experiment-results.md`。
  - 记录隐私增强对性能、可复现性、存储和运行时间的影响。

**Acceptance:**

- `make demo-check` 仍可快速通过。
- 至少一个 privacy demo 命令可在本地完成完整训练、审计、dashboard 展示。
- 报告文档能直接说明项目边界和隐私增强收益。

## 5. 建议执行顺序

推荐按以下顺序推进：

1. Milestone 1: 工程基线修复。
2. Milestone 2: 威胁模型与隐私边界。
3. Milestone 3: 隐私配置与 CLI 接入。
4. Milestone 4: 参数 envelope、内容哈希与上传验证。
5. Milestone 6: DP clipping 与 noise。
6. Milestone 8: artifact 加密落盘。
7. Milestone 7: 安全聚合 demo。
8. Milestone 9: API 与 dashboard 隐私化。
9. Milestone 10: 实验、报告与交付。
10. Hardhat 迁移或隔离策略单独决策。

安全聚合放在 DP 和 artifact 加密之后，是因为当前项目没有真实网络传输层，先完成 envelope、hash、DP、加密落盘可以更快形成可验证闭环。

## 6. 测试与验收命令

每个开发阶段至少运行：

```bash
uv run --extra app --extra dev pytest -q
uv run --extra dev ruff check .
```

涉及真实 MedMNIST/PyTorch 路径时运行：

```bash
uv run --extra app --extra ml flxbc demo --dataset pneumoniamnist --rounds 1 --clients 2 --max-train-samples 32 --max-test-samples 16 --no-failures
```

涉及链上合约路径时运行：

```bash
npm run compile
npm audit --json
```

涉及 dashboard/API 时运行：

```bash
FLXBC_DB=data/flxbc.db uv run --extra app uvicorn flxbc.api:app --host 127.0.0.1 --port 8000
FLXBC_DB=data/flxbc.db uv run --extra app streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

## 7. 完成定义

下一阶段可认为完成，当以下条件全部满足：

- 隐私模式有清晰文档，并且 README 中有可运行命令。
- 开启 `privacy_mode=full-demo` 后，训练、envelope 验证、DP、artifact 加密、审计记录、dashboard 展示形成闭环。
- 参数完整性哈希基于内容，而不是 artifact URI。
- 测试覆盖隐私核心逻辑、secure aggregation demo、artifact 加密、API token。
- `pytest` 与 `ruff` 通过。
- Hardhat 风险被明确隔离或完成迁移决策。
- `PROJECT_STATUS.md` 更新为新的真实状态，不再只描述旧 demo 能力。

## 8. 不建议立即做的事项

- 不建议直接做生产级医院多方部署。
- 不建议直接接真实患者数据。
- 不建议把 Solidity 合约作为默认运行时。
- 不建议在没有威胁模型的情况下堆叠复杂密码协议。
- 不建议为了展示效果牺牲默认 demo 的快速可运行性。
