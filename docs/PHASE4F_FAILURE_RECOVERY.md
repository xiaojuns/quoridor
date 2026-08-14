# Phase 4F 固定预算失败诊断与恢复准备

> 状态：Phase 4F6 CPU-only launcher/driver preparation completed，正式恢复训练尚未启动
>
> 日期：2026-08-04
>
> Phase 4F5 已冻结 14 GPU-hours 正式恢复训练合同，Phase 4F6 已完成当前源码验签与可恢复
> 启动器；真正启动 CUDA 训练仍需下一阶段确认。本文不授权 wall-clock arena 或产品指针更新。

## 1. 已验证结论

Phase 4F fixed track 使用冻结的 `final` 200 个开局对、候选每手 800 次
native MCTS simulation，以及经典困难 AI 每手最多 587 个节点。402 局证据已经由
Python 参考规则逐步重放：400 局 final arena 和 2 局标准初始局面均不存在非法动作、
状态哈希错误或终局不一致。

固定门禁结果：

| 指标 | 实际结果 | 产品门槛 |
| --- | ---: | ---: |
| 胜负 | 71-329 | - |
| 胜率 | 17.75% | >= 60% |
| 95% CI | [14.25%, 21.50%] | 下界 >= 55% |
| Elo | -266.38 | - |
| 标准初始局面 | 0-2 | 仅完整性检查 |

因此失败是实际棋力差距，不是 arena、身份绑定或统计实现错误。停止 wall-clock track；
它不能改变 fixed hard gate 的失败结论。

机器可读报告位于
`training/runs/phase4/phase4f/failure-analysis/report.json`，报告自哈希为
`39676508b615d0777de668a8fedf9dd865f1c26f14122b27b5a8d1134dcab278`。

## 2. 失败结构

### 2.1 成对开局普遍失败

- 200 个开局对中，132 对两盘全负，占 66%；
- 65 对取得一胜，只有 3 对两盘全胜；
- 这排除了“只是不适应某一先后手”的单一解释。

### 2.2 玩家方向不对称

- 候选作为原始 player 0：45/200，胜率 22.5%；
- 候选作为原始 player 1：26/200，胜率 13.0%；
- 9.5 个百分点的差值说明状态规范化或训练数据方向覆盖需要专项审计。

这不是规则编码错误的直接证据。现有差分与逐步回放均通过；更可能的问题是数据分布、
增强覆盖或模型学习后的统计不对称。

### 2.3 早期墙战最弱

按开局时已经落下的墙数分组：

| 已落墙数 | 局数 | 胜率 |
| ---: | ---: | ---: |
| 0-4 | 120 | 11.67% |
| 5-9 | 160 | 15.00% |
| 10-14 | 80 | 23.75% |
| 15-20 | 40 | 35.00% |

候选在墙较少、策略分支最大的阶段差距最大。恢复数据应加强早期墙响应，但不得来自本次
final 轨迹。

### 2.4 价值网络在败局中过度自信

- 329 个败局包含 8,784 个候选决策；
- 其中 4,839 个决策仍给出 `value >= 0.5`，占 55.09%；
- 败局候选 value 中位数为 `0.6241`；
- 在每局第二四分位，败局 `value >= 0.5` 的比例达到 74.39%；
- 首次候选决策的胜负符号准确率只有 57.5%。

搜索最终能在末段识别败势，但中前盘价值评估长期高估自身。这会让 PUCT 反复选择无法
被 800 次 simulation 及时证伪的墙线。

## 3. 数据隔离

本次 `final` 套件已经用于失败诊断。为了避免评测泄漏：

1. 禁止把 Phase 4F final 的精确局面、动作、value 或结果加入训练和教师数据；
2. 禁止围绕单个 final 开局手工调参；
3. 恢复数据必须从新种子生成，并与 `dev`、旧 `final` 做内容哈希和镜像轨道隔离；
4. 下一次产品门禁必须使用新冻结、此前从未查看的 final suite；
5. 旧 final 只保留为历史回归证据，不能再声明为未见测试集。

## 4. 推荐恢复方案

### 4.1 GPU 前工程

1. 增加绑定 provenance 的 generation-14 champion model-only 初始化，不导入旧 optimizer；
2. 增加可配置的经典困难 AI teacher/challenge 数据混合；
3. 在 generation health 中记录原始玩家方向和 value calibration；
4. 生成相互隔离的 recovery-dev 与新 final suite；
5. 源码变化后重新执行原生差分、replay-cache 和显存 smoke 门禁。

### 4.2 训练工作负载

- 网络保持 `192x16`，避免同时改变模型容量；
- 从已发布 generation 14 模型参数开始，重置 optimizer；
- self-play 从 256 提升到 512 simulations；
- 正式 arena 维持 800 simulations；
- 加强 0-4 墙早期局面、player 0/1 平衡、经典墙战反例和价值高估反例；
- 反例必须由新生成、内容隔离的数据产生。

### 4.3 剩余预算

Phase 4E 剩余约 26.850 GPU-hours，冻结上限建议：

| 用途 | 上限 |
| --- | ---: |
| 恢复训练 | 14.0 GPU-hours |
| development screens | 3.0 GPU-hours |
| 新 final fixed gate 预留 | 9.5 GPU-hours |

wall-clock gate 只在新 fixed gate 通过后运行；它不占用上述恢复训练额度。

## 5. 晋升顺序

1. CPU correctness、来源和数据隔离门禁；
2. 64 局 recovery-dev classic-hard screen；
3. 400 局 recovery-dev classic-hard confirmation；
4. 对当前 generation-14 champion 的历史非退化门禁；
5. 新的未见 final 400 局 fixed-budget 产品门禁；
6. fixed 通过后才执行 wall-clock 门禁。

任何一步失败都停止后续 GPU 消耗。

## 6. Phase 4F2 完成状态

Phase 4F2 已在不访问 CUDA 的条件下完成：

- 新增 `initialization.mode=champion`，同时绑定 champion id、模型文件、manifest 文件、
  manifest 自哈希和 tensor-state SHA-256；只导入模型参数并重置 optimizer；
- 新增 `training.challengeReplay`，按固定比例与 self-play tensor batch 混合；外部 replay
  必须通过 completed contract、数据集清单、train manifest 和逐局深度重放；
- generation 报告新增原始 player 0/1 胜率差、决策数、value 符号准确率、MSE 和高置信
  错误率；
- 冻结 200 对 `recovery-dev-v1` 和 200 对 `recovery-final-v1`，四套历史/恢复开局在内容
  哈希和水平镜像 position orbit 上全部隔离；
- challenge 教师生成器会在生成和恢复深验时重新计算局面镜像轨道，拒绝四套评测开局泄漏。
- 排除集合还逐文件绑定并重放旧 fixed 400 局和标准初始 2 局的完整 arena 证据；四套开局与
  历史轨迹合并后共有 21,449 个禁止使用的水平镜像局面轨道。

机器可读准备报告：

`training/runs/phase4/phase4f/recovery-preparation/report.json`

报告自哈希：

`04053d33ca11b4674532bc3986af53d39b24b66475db9ad434a30f370c2d9d1c`

准备时 GPU 总账保持 `43.14977626422679` 小时。恢复工作负载仍为不可执行草案，
`trainingAuthorized=false`；原因是 challenge replay 尚未生成/绑定，且源码变化后的 CPU
正确性门禁和后续训练 rebaseline 尚未冻结。

Phase 4F3 CPU-only challenge 数据生成使用命令：

```bash
./scripts/generate-phase4f-challenge-data.sh
```

生成结果已经完成并由 Phase 4F4 深验、冻结，详见下一节。

## 7. Phase 4F4 完成状态

Phase 4F4 在不访问 CUDA 的条件下完成了 challenge replay 冻结和 recovery workload 的
rebaseline 准备：

- challenge 数据集包含 `30,047` 个局面，其中 train 为 `27,025` 局面/`538` 局，
  holdout 为 `3,022` 局面/`60` 局；数据集指纹为
  `2a4864ad464e0448b405e4e8879b40c06249dc2bcea420fcabd204ebbd8b9e95`；
- train 与 holdout 均通过 ReplayStore 深度重放，全部 `21,449` 个禁止评测轨道重新计算
  一致；
- completed challenge contract 为
  `training/configs/phase4f-recovery-challenge-frozen-v1.json`，合同自哈希为
  `f8b30795269ca9ec17f5e0beec145e5932d4cb92ebe07eb4ca4afe780abac8f7`；
- recovery workload 为 `training/configs/phase4f-recovery-v1.json`，文件 SHA-256 为
  `74f39388aef5426a8f821c179a5e1da0a45272b48112abcc375ca798913fd7cf`，语义配置 SHA-256
  为 `5cba128159e943646259738d1aa6ad707f40ee7ada08cb522ece286c6686eb69`；
- workload 使用 generation-14 champion model-only 初始化并重置 optimizer，self-play 为
  `512 simulations`，challenge replay 比例为 `0.25`，网络仍为 `192x16`、batch 为 `512`；
- CUDA rebaseline 准备合同为
  `training/configs/phase4f-recovery-rebaseline-prepared-v1.json`，自哈希为
  `527d867c9d4253bbd3921afe8a2e5823ef00dc4f79c59cb27cccdab02216d505`；
- CPU freeze 报告为
  `training/runs/phase4/phase4f/recovery-freeze/report.json`，自哈希为
  `c2a27d4c85df8c9732d5fe9f043c58f3ea155806050a7d59320bc3b28cbd1472`。

准备前后 global GPU ledger 均为 `43.14977626422679 GPU-hours`，文件 SHA-256 保持
`703592d0ccf70dbdf1fdaffeac70a446cff47778b5c365ca9709902e35085021`。Phase 4 产物约
`3.2 GiB`，磁盘剩余约 `75 GiB`。原生测试 3/3、完整 Python 测试 403/403 均通过。

completed challenge contract 允许训练器读取 challenge replay，但 recovery workload 仍为
`trainingAuthorized=false`。旧 Phase 4E1 报告没有批准这个新的 mixed workload；下一阶段
必须单独确认并执行 **Phase 4F5 CUDA recovery rebaseline**，至少覆盖 current-source
192x16/batch-512 AMP 显存、mixed replay 输入吞吐、可恢复一代 pilot 和 GPU budget freeze。
Phase 4F5 仍不是正式恢复训练。

冻结之后不要再运行 `generate-phase4f-challenge-data.sh`，包括 `--verify-only`。教师验证器会
重写 manifest 的 `updatedAt`，从而按设计使 frozen contract 的文件哈希绑定失效。只读重验
使用：

```bash
./scripts/freeze-phase4f-recovery.sh --verify-only
```

## 8. Phase 4F5 CUDA Rebaseline 准备状态

Phase 4F5 的 CPU preparation 已完成，尚未消耗 GPU：

- current source SHA-256 为
  `6c6da19633ac648c8bc0c7d124ebf2ff8d05a0bce9fd3e6c1f8ef4679bdf5a0e`；
- preparation 自哈希为
  `c1554613bd125705e2879f536f03f11eea477e716246cbcaf779f48d190215b1`；
- 校准执行合同 SHA-256 为
  `d2d939b97d55def02e6bef772a2693f51aba87c754e53d40df997ae1583e8e53`；
- 正式 recovery v2 配置文件 SHA-256 为
  `acb14947dbf48acf2778fc6b872feeef0000ffd4b2d33be6978c2954fa72214e`；pilot v2 配置文件
  SHA-256 为 `e8b16fcbf30e8265526c3140edb5e74ce9bc12ed886030536373518c4bd82214`，有效语义
  SHA-256 为 `41b10fd9d4f56fafa2f5522a93415d8de220423c300f1aa1ef1d98da42fc116e`；
- GPU 基线为 `43.14977626422679 / 72 GPU-hours`，本阶段硬上限为 `2 GPU-hours`；首次
  三次 smoke、一次基模校验失败和一次 checkpoint schema 失败均已计费，当前总账为
  `43.29081197835091 GPU-hours`，剩余额度为 `1.858964285875878 GPU-hours`；
- 后续仍预留正式恢复训练 `14`、development screens `3`、新 final `9.5 GPU-hours`；
- native 测试 3/3、完整 Python 测试 410/410 通过。

最初的 `prepared-v1` 在运行时暴露出 `runDir` 相对路径与绝对路径的配置身份误判，pilot 在
self-play 启动前被拒绝。`prepared-v2` 保留原账本和 2 小时总上限，同时分别绑定配置文件语义
指纹与路径解析后的有效语义指纹；v1 产物保留为不可变历史证据。

`prepared-v2` 随后通过了配置授权，但暴露 F2 workload 将旧种子基模 SHA-256 用于新 recovery
种子 `20260803` 的问题；driver 在 champion 加载前正确拒绝了不一致的随机基模。`prepared-v3`
冻结 recovery/pilot v2，把 `expectedBaseModelSha256` 修正为实际按新种子构造 192x16 网络的
canonical hash `87e87ec6f4f77b6156d56e095b1c23685d10d33cf14038ca4623b8a89d0b0120`，并使用新的
`pilot-v2` runDir。Generation-14 champion artifact、manifest 和模型张量已再次 CPU 深验通过；
旧 pilot manifest 与失败账本记录完整保留。

`prepared-v3` 成功加载 champion 并完整构建 challenge tensor cache，随后暴露 checkpoint
初始化合同只允许 `random/model`、遗漏 `champion` provenance 的问题；错误文本中的“empty
replay”并非实际 replay 非空。`prepared-v4` 增加严格 champion checkpoint schema：要求精确
字段集、安全且同目录的 artifact/manifest 相对路径、generation/id、初始化模型状态一致及七项
SHA-256 provenance。生产 generation-14 artifact 已直接送入该 validator 做 CPU 复核并通过；
新运行目录为 `pilot-v3`，旧完整 cache 保留但不跨源码复用。

## 9. Phase 4F5 完成状态

最终 v4 运行完成并通过只读幂等复核：

- memory smoke 峰值 allocated/reserved 为 `1,610,482,176 / 1,723,858,944 bytes`；
- pilot 完成 `128` 局、`5,968 plies`，全部使用 `512 simulations` 和 `64 actors`；
- 终局为 goal `124`、repetition `4`，无 draw 或损坏 replay；
- 完成 `96` optimizer steps，batch `512`，challenge replay 比例 `0.25`；
- 输入 batch 平均 `0.051227s`、最大 `0.815025s`，通过 mean `2s` / live `5s` 门禁；
- pilot checkpoint SHA-256 为
  `8452cbc789a94d071f0fea63b069d3d06003d9f19c7a5e62f61a9c43452c38ab`；
- Phase 4F5 总用量 `0.8179142071425716 GPU-hours`，global ledger 为
  `43.96769047136936 / 72 GPU-hours`，剩余 `28.03230952863064 GPU-hours`；
- Phase 4 产物约 `4.1 GiB`，可用磁盘约 `74 GiB`。

完成报告自哈希为
`55f08d39ac7374af67b6e60682868689dc074bbba3adcba8e7532093939be8b2`，freeze receipt 自哈希为
`454b78a317fbc40d53edb3111cd7e4cd487f7a282beba3b228d15691b22344ec`。正式恢复训练执行合同为
`training/configs/phase4f-recovery-execution-frozen-v1.json`，SHA-256 为
`58e3eaff20acd4f6e737dab99bb5b3811b4a2620f94afad62501fac5cb755184`，固定 baseline
`43.96769047136936`、上限 `14 GPU-hours`、最多 `16` generations。

报告明确记录 `formalRecoveryTrainingAuthorized=true` 和 `trainingStarted=false`。下一阶段必须先
单独确认 **Phase 4F6 CPU-only formal recovery launcher/driver preparation**，让训练驱动严格验签
`phase4f-recovery` stage、F5 报告、正式 workload 和新 stage ledger；该准备阶段不得启动 CUDA。

用户在本机运行：

```bash
./scripts/run-phase4f5-rebaseline.sh --device-index 0 --gpu-hours-used 0
```

脚本依次执行 current-source CPU replay gate、192x16/batch-512 CUDA AMP memory smoke、
一代 `128 games / 512 simulations / 96 optimizer steps / 25% challenge replay` pilot，最后
验证结果并冻结正式恢复训练合同。`Ctrl+C` 后可用同一命令重跑；完整 self-play、checkpoint
和 GPU 计费均按既有持久化协议恢复。脚本完成只会设置
`formalRecoveryTrainingAuthorized=true`，不会自动启动正式恢复训练。

## 10. Phase 4F6 CPU-only Launcher/Driver Preparation 完成状态

Phase 4F6 已在不访问 CUDA、且不创建正式 lineage 的条件下完成：

- 训练驱动新增 `phase4f-recovery` stage，只接受 F5 receipt、完成报告、正式执行合同和
  `recovery-v2` workload 的完整交叉验签；所有检查在 GPU ledger `begin()` 和 lineage 创建前完成；
- F5 历史验签会验证 canonical JSON、自哈希、文件哈希、memory smoke、pilot report/run
  manifest、预算和执行合同，不要求恢复旧源码；
- 当前源码、native 后端、运行时有效配置和 GPU 台账追加性由新的 F6 preparation 绑定；
- 新启动器 `scripts/run-phase4f-recovery.sh` 使用绝对 generation target，自动识别新建/续训，
  并在 CUDA 前执行 native 测试和 current-source CPU replay-input gate；
- `Ctrl+C` 后保留完整 replay games 和已提交 checkpoint；同一 target 重跑可幂等续训。

不可变 preparation：

`training/configs/phase4f6-recovery-prepared-v1.json`

preparation 自哈希为
`2699ceb08952d9c1cd7e90afd3ee52ac7a2d3b60136a9528ba6411a9a8c387b7`，当前源码 SHA-256 为
`ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`。准备前后 global GPU
ledger 均为 `43.96769047136936 / 72 GPU-hours`；正式恢复上限仍为 `14 GPU-hours`，最多
`16 generations`。Phase 4 产物为 `4,376,379,944 bytes`，可用磁盘约 `73.3 GiB`。
Native 3/3、完整 Python 415/415 均通过。

首次正式训练建议只推进到绝对 target 4。获得下一阶段确认后运行：

```bash
./scripts/run-phase4f-recovery.sh \
  --target-generations 4 \
  --device-index 0 \
  --gpu-hours-used 0 \
  --new
```

中途停止使用 `Ctrl+C`。已有 `run-manifest.json` 后，续训时使用同一命令并去掉 `--new`，脚本会
自动恢复；也可显式传入 `--resume`。完成 target 4 后必须先审查四代报告、资源用量、
replay/cache、输入吞吐和健康指标，再决定是否提高 target，不自动运行 development arena。

## 11. Phase 4F Recovery Target 4 完成状态

首个正式恢复训练分块已完成 generation `0-3`、step `384`：共生成 `512` 局、`24,866`
plies，四代均完成 `96` 个 batch-512 optimizer steps，并保持 `25%` challenge replay。
最终 replay tensor cache 为 `512` 局、`24,866` plies、`299,495,058 bytes`，通过 deep
verification 和 raw/cache append-only 绑定。

训练健康趋势：

- mean loss 从 generation 0 的 `2.186555` 降至 generation 3 的 `1.923746`；
- value MSE 从 `0.927367` 降至 `0.581617`，value sign accuracy 从 `67.41%` 升至 `78.84%`；
- high-confidence wrong rate 从 `19.32%` 降至 `7.07%`；
- generation 3 的原始 player 0/1 胜率差为 `3.125%`，128 局全部以 goal 结束；
- input batch mean/max 为 `0.075420s / 0.654855s`，远低于 `2s / 5s` 门禁。

Generation 3 checkpoint SHA-256 为
`5b87052725ee354693ee6ad34496b4be8ffec7b81ae5ffaa0de534a75240da62`，candidate model
SHA-256 为 `dadf3628927006f1a5867cc9b23e0b0ec1be9416e297f735bd1b2043a2d1e806`，模型状态 SHA-256
为 `ed6f6247fdf0cf3b259b2fccc4179b54a813f46602c8d03241e7e8a297ba2021`。安全加载、有限张量、
checkpoint/candidate/report/run-manifest 绑定均通过，无 fallback。

本分块使用 `2.187431130028692 GPU-hours`，global ledger 为
`46.15512160139805 / 72 GPU-hours`；recovery stage 剩余 `11.812568869971308 GPU-hours`。
Phase 4 产物约 `5.02 GiB`，可用磁盘约 `72.31 GiB`。训练内指标证明运行健康，但不能证明
相对 classic-hard 的棋力提升。下一阶段必须单独确认，先为 generation 3 准备并执行隔离的
CPU health evidence 和 64 局 `recovery-dev-v1` screen；不得直接运行 400 局 confirmation、
提高训练 target 或触碰 `recovery-final-v1`。

## 12. Phase 4F7 Generation-3 Health 与 Development Screen 准备状态

Phase 4F7 的 CPU-only preparation 已完成，尚未启动 64 局 CUDA arena：

- generation 0-3 的报告、自哈希、checkpoint、candidate model、replay deep verification、
  input gate 和空 fallback 均已重新验证；health evidence 自哈希为
  `29374e5fde933e07a8d7cc60a0ca8cb1b874a81d339b53618838ec12df552b52`；
- generation 3 candidate 仍绑定 checkpoint
  `5b87052725ee354693ee6ad34496b4be8ffec7b81ae5ffaa0de534a75240da62` 和 model artifact
  `dadf3628927006f1a5867cc9b23e0b0ec1be9416e297f735bd1b2043a2d1e806`；
- 从 200 对 `recovery-dev-v1` 中按冻结 SHA-256 排序规则选择 32 对，screen suite hash 为
  `sha256:63408cf0bf09dc66b7f23b9ce55f2d4152f26dd564141e79fdb5171ce8c6c583`；
- screen 固定 candidate `800 simulations`、classic-hard `587 nodes / max depth 5`、32 actors、
  64 局和 90% paired bootstrap CI；
- screen 合同为 `training/configs/phase4f7-recovery-screen-frozen-v1.json`，自哈希为
  `13d103144c5703457b2b6c28ba72eacc423ce7d11117be31f7c9115e47b83c8b`；
- 开发评测执行器位于 `training/runs` 排除目录，但其文件 SHA-256 已由合同绑定，因此未改变
  recovery-v2 的训练源码 SHA-256
  `ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`，仍保留同 lineage 续训能力；
- CPU preparation 前后 global GPU ledger 均为 `46.15512160139805 GPU-hours`；development
  screen 上限为额外 `3 GPU-hours`，新 final 仍保留 `9.5 GPU-hours`。

用户运行以下可恢复命令：

```bash
PYTHONPATH="native/build/python:training" \
python3 training/runs/phase4/phase4f/recovery-evaluation-v1/phase4f7_screen.py \
  run --device cuda:0
```

执行器每 30 秒输出一次进度并按 opening pair 原子提交。`Ctrl+C` 后重复同一命令即可续跑。
结束时输出 `statistics` 与 `outcomeBand`：胜率低于 35% 为 `recovery-insufficient`，35% 至
50% 为 `continue-training`，至少 50% 为 `confirmation-candidate`。任何结果都不自动授权 400 局
confirmation、正式 promotion 或产品发布，下一阶段仍须单独确认。

## 13. Phase 4F7 64 局 Development Screen 结果

Generation 3 对 classic-hard 的 32 对/64 局 screen 已完成并通过逐步深度重放、身份、运行时和
预算验签：候选 25 胜 39 负、无 draw，胜率为 `39.0625%`，90% paired bootstrap CI 为
`[28.125%, 50.0%]`，Elo 为 `-77.25`，90% CI 为 `[-162.99, 0.0]`。

Gate 自哈希为
`ca509a14405679f282846672e11b2fe93358f28bbd04697a7cac3f523898cc31`，arena result 自哈希为
`a98e5dc0d047ebc4932a381f463eff198a4bcc72340b0c8e875e055b82085102`。结果按预先冻结的分档落入
`continue-training`：高于 35% 的恢复不足边界，但低于 50% 的 confirmation 候选边界，因此不得
运行 400 局 confirmation。

本次 screen 使用约 `0.613363 GPU-hours`，global ledger 为
`46.76848465306063 / 72 GPU-hours`，无 running ledger entry。Phase 4 产物约 `5.02 GiB`，磁盘
剩余约 `72.1 GiB`。下一阶段需单独确认：先 CPU-only 保留 generation 3 的 checkpoint/candidate
历史证据，避免 max-checkpoints=2 的正常裁剪破坏 screen provenance；再在不改源码、不摄取
screen 轨迹的条件下将同一 recovery-v2 lineage 从绝对 target 4 续到 target 8。

## 14. Phase 4F8 Recovery Target 8 准备状态

续训前的 CPU-only 归档和合同冻结已完成，未访问 CUDA：

- generation 3 的完整 checkpoint、candidate model、checkpoint metadata、run manifest 和
  generation report 已复制到
  `training/runs/phase4/phase4f/recovery-target8-preparation-v1/generation-3-archive`，共
  `193,248,744 bytes`；
- 归档清单自哈希为
  `df8806081a71faab7ae52ad27a119eb7703d2577f8d25cfb7fd8eb5b39d7fa87`，即使原训练目录按
  `maxCheckpoints=2` 正常裁剪 generation 3，screen 候选血缘仍可复核；
- target-8 续训合同为 `training/configs/phase4f8-recovery-target8-frozen-v1.json`，自哈希为
  `381d1d3004f8f75e27e247eb900ee8f220c6f36667ba5217ef9ced36e8f17fbb`；
- 训练配置仍只读取 recovery-v2 self-play replay 和冻结 challenge replay，明确不引用
  `recovery-evaluation-v1`，64 局 screen 轨迹不会进入训练；
- recovery 训练源码 SHA-256 仍为
  `ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`；
- global GPU ledger 保持 `46.76848465306063 / 72 GPU-hours`，按保守的全局差值计算，recovery
  阶段剩余 `11.199205818308727 GPU-hours`；Phase 4 产物约 `5.20 GiB`，空闲磁盘约
  `71.9 GiB`。

Native 3/3、完整 Python 415/415 通过。用户运行：

```bash
./scripts/run-phase4f-recovery.sh \
  --target-generations 8 \
  --device-index 0 \
  --gpu-hours-used 0 \
  --resume
```

target 是绝对 committed-generation 数量，因此只新增 generation 4-7。`Ctrl+C` 后重复同一命令
即可从完整 replay game 和最新 committed checkpoint 继续。完成后先审查 generation 4-7 和资源，
不自动启动下一次 screen 或 400 局 confirmation。

## 15. Phase 4F34 F33 失败复盘与 Stage A2 冻结

F34 已完成 CPU-only 复盘，未访问 CUDA，global GPU ledger 保持
`76.14802950652549 / 120 GPU-hours`。F33 的 400 局确认赛为候选 181 胜、219 负，胜率
`45.25%`，90% CI 为 `[41.25%, 49.25%]`，因此 F31 候选不得 promotion。

逐层诊断显示主要弱点是早期策略和失败局中的过量落墙：深度 8/12 的合计胜率为 `37.5%`，
深度 16/20/24 为 `50.4167%`；失败局候选落墙率为 `23.7609%`，胜局为 `11.0067%`。
候选作为 player 0/1 的胜率分别为 `46.5%` 和 `44.0%`，不足以支持视角编码故障的判断。
分析自哈希为
`b262106c419e1f5607f11070b2caa21b5f36dfdf13ac39cf65fa163efa3b38ec`。

回退基线固定为最后一个通过 classic-hard screen 的 F21 Stage A generation 1：candidate model SHA-256
为 `1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01`，checkpoint
SHA-256 为 `dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b`。
现有 classic guidance 训练集和 holdout 分别包含 `26,803` 和 `2,356` 个唯一水平镜像局面轨道；
它们与 F33 的 `19,549` 个轨道均为零重合，因此可以继续使用，且 F33 精确轨迹仍禁止进入训练。
隔离审计自哈希为
`1586a7f693dac6f86eaf810a3c3f48af2c00f86f885e0ad1740f7c34080494ea`。

冻结的 Stage A2 训练协议规定：从 F21 基线清空 optimizer 和 replay 重新初始化；classic-hard 仅通过
监督 policy/value replay 提供指导，不进入终局 reward；每个 batch 的 challenge 占比为 `50%`，其中
至少 `60%` 来自深度 8/12，并平衡行动方和动作类型；网络仍为 `192x16`，每代 128 局、每步
512 simulations、96 optimizer steps、batch 512，最多以两代为一个 micro-block。Stage B 先使用一代
`15%` challenge bridge，再切换为纯 self-play。训练协议自哈希为
`6d008e952e6f2a07cb1260cfbce67a09cabc9b183c3c2115e6af7a30bae24f55`。

新的评测协议要求使用与所有历史 screen/confirmation 不相交的新开局，先进行 64 局 screen，再按原阈值
进行 400 局 confirmation，同时保留对 F21 的 lineage gate。评测协议自哈希为
`74cc7c9b2f3f968296838f211897eb82a08a9a4102ed000805b11f0f69cbe5e6`。F34 总报告位于
`training/runs/phase4/phase4f/f34-failure-review-v1/preparation-report.json`，自哈希为
`c00ecb38994a41467fb6d821888d039cab904c95548d911511e9d4b0765d22b0`。

F34 不授权训练、pilot 或 promotion。下一阶段为 Phase 4F35：实现 A2 的分层 guidance sampler、
配置和 CPU 测试，并准备单代 CUDA pilot；开始前必须再次获得用户确认。

## 16. Phase 4F35 Stage A2 分层采样与 Pilot 准备

F35 已完成 CPU-only 实现、审计和冻结，未访问 CUDA，global GPU ledger 保持
`76.14802950652549 / 120 GPU-hours`。训练仍从 F21 Stage A generation 1 回退基线开始，清空
optimizer 与 self-play replay；classic-hard 只作为占 batch `50%` 的监督 replay，不改变终局 reward。

新增的确定性分层采样器将 `27,025` 个 guidance positions 划分为
`early/other x side 0/1 x move/wall` 八个互斥池。实际库存为：early move `10,688`、early wall
`2,638`、other move `10,942`、other wall `2,757`。每个 challenge batch 保证至少 `60%` 来自
深度 8/12，并精确平衡 side-to-move 与 move/wall；同一 batch 内不重复采样。索引只在 F35
准备时从原始 shards 构建一次，CUDA 训练时直接加载已验签 sidecar，不再执行昂贵的原始数据解码。

冻结产物如下：

- preparation report：
  `training/runs/phase4/phase4f/f35-stage-a2-preparation-v1/preparation-report.json`，自哈希
  `21988e68132d6ca303f05b7d77ab2b3fa5b376bb5bbfdfcf5fff8c2922a6795c`；
- sampling index：
  `training/runs/phase4/phase4f/f35-stage-a2-preparation-v1/guidance-strata-index.json`，文件哈希
  `f21368dd714cfdfd2861f566c78a3df5d286620c3296048347c6ceaf0d984c3d`；
- pilot config：`training/configs/phase4f35-stage-a2-pilot-v1.json`，文件哈希
  `ac18131da4ca0591aa7b8e06d15e0520f2483aeb266e7b8b263543af59fa14a7`；
- frozen execution contract：
  `training/configs/phase4f35-stage-a2-pilot-execution-frozen-v1.json`，自哈希
  `93722583d02f4545f22f75b02be55fbe43d41770ad3f46544fa5588100ddeaca`。

Pilot 固定为单代：`192x16` 网络、128 局、每步 512 simulations、96 optimizer steps、batch 512、
learning rate `5e-5` 至 `5e-6`。其阶段预算为最多 `2 GPU-hours`。准备时 Phase 4 产物约
`10.21 GiB / 12 GiB`，预留约 `1.22 GiB` 后仍满足上限；空闲磁盘约 `57.3 GiB`，高于
`20 GiB` 停止线。

F35 的 `cudaPilotAuthorized` 为 `false`，不会隐式启动 GPU 工作。下一阶段 Phase 4F36 将先执行
current-source CUDA AMP memory smoke 与 replay input gate，再运行隔离的单代 A2 pilot 并审查结果；
必须获得单独确认后才能执行：

```bash
./scripts/run-phase4f36-stage-a2-pilot.sh
```

## 17. Phase 4F36 Pilot 启动故障与 v2 恢复

F36 首次启动的 current-source CUDA AMP memory smoke 已通过，报告 SHA-256 为
`5e6390de3f936eeca2fda86541fda607c11aded15c93797f8b7fc894e8c8c061`，消耗约
`0.000769726 GPU-hours`。随后旧启动器调用 replay input gate 时未传递 F35 config 和 execution
contract，CLI 回退到 8 GiB 默认产物上限并在写入前 fail closed；训练、self-play、checkpoint 和
model artifact 均未开始。

恢复层保留原 F35 报告与启动器不变，并使用 Git 初始提交
`3d7fd8736b5075012e1e40cfa302024471558522` 对 F35 历史源码逐文件验签。F36 v2 保存完整 GPU
ledger 快照，只接受追加记录，并把原 2 GPU-hours 包络扣除已完成 smoke 后冻结为
`1.99923 GPU-hours`。新的 replay gate 显式传递 12 GiB config/contract，训练使用独立 run directory，
不会混入未提交的 v1 状态。

恢复报告位于
`training/runs/phase4/phase4f/f36-stage-a2-recovery-v1/preparation-report.json`，自哈希为
`49a7c9027756c80e5baec26e0e378c55c68cfe31e40ddda6d51a39e14de21bcd`。恢复后的唯一启动命令为：

```bash
./scripts/run-phase4f36-stage-a2-pilot-v2.sh \
  --device-index 0 \
  --gpu-hours-used 0
```

该命令可重复执行；存在 v2 run manifest 时自动追加 `--resume`。完成单代 pilot 后只进入结果审查，
不自动授权后续 Stage A2 训练或 promotion。
