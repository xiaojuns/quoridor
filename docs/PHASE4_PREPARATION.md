# Quoridor AlphaZero Phase 4 启动准备

> 日期：2026-07-14
>
> 状态：4A-4C 与 full matrix 已完成；用户已于 2026-07-22 确认开始 Phase 4D，
> replay-cache gate-v2、current-source memory smoke 和 teacher-pretraining-v4 已完成并
> 通过复核；Phase 4E 已训练至 generation 14 并发布首个 formal champion；Phase 4F
> fixed 门禁失败后，Phase 4F1 诊断与 Phase 4F2 CPU-only 恢复工程均已完成；当前等待
> 单独确认 Phase 4F3 CPU-only challenge 数据生成，GPU 恢复训练尚未授权
>
> 边界：本文保留启动时冻结的信息、风险和执行顺序。确认后可以实施 Phase 4 代码；
> GPU 校准和训练仍只由用户运行脚本，开发会话不持续监控长任务。

## 1. 当前基线

Phase 3 已通过完整验收：

- C++ 两组测试和 Python/集成测试 47 项通过；
- 48 个固定 fixtures、10,000 个状态及 10,000 个后继状态跨引擎一致；
- 100 盘长局无非法动作或状态漂移；
- 合法动作生成、已知合法动作推进和组合规则 workload 分别加速约
  `144.35x`、`28.45x` 和 `117.55x`；
- CUDA 原生 MCTS 双 actor 训练、checkpoint 恢复、replay 和 generation seed 前进通过。

启动准备时只有 smoke checkpoint；当前已有 generation 14 formal champion。现有
`training/configs/reference.json` 是 Phase 3 的吞吐参考配置，不是 Phase 4 的正式训练配置。
它使用 `192 x 16`、约 1,202 万参数、每手 256 simulations、每代 20 局、8 actors、
20 万 replay 和 40 局 arena。

已知本机资源：

```text
GPU: NVIDIA Quadro RTX 6000, 24 GiB（由 Phase 3 CUDA 报告再次确认可用）
CPU: Intel Xeon W-2245, 8 cores / 16 threads
RAM: 62.5 GiB，准备审计时约 41.6 GiB available
项目所在文件系统可用: 55,185,317,888 bytes（约 51.4 GiB，分区使用率 95%）
训练产物软上限: 8 GiB
```

当前受限开发进程不能直接访问 NVIDIA 驱动，但用户本机运行的 Phase 3 原始报告记录两轮
`device=cuda`。Phase 4 训练入口仍必须在每次启动时重新检查 CUDA、可用显存和磁盘。

## 2. 不应直接长训的原因

### 2.1 Arena 结果目前没有统计意义

当前 arena 从同一个标准初始局面开始，关闭温度和根噪声。确定性候选和 champion 的 40 局
实际只是“候选先手”和“候选后手”两条轨迹各重复 20 次，不能产生有效置信区间。当前代码
也不能直接调用浏览器的 JavaScript 困难 Negamax，没有 Elo、配对置信区间、墙钟预算或
冻结开局集。

### 2.2 网络观察未充分表达重复历史

规则中的三次重复依赖完整 `position_counts`，但当前 14 平面只编码当前局面的重复次数。
因此相同 tensor 可能对应不同的重复风险。MCTS 仍持有精确历史并能正确执行规则，但网络的
policy/value 会看到别名观察。正式数据必须改为保存可无损重建历史的原始局面/对局记录，
并在首轮正式训练前冻结历史特征合同。建议至少增加最近局面历史和“走到该目标将成为第二次/
第三次出现”的动作风险平面；该近似不宣称包含完整 Markov 状态，精确规则仍由 MCTS 裁决。

### 2.3 原生训练仍有端到端吞吐损耗

- 每手 `run_batch` 都新建并销毁 MCTS roots，没有把选中 child 提升为下一手 root；
- 固定 actor 组内的对局结束后不补位，网络 batch 会逐步缩小；
- 叶节点仍逐个转换为 Python State，Python 逐状态构造特征并把 209 个 float 转回 C++；
- 当前只有跨 root batching，没有单树多叶 virtual-loss batching。

这些问题不影响 Phase 3 正确性，但会浪费 Phase 4 的 GPU 小时。

### 2.4 Replay 和 checkpoint 不能按现状扩容

当前一个样本至少包含约 5,372 bytes 的 float32 state/policy 数据，还带 Python tensor
对象开销。20 万样本单份 raw 数据已经约 1 GiB；每个 checkpoint 又完整复制 replay，
保留 5 份会逼近 8 GiB 上限，原子替换时还需要额外临时空间。

Phase 4 应保存按对局分片的紧凑原始状态、动作序列、visit counts、root value、generation、
game/ply 和模型版本；镜像在采样时生成。训练恢复 checkpoint、replay shard 和 model-only
champion 应彼此解耦。

### 2.5 缺少发布级保护和追踪

正式训练前还需补齐：

- evaluator、loss、gradient、参数和 replay 的非有限值 fail-fast；
- generation 级不可覆盖 JSON/JSONL 报告、配置和源码 manifest hash；
- 实际非法动作/异常计数，而不是摘要中的常量字段；
- 独立且不可被训练 checkpoint 清理的历史 champion 池；
- 进程锁、checkpoint 校验和最近有效 checkpoint 回退；
- games/hour、simulations/s、batch、GPU/显存、终局分布、policy entropy、value 校准、
  replay age、p50/p95 延迟等指标。

## 3. 推荐执行顺序

Phase 4 获得确认后按以下内部 gate 执行。任何 gate 未通过都不进入下一段长训。

### 4A. 状态、Replay 和可观测性合同

1. 冻结历史特征和可无损重建的紧凑 replay schema；
2. replay 分片、恢复 checkpoint 和 model-only champion 解耦；
3. 增加数值 fail-fast、运行锁、原子 generation 报告和源码/config hash；
4. 增加 AMP、可恢复学习率调度和新数据/更新步数比例控制；
5. 为 schema round-trip、中断恢复、磁盘上限和数值异常增加测试。

### 4B. 可信竞技场和发布 gate

1. 新建通用 `MoveSelector` arena；神经网络使用 Native MCTS selector；
2. 用常驻 Node JSONL 子进程直接复用 `src/ai.js` 的困难 AI，不复制算法；
3. bridge 传递完整重复历史，并逐手校验 action legality、position hash、winner 和终局原因；
4. 建立互不重叠的 dev/tuning opening suite 与 locked-final suite；
5. 每个 opening 成对交换双方，按 opening pair 做固定种子的 cluster bootstrap；
6. 增加 Elo、置信区间、先后手分项、终局原因、nodes/simulations 和延迟报告；
7. 给困难 AI 增加可复现的 node budget 赛道，同时保留产品默认 `850 ms` 墙钟赛道；
8. 建立独立 champion archive，至少保留最近 8 个 accepted champion、4 个 milestone、
   当前最高 Elo 和产品模型。

### 4C. 原生吞吐和校准

1. 增加按游戏持久化的 MCTS root，并在落子后提升选中 child；
2. actor 结束一局后立即补位，保持稳定 batch；
3. 批量/原生特征编码和低复制 tensor/buffer binding；
4. 视实测结果决定是否加入单树多叶 virtual loss；
5. 对 actors=`8/16/32/64`、simulations=`128/256` 做完整游戏校准；
6. 比较 `192 x 16` 与 `128 x 10`。如果大模型吞吐不低于小模型的 60%，保留
   `192 x 16`；否则用等 GPU 小时的 5-generation pilot 决定。

在得到实际 games/hour、simulations/s 和 GPU 利用率前，不承诺总训练时长。

### 4D. Warm start 和冷启动对照

1. 困难 AI 生成至少 50,000 个去重原始局面，按先后手、胜方、墙数和开局深度分桶；
2. 教师 policy 不得永久裁掉合法墙动作，所有合法动作保留小的 probability floor；
3. 10% 对局作为隔离 holdout，禁止泄漏到训练；
4. warm 与 cold 各运行 5 代等算力 pilot；
5. warm start 没有可信优势时立即弃用教师数据，避免弱教师限制上限；
6. warm start 结束后丢弃教师 replay、重置 optimizer，再进入完整 MCTS 自我对弈。

### 4E. 分级自我对弈

以下只是校准前的推荐起点：

| Lane | games/generation | simulations/move | replay | batch | train steps/generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bootstrap | 64 | 256 | 100k | 512 | 64 |
| Standard | 128 | 512 | 300k | 512 | 96 |
| Strength | 256 | 800 | 500k | 512 | 192 |

共同设置：完整 209 动作、动态 Dirichlet alpha、epsilon `0.25`、前 `20-24` 手温度采样、
无投降。actors、batch 和模型大小最终由 4C 校准决定。

扩容条件：

- `256 -> 512`：紧凑 replay 至少 100k，且已经出现可信 champion 晋升；
- `512 -> 800`：按等 GPU 小时做 A/B，以 Elo/GPU-hour 决定；
- replay 不得继续包含 warm-start 教师样本；
- 连续 5 个确认周期或至少新增 10,000 盘自我对弈仍没有可分辨的 `+15 Elo`
  提升时，停止盲目续训并转向调参、网络扩容或 reanalysis。

### 4F. 后期增强

先获得稳定 baseline champion，再依次评估：

1. 对高 policy loss、高 value error、临近重复和低墙局面的 10%-20% 样本做 1600
   simulations reanalysis；
2. FPU、Gumbel/sequential halving、动态 simulation budget；
3. 困难局面采样和历史 champion 混合对弈；
4. 无墙残局辅助 oracle。

无墙残局的真实值依赖 `move_number` 和完整重复历史。仅按棋子位置构建的 board-only
tablebase 只能作为辅助标签，不能向 MCTS 声称是现行规则下的 exact value。

## 4. 建议验收门槛

### 4.1 每代健康门槛

- 非法动作、规则漂移、NaN/Inf、draw/null winner、bridge crash 均为 0；
- generation 5 以后，滚动 500 局中 repetition 低于 5%，move-limit 低于 1%；
- 训练产物不超过 8 GiB；文件系统可用空间低于 20 GiB 时停止保存和训练；
- checkpoint 恢复后的 generation、seed、optimizer、scheduler 和 replay cursor 严格前进。

### 4.2 Champion 晋升

- 快速筛选：32 个 dev openings 成对交换，共 64 局、256 simulations；只用于淘汰，
  不用于声明棋力；
- 正式晋升：200 个冻结 openings 成对交换，共 400 局、800 simulations；候选裸胜率
  至少 55%，且按 opening pair bootstrap 的单侧 95% 下界高于 50%；
- 对最近 5 个、最高 Elo 和产品 champion 做非劣回归；裸胜率至少 50%，单侧 95%
  non-inferiority 下界高于 45%。

### 4.3 最终超过困难 AI

固定 workload 和产品墙钟两条赛道各使用 200 个 locked-final openings、成对交换共 400 局：

- 裸胜率至少 60%；
- paired-bootstrap 双侧 95% 下界至少 55%；
- 非法动作、draw、bridge mismatch 和超时崩溃为 0；
- 墙钟赛道双方每手最多 850 ms，并报告 p50/p95、实际 nodes/simulations 和 timeout；
- 标准初始局面的两盘单独报告，不混入 400 局统计样本。

### 4.4 战术和回归

- 至少 200 个独立基础局面；镜像/颜色变换只验证等变性，不重复计数；
- 一步必胜、必须防守、三次重复自杀规避和第 256 手自杀规避达到 100%；
- 总体 top-1 命中任一最优动作至少 90%，单类别至少 85%；
- 相对当前 champion 不下降超过 2 个百分点；
- locked-final opening 和最终战术集不得用于调参或 early stopping。

## 5. 产物与依赖

建议 Phase 4 新增的主要模块和产物：

```text
training/quoridor/replay_store.py
training/quoridor/run_manifest.py
training/quoridor/openings.py
training/quoridor/statistics.py
training/quoridor/selectors.py
training/quoridor/classic_bridge.py
training/quoridor/phase4_driver.py
training/configs/phase4-*.json
scripts/classic-ai-bridge.mjs
scripts/calibrate-phase4.sh
scripts/train-phase4.sh
scripts/evaluate-phase4.sh
training/fixtures/openings/{dev,final}.json
training/fixtures/tactics/
training/runs/phase4/{replay,checkpoints,champions,reports,logs}/
```

继续复用现有 Python 3.10、PyTorch 2.7.1+cu126、C++17/Ninja 和 Node.js，不计划新增
云服务或必须下载的运行时。由于当前 workspace 没有可用 Git 元数据，运行 manifest 应记录
相关源码文件的稳定 SHA-256，而不能依赖 Git commit id。

## 6. 已确认约束

1. 按本文范围开始 Phase 4，先完成 4A-4C 工程门槛，不直接启动长训；
2. 继续保持“仅本机、Phase 4 产物上限 8 GiB、磁盘低于 20 GiB 自动停止”；
3. 累计 GPU 预算冻结为 `72 GPU-hours`，实际 generation 数在 4C 实测后换算；
4. 校准和长训练由用户运行脚本，脚本输出和 JSON 报告回传复核，不由开发会话持续监控。

## 7. 已冻结的竞技场开局资产

`training/fixtures/openings/dev.json` 与 `final.json` 各包含 200 条合法、完整复放、无重复局面的
分层前缀，两套内容互斥。快速筛选固定使用 `dev` canonical 顺序的前 32 条；正式晋升使用全部
200 条 `dev`；`final` 只用于最终困难 AI 对比，不得用于调参或 early stopping。

```text
dev:   sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4
final: sha256:65f1bb5223e8e7c9da469b69b891443458d631e8058cfb4a14f57f3d1243833d
```

以上为 arena policy 使用的 canonical schema `contentHash`，不是 JSON 文件的逐字节摘要。
运行 `./scripts/generate_phase4_openings.py --check` 可重新生成并逐步复放，然后比较冻结内容。

## 8. Quick 校准结果

本机 quick 运行 `phase4-calibration-20260714-191830-2804640` 已通过接线 gate。报告、manifest、
源码指纹、GPU 账本和 2 个 raw replay shard 均已校验；2 局共 178 half-moves。场景为
`128x10 / 2 actors / 16 simulations`，得到约 `859.71 games/hour`、`340.06 scheduled
simulations/second`、batch p50/p95=`2/2`、GPU utilization p50=`33.5%`，累计使用
`0.002668 GPU-hours`。该 quick workload 只验证链路，不用于冻结正式配置；full matrix 必须在
再次确认后运行。用户已于 2026-07-21 明确确认开始 full matrix，执行仍由用户在本机完成。

## 9. Full 校准结果与正式选型

Full run `phase4-calibration-20260721-171946-972095` 已完成并通过独立复核：16 个场景、960 局、
109,539 half-moves 和 20,869,632 scheduled simulations；所有 replay shard 深度验证通过，
`decisionEligible=true`。本次使用 `4.036185 GPU-hours`，Phase 4 累计使用 `4.038853 hours`，
剩余 `67.961147 hours`。

`192x16` 相对 `128x10` 的匹配 sim/s 比率最低 `74.21%`、平均 `84.03%`，每组均超过冻结的
60% 门槛。因此正式网络为 `192x16`，无需运行小模型 5-generation pilot。Bootstrap 与
warm/cold pilot 使用现有 `phase4-bootstrap.json`：64 games、256 simulations、32 actors；
该 workload 实测 `1430.43 sims/s`、约 `179.28 games/hour`。当 Standard lane 每代至少 128 局
时切换到 64 actors，其匹配 workload 实测 `1750.57 sims/s`。64 actors 不是 64 局 workload
的已校准配置，不能直接套用。

Full matrix 只冻结吞吐配置，不提供棋力证据。`192x16`、batch 512、AMP 反向传播显存
smoke 已通过；Phase 4D 已于 2026-07-22 获得确认，GPU 工作仍只由用户运行脚本。

## 10. Phase 4D 已冻结合同

用户确认 Phase 4D 使用最多 `10 GPU-hours` 的阶段子预算；它从 full calibration 后的累计
`4.038853 GPU-hours` 起算，不改变全局 `72 GPU-hours`、仅本机、`8 GiB` 产物上限和
`20 GiB` 磁盘停止线。所有 CUDA 入口共用全局锁、GPU 账本和自校验阶段账本。

已实现的首批工程门包括：

- 固定 `192x16`、batch 512 的真实 AdamW/AMP forward、loss、backward、scaler、optimizer 和
  scheduler 显存 smoke；不生成 self-play、replay、checkpoint 或模型产物；
- 与 config/source lineage 绑定的 generation `-1` 初始化 checkpoint、损坏 checkpoint 回退、
  raw replay 高水位校验、完整局恢复和最多 5 代的正式 pilot driver；
- 困难经典 AI 的固定 node-budget teacher 生成器：至少 50,000 个水平镜像 orbit 去重局面，
  整局 90/10 train/holdout 隔离，所有合法动作保留 `1e-4` policy floor；
- warm artifact 同时绑定 cold 基础 tensor-state hash、warm tensor-state hash、teacher dataset
  指纹和预训练报告指纹；warm 导入后不导入 teacher optimizer/replay。

首个 GPU gate 的冻结入口为：

```bash
./scripts/run-phase4d-training-memory-smoke.sh
```

只有该报告 `passed=true` 后才生成长期 teacher 数据并进入预训练和 warm/cold pilot。脚本可执行
权限、实际 GradScaler/forward autocast/output dtype、`nvidia-smi` 至少两个有效样本、Torch 峰值
显存、报告大小和所有资源停止线均为硬门。

初始冻结的后续入口依次为；其中 v2 summary 现仅保留为历史 lineage：

```bash
./scripts/generate-phase4-teacher-data.py
./scripts/pretrain-phase4d-teacher.sh
./scripts/run-phase4d-warm-cold-pilot.sh \
  --pretraining-summary training/runs/phase4/teacher-pretraining-v2/summary.json
```

预训练和 pilot 的深度复验都固定读取
`training/runs/phase4/teacher-data`；预训练 CLI 不接受替代 teacher root，避免生成 pilot 无法按同一
路径和 lineage 复验的 warm artifact。预训练默认写入新的
`training/runs/phase4/teacher-pretraining-v2` lineage，并在 GPU ledger 开始前构建不超过
1 GiB 的 CPU 双镜像预编码缓存；旧 `teacher-pretraining` 目录保持不变。

最后一个脚本从已完成预训练的 exact-v1 summary、report 和 model-only artifact 生成不可变的
cold/warm effective config。两条 lane 使用相同 seed `20260714`、`192x16`、每代 64 局、32
actors、每步 256 simulations、batch 512 和每代 64 个 optimizer steps，并按
`cold gN -> warm gN` 交错运行到 generation 4。训练入口使用累计
`--target-generations N`；若中断发生在 checkpoint 已提交而 report 尚未提交的窗口，重跑只做
reconcile，不会额外推进一代。初始化 checkpoint 必须证明 optimizer/replay 为空，之后两个 raw
replay 都会深度扫描并拒绝任何 teacher model/data 引用。

pilot 不接受只做自哈希的浅层 report。开始前会完整验证 teacher 两套 ReplayStore、当前通过的
显存 smoke、固定 `64..512` 共 8 次 holdout evaluation、run/completion manifest、最终
checkpoint 和 warm artifact 的同一 lineage。基础协议绑定仓库内
`training/configs/phase4-bootstrap.json` 的精确文件 SHA；`--base-config` 只能指向完全相同的
副本，不能改 learning rate、PUCT、temperature 或 checkpoint 数量。prepare 还会冻结默认
训练源码指纹和当前加载的 native extension SHA；每次 `next-step` 和 lane validation 都在启动
另一条 lane 前重新核对，包括只有一条 lane 已启动的恢复场景。

pilot 最后使用冻结 dev suite 的前 32 个 opening 做 64 局换边对弈，双方都是 256 simulations
的真实 native MCTS。checkpoint、effective config、source 和训练/arena native backend hash 必须
全部匹配。warm 只有在 raw win rate `>=55%` 且 paired bootstrap 的 one-sided 95% lower
bound `>50%` 时才被选择，否则选择 cold。这是 bootstrap 选择 screen，不是正式 champion 声明；
arena 中断时 partial JSONL 不构成结果，完整 64 局必须重新运行后才能原子提交结果 manifest。

## 11. 当前执行证据（2026-07-26）

gate-v2 current-source memory smoke
`phase4d-training-memory-smoke-20260725-234218-208557` 已通过全部 11 个门槛，绑定完整训练
源码 SHA `bb48c84b08cf641f7b2c5cc8384b4380e841322383c0de763004e5be7930a165`，report core SHA
为 `76a4c5a5095e9ffeeb214eb15f4ba4e6372a914736c6a605b5eaab540208651b`。

current-source 教师预训练位于
`training/runs/phase4/teacher-pretraining-v4`，其 `summary.json` 已完成全量 bundle 复验：
数据集 SHA 为 `9eb2f39029758589d753c49d2c765001b4f7fe42f7468af01cfa7233a634fe3c`，选中模型
tensor-state SHA 为 `03c97f88b8cd2d21bc9eb4b30a371f0d0ec9c1a0bbb938bcd5695390390a3afe`，warm artifact
文件 SHA 为 `e1e7dbba95e18ea22b0e1c3a16ca22c4eae89c493651f00ca6ef81888f1c6f8d`，report 文件 SHA
为 `fd40ea591ee9a47d858ba85ca36f09286a1f9c59ce6b1e399249ff5208bf8b8e`。

最终 GPU 账本为 `5.627920734/72 hours`；Phase 4D 使用 `1.589067578/10 hours`，剩余
`8.410932422 hours`。Phase 4 产物约 `1.630 GiB`，空闲磁盘约 `77.59 GiB`。

旧 `training/runs/phase4/phase4d-pilot` 已包含旧源码的 partial contract 和 cold lane，不能
用于 current-source 恢复。用户已明确确认下一阶段；必须用 v4 summary 和尚不存在的新 root
执行一条完整命令：

```bash
./scripts/run-phase4d-warm-cold-pilot.sh --pretraining-summary training/runs/phase4/teacher-pretraining-v4/summary.json --output-root training/runs/phase4/phase4d-pilot-v2
```

中断后只用同一条完整命令恢复该 root；不得省略 `--output-root` 回落到旧默认目录。

pilot-v2 随后完成 cold/warm 各 5 代、每条 lane 320 局 replay 和最终 64 局 paired screen。
warm 以 62-2 获胜，raw win rate 为 `0.96875`，opening-pair bootstrap 单侧 95% 下界为
`0.921875`，通过 `>=0.55` 和 `>0.50` 两项冻结门槛。两条 lane 的完整 replay、最终
checkpoint、arena 1,907 plies、全部 hash 和 provenance 均通过独立深验，因此后续自我对弈
选择 warm g4。

本次结果仅是 `screenOnly` 的 bootstrap 选择，不是正式 champion 晋升。最终全局 GPU 使用量
为 `8.459698905/72 hours`；Phase 4D 累计使用 `4.420845748/10 hours`，剩余
`5.579154252 hours`。Phase 4 产物约 `2.175 GiB`，空闲磁盘约 `77.01 GiB`。Phase 4D
至此完成；开始 4E 分级自我对弈和正式 champion 路径前必须重新确认。
