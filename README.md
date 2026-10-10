# Quoridor

## AlphaZero 工程入口（M0～M2）

新一轮 AI 改造的入口是 [AGENTS.md](AGENTS.md)、[progress.md](progress.md)
和 [tasks.yaml](tasks.yaml)，行为规范见 [docs/spec.md](docs/spec.md)。
M0～M2 已完成，包含独立规则引擎、episode outcome、版本化编码器与纯张量网络。
编码、变换和模型权重格式见 [docs/model-v1.md](docs/model-v1.md)。下文浏览器和旧 Phase
训练流程属于历史实现，尤其重复判负、步数上限判负与新规范不同。
新代码显式导入 `quoridor.engine`、`quoridor.episode`；`quoridor` 根导出仍是旧接口。

Linux x86_64 / Python 3.10（含 pip）、Git、GNU Make 使用独立的 CPU 环境，
首次安装需要网络与磁盘空间；`make install PYTHON=python3` 可指定 Python 命令：

```bash
make install
make check
make check-m1
make check-m2
make help
```

依赖由 `training/uv.lock` 固定，配置在 `configs/*.yaml`，新运行目录为 `runs/`。
`make smoke`、`make arena`、`make experiment` 已预留接口，目前明确报
`NOT IMPLEMENTED` 并返回非零状态；不会启动旧训练脚本。后续分别由 M4、M5、M6
接通。旧模型与数据保存在 Git 忽略的 `training/runs/` 中，新克隆不包含这些文件。
新模型架构由 `configs/model.yaml` 配置；输入是明确标注的部分历史观测，搜索必须
保留完整 episode history。模型权重可以校验保存/加载，完整训练恢复仍待 M4。

## 现有浏览器游戏

一个零依赖的本地 Quoridor（步步为营）游戏，支持：

- 本地双人对战
- 本地人机对战
- AI 对 AI 自动连续对弈
- AI 搜索在 Web Worker 中运行，可中止且不阻塞页面主线程
- 人类先手或 AI 先手
- 开局可选择红方在下或蓝方在下的棋盘视角
- 简单、普通、困难三档 AI
- 完整走棋、跳跃、斜跳、放墙和封路判定
- 三次重复或第 256 个半回合由触发行动方判负

## 运行

在项目根目录启动服务：

```bash
npm run dev
```

默认访问地址为 `http://127.0.0.1:4173`。

如果默认端口已被占用，可以指定其他端口，例如：

```bash
PORT=4174 npm run dev
```

然后访问 `http://127.0.0.1:4174`。

## 停止服务

如果服务正在当前终端前台运行，按 `Ctrl+C` 即可停止。

如果启动服务的终端已经关闭或找不到，可以先按端口查询进程：

```bash
lsof -ti tcp:4174
```

再停止查询到的进程：

```bash
kill <PID>
```

如果服务使用的是默认端口，请将上述 `4174` 替换为 `4173`。

## 测试

```bash
npm test
```

重新生成 AlphaZero 跨语言一致性夹具：

```bash
npm run generate:fixtures
```

运行 Phase 2 Python 参考训练测试：

```bash
npm run test:training
```

构建 Phase 3 C++ 原生规则和 MCTS：

```bash
npm run build:native
```

运行 Phase 3 完整差分、性能和原生训练验收：

```bash
npm run validate:native
```

原生训练、自动续训和停止方法见
[`training/README.md`](training/README.md)。

训练闭环和 CUDA 使用说明见
[`training/README.md`](training/README.md)。

## Phase 4E0 准备

Phase 4E0 只执行 CPU 工程和证据准备，不会启动 CUDA。创建或复核 warm generation 4
的 model-only baseline 和 replay 迁移候选，并构建/测试原生引擎、生成后续训练需要的
current-source CPU replay-input gate：

```bash
./scripts/prepare-phase4e.sh
./scripts/prepare-phase4e.sh --verify-only
```

准备过程可用 `Ctrl+C` 停止；重新运行同一命令会验证并复用已完整提交的产物。Phase 4E0
最初生成的 execution contract 为 `prepared/pending`。2026-07-28 产生的旧 E1 报告和
`training/configs/phase4e-execution-frozen-v1.json` 目前只保留为历史证据，不能再作为
训练授权：首次 Bootstrap 初始化因 driver 中 bootstrap verifier 的作用域错误触发
`NameError`，在 generation、checkpoint、replay 对局和 optimizer step 均为零时停止。
失败 invocation 的 `0.0003736264895026882 GPU-hours` 已经入账；全局总账为
`10.041741419628574 / 72 GPU-hours`，不会删除或抵扣。

Replay 准备会多次执行 CPU 深度校验，可能持续数分钟；脚本会在进入该阶段前输出状态，这
不是 GPU 训练。

CPU-only Bootstrap 初始化修复和 E1 rebaseline 准备已完成，不授权 CUDA、E1 校准或
Bootstrap 训练。最终源码指纹为
`a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381`。
失败的 Bootstrap lineage 已保留式封存，incident 自哈希为
`a3165c13a606a94c42c6f59775e2244f760ebddcb35cede65aabb7a77c69295d`；
旧 E1 完成证据也已 supersede，incident 自哈希为
`70089388d5e4c2198de68ceea40868402b8f65316fd28bfe2ea9f3aade176d43`。
旧的 `frozen-v1` 五代命令已经暂停，不应重跑。

当前源码的 replay input gate 已通过，报告为
`training/runs/phase4/replay-input-gate/report-a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381-ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3-8e70f78c7d29e21eef999124206bcb8c83850d849b869ff866a7954ece14f6c5.json`，
SHA-256 为 `2643d1fc1e67cec4756f66525daab30a3af1e69902f94fa3108e9da7c8701959`。
原生测试为 3/3，完整 CPU Python 测试为 381/381。

Phase 4E 的全部产物仍受 `8 GiB` 上限约束；可用磁盘低于 `20 GiB` 时准备和训练入口均会
停止。用户已于 2026-07-28 确认重新运行 E1 calibration；该校准及其审核、`frozen-v2`
生成和 Phase 4E stage-ledger rebase 均已完成。Bootstrap 已训练至 generation 14；该候选
通过 health、64 局 screen 和 400 局 confirmation，并已发布为首个 formal champion。

## Phase 4E1 校准

旧 Phase 4E1 已运行过本机 CUDA 校准，但其报告绑定修复前源码，必须由新源码下的
rebaseline 结果 supersede。固定校准合同为
`training/configs/phase4e1-calibration-v1.json`，SHA-256 为
`47fbe72edd9c75c10b97b0b3f1475205bf6d1630234314a5d11d1ceb43899955`。
CPU 修复与保留式 quarantine 已验收完成，本次 rebaseline 校准已经用户明确确认。启动和
中断恢复均使用同一命令：

```bash
./scripts/calibrate-phase4e1.sh \
  --contract training/configs/phase4e1-calibration-v1.json \
  --device-index 0 \
  --gpu-hours-used 0
```

运行期间不要修改 `training/quoridor/`、`native/`、`src/` 或 `scripts/` 下参与 source
fingerprint 的文件。`--gpu-hours-used` 表示共享 ledger 之外的额外用量；当前全部用量都已
入账，因此必须保持为 `0`。

脚本先构建并测试原生引擎、复核 current-source replay input gate，然后依次执行：

- Standard-shaped self-play：`192x16`、128 局、64 actors、每步 512 simulations、
  前 24 手使用训练温度；
- batched formal-arena throughput smoke：固定 `dev[:40]` 开局，双方均为已审计的 warm
  bootstrap baseline，共 80 局、64 actors、每步 800 simulations。它只测正式 arena
  调度与吞吐，不是晋升比赛。

Arena 的完整搜索配置还会以 canonical SHA-256
`89545b401d92a26c2d659ea4736ea6ffc76a56ab7733f8fe1c1f6636c0f6428a`
绑定到双方 selector identity、pair store 和报告；其中固定 PUCT 参数、`temperature=0`、
关闭 root noise 和 seed `20260714`，不能只靠“800 simulations”冒充同一 workload。

可用 `Ctrl+C` 停止，随后重新运行完全相同的命令继续。已提交的 self-play report/replay
或已原子落盘的完整 candidate、完整 opening pair、GPU 计费和聚合性能指标会被复核并复用；
已有组件及 partial arena pair 的 CPU 深验发生在新一轮 GPU 计费开始前。最终报告嵌入
自哈希 GPU ledger snapshot；后续允许追加训练 run，但不能回滚快照中的计费记录。默认最多
使用 6 GPU-hours，并继续遵守全局 72 GPU-hours、8 GiB 产物和 20 GiB 可用磁盘停止线。

上述复用仅适用于源码指纹未变化的中断恢复。若 runner 检测到未完成 run 的源码指纹与当前
源码不同，它会在全局 Phase 4 锁内、CUDA 查询和新 ledger invocation 之前，将旧 run 整体
原子移动到 `training/runs/phase4/phase4e1/quarantine/`。自哈希 incident 会记录旧 manifest、
完整文件 inventory、新旧源码哈希和当时的 ledger snapshot，并由新 run manifest 绑定。
旧 GPU 用量不会删除或抵扣；新源码会从头执行完整 E1 workload。

新一轮结果仍固定写入
`training/runs/phase4/phase4e1/calibration/phase4e1-calibration-47fbe72edd9c75c1/report.json`。
本次 rebaseline 已完成并通过官方 verify：报告 SHA-256 为
`c162921ba56ed0d4d6b86bc430a74963bddb54fe2029ec02478d34ac4c587039`，`decisionEligible=true`，
GPU 总账为 `11.115946631046116`，本次 Phase 4E1 累计为 `2.656247726123418 GPU-hours`。
Self-play 为 128 局/64 actors/512 simulations，arena smoke 为 80 局/64 actors/800
simulations；没有训练、checkpoint 或正式晋升。该报告已审核，但仍不能直接授权训练。
该脚本不会训练模型、写训练 checkpoint、执行正式晋升或自动生成 frozen execution
contract。旧报告曾通过完整审核，旧 Phase 4E execution contract 的文件 SHA-256 为
`0ae33064ebd4ded3d9f9f0b96e9aaa89a736dcf3b5b05bcaa44d418afc9743ad`，阶段上限为
`59.96104 GPU-hours`、最多 64 generations；这些数值现为历史 provenance。新源码的
frozen-v2 SHA-256 为 `453c60235859f417d5f7b295d0d764ac5778a7b9927bdfcedcb4dade0b304264`，
stage ledger 已更新为 `a05cf7b338b25c281730f24946cecd7c8f0e8d189d04c2b294c632d395c6ec46`；
GPU 总账和既有失败计费始终单调保留。

Bootstrap generation 0–14 已提交，最终 step 为 `960`。Generation 14 已作为
`g00000014-s00000000-67b580c9087f` 正式发布，模型 SHA-256 为
`67b580c9087f9f4c73f4a3ca3887b05bfdadd57baa20df2406bea542bed45696`。
Phase 4F1 复核时 GPU 总账为约 `43.150 / 72 GPU-hours`；`latest` 已设置，`best` 和
`product` 尚未设置。

新 checkpoint 会同时提交完整恢复文件和仅含 tensor 的安全候选模型；正式 arena 与冠军发布
只绑定后者。训练期 screen/confirmation 只使用冻结的 `dev` 开局集，`final` 集保留到最终
经典困难 AI 对比。

## Phase 4F 最终棋力验证

Phase 4F0 CPU-only 准备已经完成。冻结合同为
`training/configs/phase4f-final-frozen-v1.json`，文件 SHA-256 为
`bfd4abc327518b6215b27c16bd673ebb5247fe950fcc99d8d0a46fb73c82f6b2`。
固定赛道使用 587 个经典搜索节点；墙钟赛道使用 800 ms 搜索预算并执行 850 ms 硬上限。
准备阶段没有访问 CUDA，也没有运行 locked-final 对局或授权产品发布。

只读重验：

```bash
./scripts/prepare-phase4f.sh --verify-only
```

固定赛道已经完成，但候选仅取得 71 胜 329 负，未通过产品门禁。因此 wall-clock 赛道停止，
不得继续执行下列历史评测命令：

```bash
./scripts/run-phase4f-final.sh --track fixed
./scripts/run-phase4f-final.sh --track wall-clock
```

CPU-only 失败分析可幂等重验：

```bash
./scripts/analyze-phase4f-failure.sh
```

完整诊断和恢复边界见
[`docs/PHASE4F_FAILURE_RECOVERY.md`](docs/PHASE4F_FAILURE_RECOVERY.md)。旧 `final` 已用于诊断，
禁止将其轨迹用于恢复训练；下一次产品门禁必须使用新的未见 final suite。

Phase 4F2 CPU-only 恢复工程已经完成。它增加了严格绑定 champion manifest 的 model-only
初始化、可配置 challenge replay 混合、玩家方向/value 校准健康指标，并冻结了与旧
`dev`、旧 `final` 在内容及水平镜像轨道上均隔离的 `recovery-dev-v1` 和
`recovery-final-v1`。准备报告位于
`training/runs/phase4/phase4f/recovery-preparation/report.json`；当前仍明确
`trainingAuthorized=false`。

幂等重验 Phase 4F2：

```bash
./scripts/prepare-phase4f-recovery.sh
```

下一阶段是 CPU-only challenge 数据生成。按照阶段确认协议，未确认前不要运行；确认后使用：

```bash
./scripts/generate-phase4f-challenge-data.sh
```

该脚本支持 `Ctrl+C` 中断后重跑续生成；只读深验使用 `--verify-only`。

AI 的设计与实现说明见 [`docs/AI_STRATEGY.md`](docs/AI_STRATEGY.md)。

AlphaZero 高强度 AI 的技术方案与分阶段执行计划见
[`docs/ALPHAZERO_PLAN.md`](docs/ALPHAZERO_PLAN.md)。
