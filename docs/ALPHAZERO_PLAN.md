# Quoridor AlphaZero 技术方案与执行计划

> 文档状态：Active implementation plan
>
> 创建日期：2026-07-12
>
> 当前执行状态：Phase 0-3 已完成；Phase 4D warm bootstrap、Phase 4E generation-14 champion
> 晋升均已完成。Phase 4F fixed 产品门禁以 71-329 失败后，已完成失败分析、隔离 challenge
> replay、Phase 4F5 CUDA rebaseline 和 Phase 4F6 CPU-only formal recovery launcher/driver
> preparation。正式恢复训练合同冻结为 14 GPU-hours、最多 16 generations；当前尚未创建
> `recovery-v2` lineage，启动 CUDA 训练仍需单独确认。

## 1. 目标

为当前 Quoridor 项目构建一个以最高棋力为主要目标的 AI Player。最终方案采用：

1. 完整合法动作空间；
2. 策略价值神经网络；
3. 神经网络引导的 Monte Carlo Tree Search；
4. 大规模自我对弈训练；
5. 高性能原生规则引擎；
6. 无墙残局精确求解；
7. 浏览器 WebGPU/WASM 推理和现有 Negamax 降级路径。

这里的“最高棋力”是工程目标，不预先承诺某个绝对水平。最终强度必须通过固定预算、交换先后手的竞技场对局和 Elo/置信区间来证明。

## 2. 非目标

- 不在浏览器主线程中训练模型。
- 不使用启发式奖励替代最终胜负目标。
- 不让训练引擎复制一套未经验证的规则后直接生成数据。
- 不在 AlphaZero MCTS 中永久裁掉低先验墙动作。
- 不在缺少模型或硬件支持时移除现有可运行的 Negamax AI。

## 3. AlphaZero 核心算法

神经网络接收状态 `s`，同时输出策略先验和局面价值：

```text
(p, v) = f_theta(s)
```

- `p[a]`：动作 `a` 的先验概率；
- `v`：从当前行动方视角预测的最终结果，范围为 `[-1, 1]`。

MCTS 的每条边保存：

```text
P(s,a)  神经网络先验
N(s,a)  访问次数
W(s,a)  累计价值
Q(s,a)  平均价值，W / N
```

选择动作时最大化：

```text
Q(s,a) + U(s,a)
```

推荐以 AlphaZero 补充材料中的动态探索系数为起点：

```text
U(s,a) = C(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

C(s) = log((N(s) + c_base + 1) / c_base) + c_init

c_base = 19652
c_init = 1.25
```

每次 simulation：

1. 从根节点沿最大 `Q + U` 选择到叶节点；
2. 终局叶节点使用规则给出的精确结果；
3. 非终局新叶节点调用一次神经网络；
4. 使用完整合法动作掩码展开节点；
5. 沿访问路径回传价值；
6. 每跨过一个玩家回合将价值取反。

根节点访问次数形成改进策略：

```text
pi(a|s) = N(s,a)^(1/tau) / sum_b N(s,b)^(1/tau)
```

每个自我对弈样本为：

```text
(canonicalState, rootVisitPolicy, finalOutcome)
```

训练损失：

```text
loss = valueMSE(z, v)
     + policyCrossEntropy(pi, p)
     + weightDecay
     + optionalAuxiliaryLosses
```

训练自我对弈时在根节点加入 Dirichlet noise：

```text
P_root = 0.75 * P + 0.25 * Dirichlet(alpha)
```

Quoridor 初始建议：

```text
alpha = clamp(10 / legalActionCount, 0.03, 0.30)
```

评估和正式对局关闭噪声，并使用最大访问次数动作。

## 4. Quoridor 固定动作空间

策略空间固定为 209 维。

| 范围 | 动作 | 编码 |
| --- | --- | --- |
| `0..80` | 棋子移动目标格 | `row * 9 + col` |
| `81..144` | 横墙 | `81 + row * 8 + col` |
| `145..208` | 竖墙 | `145 + row * 8 + col` |

解码结果必须直接兼容现有动作对象：

```js
{ type: 'move', row, col }
{ type: 'wall', orientation: 'h' | 'v', row, col }
```

非法动作由规则引擎生成的 209 位 mask 排除：

```text
maskedLogits[illegal] = -Infinity
policy = softmax(maskedLogits)
```

### 4.1 强制约束

- 所有合法棋子动作必须保留；
- 所有合法横墙和竖墙必须保留；
- 跳跃和斜跳通过目标格动作自然表达；
- 墙合法性必须包含“不封死任一玩家”的路径检查；
- 当前 Negamax 的候选墙裁剪不得进入 AlphaZero 的训练 target 或完整搜索。

## 5. 状态规范化

神经网络始终从当前行动方视角观察局面：

- 当前行动方被视为 `self`；
- `self` 始终显示在棋盘下方并向上抵达目标边；
- 对手被视为 `opponent`。

当原状态的 `currentPlayer === 1` 时：

```text
交换双方棋子、墙数和历史通道
棋子坐标：(row, col) -> (8 - row, 8 - col)
墙锚点：  (row, col) -> (7 - row, 7 - col)
墙方向：  h/v 保持不变
```

模型选出的动作需要通过逆变换还原为现有规则引擎动作。

UI 的 `bottomPlayer` 只是显示视角，不得进入模型状态，也不得影响训练规范化。

## 6. 对称增强

规范化后允许使用左右镜像：

```text
棋子目标 col -> 8 - col
墙锚点 col   -> 7 - col
```

不得直接使用 90 度旋转。Quoridor 的目标边具有方向性，90 度旋转不是规则对称。

180 度旋转加玩家交换已经包含在当前行动方规范化中，不需要重复产生等价样本。

## 7. 网络输入

建议第一版模型使用约 18 个 `9x9` 平面：

1. 当前玩家棋子；
2. 对手棋子；
3. 横墙锚点，`8x8` 填充到 `9x9`；
4. 竖墙锚点，`8x8` 填充到 `9x9`；
5. 当前玩家剩余墙数常量平面；
6. 对手剩余墙数常量平面；
7. 合法棋子目标格；
8. 当前玩家 BFS 距离图；
9. 对手 BFS 距离图；
10. 当前玩家最短路径并集；
11. 对手最短路径并集；
12. 重复局面次数或回合上限信息；
13-18. 最近若干时刻双方棋子位置。

BFS 和最短路径平面属于领域特征。它们不改变最终胜负目标，但能提高非局部墙作用的学习效率。

所有派生平面必须在 JavaScript、Python、原生训练引擎和浏览器推理后端之间使用同一份 golden fixtures 验证。

## 8. 网络结构

### 8.1 高强度教师模型

```text
输入：9x9xC
主干：16-20 个 residual blocks
通道：192-256
卷积：3x3
增强：Squeeze-Excitation 或少量全局注意力
```

### 8.2 策略头

策略头拆成三个空间输出：

```text
9x9  棋子目标 logits
8x8  横墙 logits
8x8  竖墙 logits
```

可增加动作类型 gate：

```text
P(move), P(horizontalWall), P(verticalWall)
```

最终仍需合并为统一 209 维分布，并应用完整合法 mask。

### 8.3 价值和辅助头

价值头输出当前行动方视角的 `tanh` 标量。

辅助头可以预测：

- 双方最短路径长度；
- 预计剩余回合；
- 下一动作类型；
- 搜索后的局面价值。

辅助目标只用于表示学习，不加入环境奖励。

### 8.4 浏览器学生模型

浏览器部署可以使用教师模型蒸馏后的模型：

```text
10-12 residual blocks
96-128 channels
FP16 WebGPU 或 INT8/WASM SIMD
```

## 9. Quoridor MCTS 强化项

### 9.1 必须实现

- 完整合法动作展开；
- 精确终局判断；
- 价值逐层取反；
- 路径内重复状态检测；
- 落子后复用被选择的子树；
- 批量神经网络叶节点推理；
- 多线程搜索的 virtual loss；
- 固定随机种子下可复现的测试模式。

### 9.2 后续强度增强

- Gumbel AlphaZero 或 sequential halving 根搜索；
- First Play Urgency；
- 动态 simulation budget；
- Monte Carlo Graph Search / transposition sharing；
- 困难局面 reanalysis；
- 低墙数精确求解器；
- 无墙残局 tablebase。

当双方剩余墙数均为 0 时，状态空间足够小，应构建精确残局表库。该表库可以直接给 MCTS 叶节点返回精确结果。

### 9.3 不采用的优化

不得永久只保留“当前增加对手最短路径”的墙。某些强墙具有延迟收益，永久裁剪会限制模型和搜索上限。

如果为了低 simulation budget 使用 progressive widening，必须保证访问量增长后最终能够开放全部合法动作。

## 10. 自我对弈

推荐初始配置：

| 参数 | 建议起点 |
| --- | ---: |
| 自我对弈 simulations/手 | `800` |
| 强力评估 simulations/手 | `2000-10000` |
| 根噪声 epsilon | `0.25` |
| 温度采样阶段 | 前 `16-24` 手 |
| 训练后期动作选择 | 最大访问次数 |
| replay window | `100万-500万` 局面 |
| value 范围 | `[-1,1]` |

每局流程：

```text
state = initialState
examples = []

while state is not terminal:
    canonical = canonicalize(state)
    pi = runMCTS(canonical, trainingNoise=true)
    examples.append(canonical, pi, playerToMove)
    action = sample(pi, temperatureSchedule)
    state = applyAction(state, action)

z = finalOutcome
attach z from each sample's player perspective
push examples into replay buffer
```

### 10.1 预训练

允许使用：

- 当前困难 Negamax 生成的轨迹；
- 人类高质量棋谱；
- 精确残局求解结果；
- 随机可达局面的 BFS 辅助标签。

这些数据只用于 warm start。后续策略必须由完整动作 MCTS 自我对弈继续改进，避免被弱教师限制。

### 10.2 投降

早期训练禁用投降。只有价值头充分校准，并通过低误投降率评测后才启用。

## 11. 重复局面与回合上限

Phase 0 前的产品规则没有重复局面或最大回合定义，但棋子可以循环移动。Phase 0 已为产品和训练统一增加零和的反拖延终局规则。

已确认规则：

```text
行动后同一局面第三次出现：造成该重复的行动方判负
行动后达到 256 个 half-moves：执行第 256 手的行动方判负
```

如果某一步同时抵达目标边和触发限制，正常抵达目标的胜利优先。如果重复和 256 手限制同时触发，终局原因记录为 `repetition`。双方不能同时得到负奖励，训练结果始终保持严格零和的 `+1/-1`。

局面键应包含：

- 当前行动方；
- 双方棋子位置；
- 双方剩余墙数；
- 横墙和竖墙；

不应包含 `moveNumber` 或派生的 `winner` 字段。

该规则同时进入产品和训练规则，避免环境不一致形成可被利用的分布偏差。

## 12. 训练架构

训练采用 actor-learner 架构：

```text
Self-play actors
    -> game records / replay buffer
    -> trainer
    -> new checkpoint
    -> arena / champion evaluation
    -> actors pull accepted/latest checkpoint
```

### 12.1 建议技术栈

- Python 3.10+；
- PyTorch；
- Rust 或 C++ 高性能规则/MCTS 引擎；
- Python 原生绑定；
- 混合精度训练；
- 批量叶节点推理；
- 可恢复 checkpoint 和可追溯配置文件。

训练依赖放在独立目录，不作为浏览器生产依赖：

```text
training/
  pyproject.toml
  quoridor/
    codec.py
    network.py
    mcts.py
    selfplay.py
    replay.py
    arena.py
  native/
  configs/
  export_model.py
```

## 13. 竞技场与发布门槛

每个候选模型至少评估：

1. 对当前困难 Negamax；
2. 对当前 champion；
3. 对多个历史 checkpoint；
4. 对固定战术题集；
5. 在双方颜色和先后手交换后重复；
6. 在固定 simulation 和固定墙钟时间两种预算下重复。

记录：

- 胜/负，任何和棋结果都视为规则实现错误；
- Elo 和置信区间；
- 平均搜索节点；
- p50/p95 落子延迟；
- 非法动作率，必须为 0；
- 重复或回合上限判负率；
- 不同开局策略覆盖率。

AlphaZero 原论文持续更新最新模型。当前产品仍应维护 champion 和发布 gate，避免训练波动进入用户版本。

## 14. 当前项目集成

### 14.1 保留的模块

- `src/game.js`：产品和测试的规则权威；
- `src/ai.js`：经典 AI、基准对手和降级方案；
- 当前 UI、本地双人、人机、AI 对弈和视角；
- 现有规则与 AI 测试。

### 14.2 新的浏览器 Agent 接口

所有 AI 改为异步接口：

```js
Agent.choose(state, options, signal) -> Promise<{
  action,
  value,
  simulations,
  nodes,
  elapsedMs,
  engine,
  depth,
  score,
}>
```

建议目录：

```text
src/agents/agent.js
src/agents/classic-agent.js
src/agents/worker-agent.js
src/agents/alpha-zero-agent.js
src/workers/classic-ai.worker.js
src/workers/alpha-zero.worker.js
src/alphazero/codec.js
src/alphazero/backend.js
src/alphazero/webgpu.js
src/alphazero/wasm-loader.js
models/<version>/manifest.json
```

### 14.3 推理后端顺序

```text
WebGPU
  -> WASM SIMD
  -> 当前 Negamax fallback
```

最大强度比赛模式可以连接本地原生程序或 GPU 服务端。纯浏览器版本使用蒸馏和量化模型。

## 15. 高性能规则引擎

现有 JavaScript 引擎会为墙合法性反复克隆状态并执行 BFS，适合作为规则权威，但不适合大规模 MCTS 自我对弈。

原生引擎需要：

- 紧凑位图或边阻塞表示；
- 原地 `make/unmake`；
- 增量 Zobrist Hash；
- 快速 BFS/位并行可达性；
- 缓存合法 mask；
- 批量自我对弈；
- 与 JavaScript 引擎逐状态差分测试。

未经差分测试通过的原生引擎不得生成训练数据。

## 16. 测试策略

### 16.1 Codec

- 209 个动作全部 encode/decode round-trip；
- 规范化两次还原；
- 镜像两次还原；
- 状态和动作变换满足 `applyAction` 等变性；
- policy/mask 变换一致。

### 16.2 跨引擎一致性

生成数千个带固定种子的可达局面，比较：

- legal mask；
- 所有抽样合法动作的下一状态；
- winner；
- BFS 距离和路径合法性；
- canonical hash。

JavaScript、Python、原生和 WASM 必须读取同一批 fixtures。

### 16.3 MCTS

使用确定性 mock network 验证：

- PUCT 选择；
- 价值符号交替；
- 终局不调用网络；
- 第三次重复路径让触发行动方判负；
- 根访问分布；
- 只能返回合法动作；
- 能抓住立即获胜和必须阻挡。

### 16.4 模型导出

比较 PyTorch、ONNX、WebGPU、WASM 的固定输入输出，并规定 FP32/FP16/INT8 容差。

### 16.5 Worker 生命周期

- 初始化；
- 模型加载失败和 fallback；
- 取消；
- 新对局/撤销后的 stale result；
- AI 对弈连续异步调度；
- 页面主线程无长任务。

## 17. 分阶段执行计划

### Phase 0：规则和 Codec 合同冻结

状态：已完成（2026-07-12）。

目标：在任何训练开始前冻结可跨语言验证的规则、动作和状态合同。

计划工作：

1. 编写 209 动作 codec；
2. 编写 canonical state/action 变换；
3. 编写左右镜像变换；
4. 定义不含 `moveNumber` 的位置 hash；
5. 增加 codec、mask、等变性和随机可达状态测试；
6. 生成第一版 conformance fixtures；
7. 决定重复局面与最大回合规则；
8. 为 codec 和 fixture 标记版本号。

预计新增文件：

```text
src/alphazero/action-codec.js
src/alphazero/state-codec.js
scripts/generate-alpha-zero-fixtures.js
tests/alpha-zero-action-codec.test.js
tests/alpha-zero-state-codec.test.js
tests/alpha-zero-conformance.test.js
tests/fixtures/alpha-zero-conformance-v1.json
```

验收条件：

- 所有 209 动作 round-trip；
- 初始合法 mask 与 `getLegalActions` 完全一致；
- 随机可达状态 mask 完全一致；
- canonical/mirror 变换满足等变性；
- 原有测试全部通过；
- 重复/回合上限判负规则已书面确认。

### Phase 1：异步 Agent 和 Worker 边界

状态：已完成（2026-07-12）。

目标：先把现有 AI 移出 UI 主线程，为后续模型推理建立稳定接口。

计划工作：

1. 定义异步 Agent 接口；
2. 为现有 Negamax 编写 adapter；
3. 在 Web Worker 中执行经典 AI；
4. 加入取消、超时和 stale-result 防护；
5. 保持现有模式和视角，按用户要求取消 AI 对 AI 的固定 3 秒等待；
6. 增加 worker 生命周期测试。

验收条件：

- 现有 AI 行为和合法性不回归；
- 搜索期间 UI 保持响应；
- 新对局、重开和撤销可取消旧请求；
- 模型后端可在不改 UI 调度的情况下替换。

实际新增文件：

```text
src/agents/agent.js
src/agents/classic-agent.js
src/agents/worker-agent.js
src/workers/classic-ai.worker.js
tests/classic-agent.test.js
tests/worker-agent.test.js
tests/classic-worker-integration.test.js
tests/helpers/browser-worker-bootstrap.js
tests/phase1-ui-contract.test.js
```

### Phase 2：参考 AlphaZero 实现

状态：已完成（2026-07-12）。

目标：得到正确但未扩展的 Python 训练闭环。

计划工作：

1. Python codec 和规则 adapter；
2. PyTorch 策略价值网络；
3. 完整动作 PUCT MCTS；
4. replay buffer；
5. 单机自我对弈；
6. trainer、checkpoint 和 arena；
7. 与 JavaScript fixtures 差分测试。

验收条件：

- Python 与 JavaScript codec/规则一致；
- mock-network MCTS 测试通过；
- 能完成端到端自我对弈和一次参数更新；
- checkpoint 可恢复；
- 小规模训练曲线稳定且非法动作率为 0。

实际新增目录与核心文件：

```text
training/pyproject.toml
training/configs/smoke.json
training/configs/reference.json
training/quoridor/game.py
training/quoridor/codec.py
training/quoridor/network.py
training/quoridor/evaluator.py
training/quoridor/mcts.py
training/quoridor/selfplay.py
training/quoridor/replay.py
training/quoridor/trainer.py
training/quoridor/checkpoint.py
training/quoridor/arena.py
training/quoridor/cli.py
training/tests/
```

### Phase 3：原生引擎和规模化训练

状态：已于 2026-07-13 获得用户确认，并于 2026-07-14 完成实现和完整验收。

目标：移除 JavaScript/Python 规则吞吐瓶颈。

计划工作：

1. Rust/C++ 原生规则和 MCTS；
2. make/unmake、hash 和快速合法 mask；
3. Python binding；
4. 批量叶节点推理；
5. 多 actor 自我对弈；
6. 可恢复 replay/checkpoint；
7. 性能与一致性基准。

验收条件：

- 跨引擎 fixtures 全部通过；
- 吞吐量达到训练预算要求；
- 长时间自我对弈无非法动作和状态漂移；
- 训练可中断和恢复。

### Phase 4：棋力提升

状态：已于 2026-07-14 获得用户确认并开始。4A-4C 工程实现和 full matrix 校准已完成；
正式网络冻结为 `192x16`，Bootstrap/warm-cold pilot 冻结为 32 actors。Phase 4D 已于
2026-07-22 获得确认；teacher 数据和 v3 教师预训练已完成并通过全量审计。首次等算力
warm/cold pilot 在 warm generation 0 的 replay-cache 门禁处安全停止；gate-v2 修复已于
2026-07-25 获得确认并完成 CPU 验收及 current-source memory smoke；
teacher-pretraining-v4 随后已完成并通过全量 bundle 审计。下一步等待确认后在全新 root
重启 warm/cold pilot；该阶段已于 2026-07-26 获得确认并完成，warm 通过冻结的 bootstrap
screen 并被选为后续自我对弈起点。Phase 4D 至此完成；Phase 4E0 Preparation 已于
2026-07-26 获得确认并开始，范围仅限 CPU 工程和审计。

详细准备审计、推荐执行顺序和拟议验收门槛见
[`PHASE4_PREPARATION.md`](./PHASE4_PREPARATION.md)。

目标：训练并证明超过当前困难 AI 的 champion。

计划工作：

1. warm start；
2. 大规模完整规则自我对弈；
3. PUCT、噪声和温度调参；
4. 历史 checkpoint pool；
5. reanalysis 和困难局面采样；
6. 无墙残局表库；
7. 教师模型训练；
8. 竞技场和 Elo 报告。

验收条件：

- 在交换先后手和固定预算下显著超过当前困难 AI；
- 对历史 champion 无明显策略退化；
- 战术题集通过率达到约定目标；
- 结果具有统计置信度。

### Phase 5：浏览器部署

目标：在不破坏现有本地游戏的情况下提供 AlphaZero 专家 AI。

计划工作：

1. 模型蒸馏和量化；
2. 模型 manifest 和 hash；
3. WebGPU 推理；
4. WASM SIMD fallback；
5. Worker MCTS；
6. 懒加载、进度和错误处理；
7. 专家难度 UI；
8. 浏览器性能和兼容性测试。

验收条件：

- WebGPU/WASM 与 PyTorch golden vectors 一致；
- 浏览器始终返回合法动作；
- UI 无搜索长任务；
- 加载失败自动回退经典 AI；
- 浏览器专家 AI 达到约定的棋力和延迟目标。

## 18. 阶段确认协议

用户要求每个阶段开始前确认。执行过程必须遵守：

1. 完成当前阶段后停止，不自动进入下一阶段；
2. 开始每个阶段前报告以下信息：
   - 当前代码和测试状态；
   - 上一阶段验收结果；
   - 本阶段目标；
   - 计划修改的文件和模块；
   - 需要用户决定的信息；
   - 预计风险、计算资源和外部依赖；
3. 明确询问“是否确认开始 Phase N”；
4. 只有收到明确确认后才能修改该阶段代码；
5. 用户修改范围或决策后，更新本文档再执行；
6. 未通过验收时不得把阶段标记为完成。

## 19. Phase 0 确认与执行记录

### 19.1 重复与回合上限规则

已确认：训练和产品统一采用反拖延判负规则。造成第三次重复的行动方判负；执行第 256 个 half-move 的行动方判负。该规则保持双人零和，不产生双方同时为负的训练样本。

### 19.2 训练硬件

已检测到本机训练硬件：

```text
GPU: NVIDIA Quadro RTX 6000
显存: 24576 MiB
检测时可用显存: 22750 MiB
Compute Capability: 7.5
驱动: 590.48.01
```

当前计划只使用本机训练，不依赖云 GPU。训练时长和功耗预算在 Phase 2 开始前再次确认。

Phase 0 本身不需要 GPU，但这些信息会影响 Phase 2-4 的网络规模和训练架构。

### 19.3 原生语言

推荐优先 Rust：

- 内存安全；
- Python binding 和 WASM 部署路径较统一；
- 便于复用同一套核心规则。

如果团队已有成熟 C++ 工具链，也可以选择 C++。

该选择最晚需要在 Phase 3 开始前确认。

### 19.4 Phase 0 实施结果

已冻结并实现以下合同：

- 209 维动作空间，棋子目标格、横墙、竖墙分别占用固定索引段；
- 当前位置 hash 的 codec version 为 `1`，不包含 `moveNumber`、`winner`、终局原因或历史；
- 当前行动方规范化、动作正反变换和左右镜像；
- 三次重复与 256 个 half-moves 均由触发行动方判负；
- 终局优先级为 `goal > repetition > move-limit`；
- 产品界面会按真实终局原因显示结算信息。

第一版跨语言一致性夹具：

```text
schemaVersion: 1
codecVersion: 1
seed: 0x5eed1234
reachable positions: 48
SHA-256: 1b19efd6c1108f9b4ffa763caf4d25ab6c4dd00976be3e94446a732898ea93fa
```

夹具固定每个局面的原始、canonical 和 mirrored 状态、position hash、完整合法动作索引，以及一个抽样合法动作的变换索引和后继 hash。

Phase 0 验收结果：

- 209 个动作全部 encode/decode round-trip；
- 初始局面 131 个合法动作与规则引擎一致；
- 48 个固定可达局面的 mask、变换和后继状态测试通过；
- 额外检查 24 盘共 1,075 个随机合法后继，合法集与后继 hash 等变；
- 重复、步数上限和三类终局优先级测试通过；
- 现有规则和经典 AI 测试无回归；
- `npm test` 全部 6 个测试文件通过。

Phase 1 已于 2026-07-12 获得用户明确确认。用户同时要求取消 AI 对 AI 的固定 3 秒调度等待；这项行为变更进入 Phase 1 范围，不再作为回归约束。

### 19.5 Phase 1 实施结果

已完成以下异步边界：

- `Agent.choose(state, options, signal)` 统一 Promise 接口和稳定结果结构；
- 现有 Negamax 通过 `ClassicAgent` adapter 复用，不复制搜索实现；
- 产品 UI 只加载 `ClassicWorkerAgent`，同步搜索只在模块化 Worker 内运行；
- 请求支持抢占、Abort、5 秒硬超时、stale request id 隔离和幂等 dispose；
- Abort 或超时会终止正在同步搜索的 Worker，后续请求按需重建；
- UI 使用 session version 进行第二层 stale-result 防护；
- 撤销、重开和新对局在 AI 思考时保持可用，并会取消当前请求；
- AI 对 AI 移除固定 3 秒等待，每手在下一动画帧连续启动。

Phase 1 验收结果：

- 经典 AI adapter 保持原动作与搜索统计字段；
- FakeWorker 覆盖请求复用、抢占、Abort、超时、stale、协议错误和销毁；
- 真实 `worker_threads` 往返验证完整棋局状态可跨线程传输并返回合法动作；
- UI 架构合同保证主线程不直接导入同步 `chooseAIMove`；
- 三个浏览器模块均由本地服务以 JavaScript MIME 和 HTTP 200 提供；
- `npm test` 全部 10 个测试文件通过。

本机 Snap Chromium/Firefox 无法用于最终浏览器自动化，因此未完成截图级浏览器检查；Worker
消息边界由真实 Node Worker 集成测试覆盖。

### 19.6 Phase 2 启动记录

Phase 2 已于 2026-07-12 获得用户明确确认。启动时重新检测的本机资源为：

```text
Python: 3.10.12
PyTorch: 2.7.1+cu126
CUDA: available
GPU: NVIDIA Quadro RTX 6000
显存: 23.44 GiB
磁盘可用: 52 GiB（55,523,127,296 bytes）
inode 使用率: 7%
```

Phase 2 复用现有 PyTorch/CUDA，不下载第二套运行时。训练产物软上限设置为 `8 GiB`，smoke
配置只产生小型 checkpoint；至少保留约 40 GiB 磁盘安全余量。Python 参考实现兼容本机
Python 3.10，正式训练环境仍通过 `pyproject.toml` 固定依赖合同。

### 19.7 Phase 2 实施结果

已完成 correctness-first 的 Python AlphaZero 参考闭环：

- Python 规则、209 动作 Codec、canonical/mirror 与 JavaScript Phase 0 合同一致；
- 48 个 conformance fixtures 的 hash、合法动作、变换和后继状态全部差分通过；
- 14 平面 canonical 输入，默认 `192 × 16` SE-ResNet 输出 209 policy logits 和标量 value；
- PUCT 对每个节点展开完整合法动作，不裁剪墙动作；
- 支持动态根噪声 `alpha = clamp(10 / legalCount, 0.03, 0.30)`、温度和逐层价值反号；
- self-play 生成 canonical policy、最终 `+1/-1` value 和左右镜像样本；
- replay、AdamW trainer、8 GiB 产物预算、原子 checkpoint、RNG/optimizer/replay 恢复；
- arena 交换候选模型先后手，结果只有胜负，没有和棋桶；
- CLI 提供快速 `smoke`、标准初始局面 `train` 和 `--resume`。

验收结果：

- Python 标准库 unittest 全部 33 项通过；
- CPU 端到端 self-play、更新、checkpoint 恢复和 arena 通过；
- CUDA 端到端 smoke 和从 checkpoint 继续训练通过；
- 标准初始局面 CUDA 自我对弈在第 79 个 half-move 因三次重复终局，生成 158 个含镜像样本；
- 该完整闭环的 4 次 loss 为 `6.0361, 6.1097, 5.5799, 5.1763`，非法动作数为 0；
- 恢复训练已从 step 4 继续到 step 8，并进一步验证到 step 12；
- 默认 `192 × 16` 网络为 12,023,026 参数，batch 256 FP32 前后向峰值显存约 1,450.8 MiB，耗时约 0.39 秒；
- 当前 smoke 产物约 7.5 MiB，远低于 8 GiB 上限，磁盘仍约有 52 GiB 可用。

smoke arena 只有两局且候选与 champion 各胜一局，不能用于声明棋力提升或发布 champion。
这是 Phase 2 结束时的状态；Phase 3 随后经用户确认并按下一节记录开始实施。

### 19.8 Phase 3 启动记录

Phase 3 已于 2026-07-13 获得用户明确确认。原生实现选择 C++17，使用本机已有的
GCC 11.4、CMake 3.22 和 Ninja；Python 绑定复用 PyTorch 随附的 pybind11 头文件，
不下载第二套编译工具或运行时。

本阶段交付 C++ 规则/合法动作/MCTS、Python binding、训练管线适配，以及由用户本机执行的
一键构建和验收脚本。长时间验收不由开发会话持续监控；脚本会同时写出机器可读 JSON 报告，
用户运行后提供完整终端输出和报告，由下一轮复核。

Phase 3 验收门槛冻结为：

- 至少 10,000 个确定性可达状态与 Python 参考实现逐状态一致；
- 长局自我对弈非法动作数为 0，且无状态漂移；
- 规则和合法动作热路径相对 Python 参考实现至少加速 20 倍；
- 构建、验收和训练均可由脚本显式启动，退出码和报告可用于离线复核；
- 继续遵守 8 GiB 训练产物软上限。

Phase 3 当前实现包括：

- C++17 规则、Codec、完整合法动作、路径检查和不可变状态转换；
- 持久化增量重复历史，MCTS 子节点不再深拷贝整张局面计数表；
- C++ PUCT MCTS、动态根噪声、canonical policy 映射和逐层价值反号；
- 真正的多 root 批调度，每轮合并各 actor 的一个叶节点进行网络推理；
- 无额外下载的 PyTorch bundled-pybind11 Python binding，完整传递重复局面历史；
- Python `NativeMCTS` adapter，self-play/arena/CLI `--engine native` 接入；
- `selfPlay.actors` 多 actor 自我对弈和 `TorchEvaluator.evaluate_batch`；
- native arena 批量调度、candidate/champion 独立保存和拒绝后不晋升；
- generation/actor seed 持久化，resume 不会重放上一轮自我对弈随机流；
- C++ 单测、48 fixtures、10,000 状态差分、长局和性能验收器；
- 一键构建、完整验收、原生训练和自动 checkpoint 续训脚本。

开发会话内的短验收结果：C++ 两组测试和 ASan/UBSan 通过，48 fixtures 通过，1,000 个
随机可达状态和后继状态一致，4 盘完整随机长局非法动作数为 0；代表性短样本中合法动作生成
约 150 倍、已知合法动作推进约 29 倍，均超过 20 倍门槛。CPU 上双 actor 原生训练 smoke
已连续完成两轮 self-play、参数更新、checkpoint/replay/RNG 恢复、generation seed 前进和
批量 arena。完整 10,000 状态、100 盘长局及本机 CUDA smoke 由用户运行
`./scripts/run-phase3-validation.sh` 后回传报告，并按以下结果完成复核。

### 19.9 Phase 3 验收结果

用户于 2026-07-14 在本机运行完整验收脚本，脚本最终返回 `PASS`。原始报告为：

- `training/runs/native-validation/rules-20260714-110105.json`，SHA-256
  `f2c145e60192518ee63e223a6711e4773e7d8079a9215c7ed0619dd859fa5bec`；
- `training/runs/native-validation/training-smoke-20260714-110105.json`，SHA-256
  `7b59d336dac7dc13186d7b16afb98b5fe66ed07b7bd41de4af296a1a4b42d647`。

正式验收结果：

- C++ 两组测试全部通过，Python 和集成测试 47 项全部通过；
- 48 个跨引擎 fixtures、10,000 个确定性可达状态及其 10,000 个后继状态全部一致，
  failure 和非法动作均为 0；
- 100 盘随机长局全部完成，共 5,237 half-moves，最长 135，非法动作和状态漂移均为 0；
- 合法动作生成加速 `144.35x`、已知合法动作推进加速 `28.45x`、组合规则 workload
  加速 `117.55x`，均超过冻结的 `20x` 热路径门槛，Python/native checksum 全部一致；
- CUDA 原生 MCTS 双 actor smoke 完成两轮训练，checkpoint 从 step 4 恢复并推进到 step 8，
  generation 从 0 推进到 1，generation seed 从 20,260,712 推进到 21,260,715，
  replay 从 16 增长到 20，训练指标均为有限值；
- 两轮训练产物分别为 3,702,822 和 7,431,987 bytes，低于 8 GiB 软上限。

100 盘均以 `repetition` 结束是均匀随机长局的预期分布：早期墙动作数量占优，墙耗尽后无目标的
棋子随机游走通常先触发三次重复。该 workload 用于验证长序列一致性，不用于评估策略质量；
`goal`、`move-limit` 及终局优先级由定向测试覆盖，CUDA smoke 的 4 盘自我对弈均以 `goal`
结束。若 Phase 4 的正式模型自我对弈仍长期呈现这一分布，则应作为训练质量问题处理。

因此 Phase 3 的全部冻结门槛已经满足，状态更新为完成。Phase 4 尚未开始，开始前必须重新汇报
当前状态、训练预算、预期产物和验收方式，并等待用户明确确认。

### 19.10 Phase 4 启动准备记录

用户于 2026-07-14 要求开始 Phase 4 准备工作。本轮只完成资源盘点、训练和竞技场代码审计、
预算与验收方案设计，没有修改 Phase 4 代码或启动训练。审计结论是现有 `reference.json`
只能作为 Phase 3 参考 workload，不能直接用于正式高棋力训练；可信多开局竞技场、重复历史
输入/replay 合同、MCTS 子树复用、紧凑存储、数值 fail-fast 和可追溯报告必须先完成。

完整启动提案保存在 [`PHASE4_PREPARATION.md`](./PHASE4_PREPARATION.md)。收到用户对范围、
本机/磁盘限制和 GPU 小时预算的明确确认后，才把 Phase 4 标记为开始并实施 4A。

### 19.11 Phase 4 确认与 4A-4C 实施记录

用户于 2026-07-14 明确确认开始 Phase 4，并冻结以下约束：累计 `72 GPU-hours`、仅本机、
Phase 4 全部产物不超过 `8 GiB`、文件系统低于 `20 GiB` 时停止，训练和长校准脚本由用户
运行并回传 JSON 报告。实施时重新检查项目文件系统约有 `88 GiB` 可用空间，高于停止线；
本轮没有启动 GPU 校准或训练。

已完成的 4A-4C 工程门槛包括：

- 完整重复历史的 16-plane 输入合同、Python/C++ bit-exact 批量编码和数值 fail-fast；
- 按完整对局分片、带 checksum 的紧凑 raw replay，model-only champion archive、运行锁、
  源码/config manifest、原子 generation 报告和可恢复 AMP/LR runtime；
- 常驻 Node 经典困难 AI bridge、成对换边 opening arena、cluster bootstrap、Elo/置信区间，
  以及绑定 native/classic 后端摘要的正式发布 gate；
- 冻结且内容互斥的 200 条 `dev` 与 200 条 `final` 分层 opening 资产；canonical
  `contentHash` 中 `dev` 为
  `sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4`，`final`
  哈希为 `sha256:65f1bb5223e8e7c9da469b69b891443458d631e8058cfb4a14f57f3d1243833d`；
- 持久 MCTS session、子树提升、actor 动态补位和原生状态编码；
- Phase 4 校准器：原始 replay 实际落盘、场景内资源检查、跨运行 GPU 账本、全局单写锁、
  GPU utilization、吞吐/batch/显存/终局指标和严格 `decisionEligible` 合同。

候选 `phase4-*.json` 已标记为 `pipeline=phase4`，旧 reference CLI 会拒绝执行，避免在校准前
误用内存 replay 或旧竞技场。下一步先运行 quick 校准检查本机接线；review 通过后再运行
固定 full matrix。只有 full 报告通过合同检查后，才冻结正式 pilot 配置并实现/启用 4D 训练入口。

### 19.12 Phase 4 quick 校准复核

用户于 2026-07-14 在本机完成 quick 校准，2026-07-21 完成持久化报告复核。运行 ID 为
`phase4-calibration-20260714-191830-2804640`。report、manifest、GPU 账本与当前源码指纹均通过
校验；紧凑 replay 的 2 个 shard 深度复放为 2 局、178 half-moves，没有数据缺失。

`128x10`、2 actors、16 simulations 的场景耗时 `8.375s`，约 `859.71 games/hour`、
`340.06 scheduled simulations/second`，batch p50/p95 均为 `2`。GPU utilization p50 为
`33.5%`，PyTorch peak reserved 为 `40 MiB`；该低负载结果只证明 CUDA、原生 MCTS、批量推理、
遥测和 raw replay 链路连通，不用于选择正式配置。GPU 账本累计 `0.002668 hours`，剩余
`71.997332 hours`。quick 按合同保持 `decisionEligible=false`，状态为通过 smoke gate，下一步
16 场景、960 局的 full matrix 已于 2026-07-21 获得明确确认，仍由用户在本机运行并回传报告。

### 19.13 Phase 4 full matrix 复核与选型冻结

用户于 2026-07-21 完成 full calibration run
`phase4-calibration-20260721-171946-972095`，总耗时 `4.036185 GPU-hours`。独立复核确认
report/manifest/GPU 账本/当前源码/native extension 的全部摘要一致；16/16 replay stores 的
960 局、109,539 half-moves、5,001,528 committed bytes 均通过完整合法复放。20,869,632 次
scheduled simulations 的报告状态为 `completed` 且 `decisionEligible=true`。累计 Phase 4 GPU
使用量为 `4.038853 hours`，剩余 `67.961147 hours`。

以 scheduled simulations/second 执行冻结的 60% 规则时，`192x16` 相对 `128x10` 的八组
匹配比率为 `74.21%-97.13%`，平均 `84.03%`，因此正式网络选择 `192x16`，不运行小网络的
5-generation 棋力 pilot。actor 配置按实际 workload 冻结：64 局 Bootstrap 和 warm/cold pilot
使用 32 actors，与 `192x16/a32/s256/g64` 实测场景完全一致，吞吐为 `1430.43 sims/s`；
每代至少 128 局的 Standard lane 才使用 64 actors，其匹配场景为 `1750.57 sims/s`。不把
64 actors 的两波 128 局结果外推到只有一波的 64 局 workload。

校准对局使用随机初始化网络，只能决定系统吞吐配置，不能证明棋力。GPU utilization p50 仍为
`18%-31%`，但多叶 virtual loss 的实现和搜索偏差风险不应阻塞第一轮可信 pilot；先以当前
baseline 建立棋力结果，再根据 Elo/GPU-hour 决定是否优化。下一阶段是 4D：实现正式可恢复
训练入口、训练显存 smoke、困难 AI 教师数据以及 warm/cold 等算力 pilot。用户已于
2026-07-22 确认开始 4D，并冻结其阶段子预算为 `10 GPU-hours`；CUDA 脚本仍由用户运行。

### 19.14 Phase 4D 启动记录

Phase 4D 的正式训练入口使用 generation `-1` 初始化 checkpoint，把 seeded cold 基础模型、
可选 warm model、空 optimizer 和空 replay 的身份先提交，再开始 generation 0。后续 checkpoint
绑定 config/source/run manifest、trainer/scaler/scheduler/RNG 和 raw replay 高水位；损坏的新
checkpoint 只能回退到仍可验证的上一份，并拒绝与不可变 generation report 冲突的恢复。

teacher 数据默认用经典困难 Negamax 的固定 512-node、max-depth 5 赛道，目标至少 50,000 个
训练可见的去重局面。数据按完整对局分到 90% train 和 10% holdout，并按原局面与水平镜像的
canonical orbit 全局去重，防止 sample-time 镜像增强泄漏到 holdout。所有合法动作都有正 policy
质量，teacher replay 只用于 warm 预训练，不写入后续 MCTS self-play replay。

本轮没有在开发会话中启动 CUDA。项目文件系统复核约有 `94.18 GB` 可用，Phase 4 当前产物
约 `5.19 MB`。首个用户运行 gate 是固定 `192x16`、batch 512 的 AMP 反向传播 smoke；通过后
才执行 teacher 生成、warm 预训练和 warm/cold 各 5 代 pilot。

### 19.15 Phase 4D teacher 数据与预训练恢复记录

用户于 2026-07-22 完成首个训练显存 smoke：
`phase4d-training-memory-smoke-20260722-110812-1624448`。固定 `192x16`、batch 512、
CUDA AMP 的 4 个 optimizer steps 全部通过，峰值 allocated/reserved 分别为
`1,610,482,176` 和 `1,723,858,944` bytes，Phase 4D 计费约 `0.000561 GPU-hours`。

经典困难教师数据随后完成并通过完整复核。不可变数据指纹为
`9eb2f39029758589d753c49d2c765001b4f7fe42f7468af01cfa7233a634fe3c`，共 1,502 局、
50,092 个训练位置；train 为 1,352 局/45,059 positions，holdout 为 150 局/5,033 positions，
没有重复轨迹。数据生成只使用 CPU，不计 GPU 预算。

2026-07-25 启动的旧 `teacher-pretraining` lineage 在用户手动 `Ctrl+C` 后安全停止。GPU
ledger 记录该 run 为 `interrupted`，计费 `3,469.474769s`，有效检查点停在 step 19，
文件大小 `192,983,392` bytes，SHA-256 为
`fbfb1f90bde7929865d6f0fdb4a39a4238730b4e48865a4573fabcfe68e30e1c`。检查点、sidecar、
latest cursor、CRC、模型、optimizer、scaler 和 scheduler 均通过语义复核；旧目录仅保留作
审计，不跨源码或缓存合同恢复。

性能问题来自旧训练路径：每个 512-position batch 都重新读取大量 replay shard 并重放整局，
实际约 3 分钟才完成一个 optimizer step。v2 改为在 GPU ledger 开始前一次性构建 CPU
float32 双水平视角缓存；当前数据的精确大小为 `603,308,048` bytes（约 `575.36 MiB`），
不落盘且低于 1 GiB 上限。缓存严格复现原 `random.sample`、独立 mirror RNG、policy/value
标签和固定 holdout 排列；构建前后复核 teacher/run/train/holdout 四个 manifest 与 native
encoder SHA。前 8 个新 batch 的缓存取数加 H2D 均值必须不超过 1 秒，否则在第 8 次参数更新
前安全停止。

v2 默认写入 `training/runs/phase4/teacher-pretraining-v2`，日志使用 unbuffered Python 和
interrupt-safe `tee`，dataset/cache/training/holdout/checkpoint 均输出实时 JSON 进度。CPU
验收为 Python `212/212`、native `3/3`、Node `11/11` 全部通过，未在开发会话中运行 CUDA。
截至本次中断，global GPU 使用量为 `5.003157613/72 hours`，Phase 4D 使用
`0.964304456/10 hours`，阶段剩余 `9.035695544 hours`；可用磁盘约 `78.79 GiB`，Phase 4
产物约 `196.26 MiB`。v2 源码使旧显存 smoke 的 source fingerprint 按合同失效，因此要求
用户重跑 current-source memory smoke，复核通过后才能启动 v2 教师预训练。

用户随后完成 current-source smoke
`phase4d-training-memory-smoke-20260725-161934-4134656`。report/run-manifest 自哈希、
源码指纹 `4538500ba97d0f900460ccf2bb5b6251562884a299cb4d036b7501a643013947`
和 GPU/stage ledger 均通过复核；11 个 gate 全部为 `true`。4 次真实 AMP optimizer update
耗时 `1.995994s`，测量阶段均值约 `0.102899s/step`，峰值 allocated/reserved 分别为
`1,610,482,176`/`1,723,858,944` bytes，没有 self-play、replay、checkpoint 或 model
artifact 写入。本次计费 `0.000554443 GPU-hours`；Phase 4D 累计使用
`0.964858899/10 hours`，剩余 `9.035141101 hours`。该门禁通过后，v2 教师预训练已具备
启动条件。

在实际启动前的最后一次真实 teacher 文件审计中，发现缓存前置门把 teacher
`run-manifest.json` 的内嵌 self hash 与整个文件 SHA 当成同一种摘要。两者按设计必然不同；
若不修复，预训练会在深验后、缓存开始前 fail closed。实现已改为先用 `load_run_manifest`
校验 self hash 和 lineage，同时独立记录 teacher/run/train/holdout 四个文件 SHA，并在缓存
完成后逐字节复核四份摘要。测试 fixture 也改为真实自哈希 manifest，覆盖合法 self/file hash
不同、前置篡改和构建中替换三种情况。

最终源码在真实 50,092-position teacher 数据上完成纯 CPU 验收：dataset 深验通过，缓存
train/holdout 分别为 `542,690,596`/`60,617,452` bytes，总计 `603,308,048` bytes，构建耗时
`622.057236s`，后置四文件、数据指纹和 native encoder SHA 全部一致。最终回归为 Python
`215/215`、native `3/3`、Node `11/11`。该生产修复把源码指纹更新为
`efddd4b4b112fd973e5237d22d1b0991cb49c258b4a5575b97eb3940c720460e`，因此上一份 smoke
按 current-source 合同再次失效；必须为该最终指纹再运行一次短显存 gate，之后才能启动
v2 教师预训练。

最终 current-source smoke
`phase4d-training-memory-smoke-20260725-170150-4159655` 已由用户完成并通过持久化复核。
report core hash 为
`9085f6d3b8cdb7bfe45f3edfa60dfaf812fc2de4fe00754d748e87f067cdc31a`，run-manifest、
当前源码指纹和 GPU/stage ledger 全部匹配；11 个 gate 均为 `true`。4 次 AMP update 耗时
`1.965129s`，峰值 allocated/reserved 仍为
`1,610,482,176`/`1,723,858,944` bytes，且没有 forbidden artifact。Phase 4D 累计使用
`0.965404768/10 hours`，剩余 `9.034595232 hours`。新的
当时 `training/runs/phase4/teacher-pretraining-v2` lineage 尚未创建，已满足启动条件。

### 19.16 Phase 4D v2 教师预训练完成记录

用户完成 v2 教师预训练后，512 个 optimizer steps、每 64 步一次的 8 次固定 holdout 评估和
8 次 checkpoint 均完整提交。缓存取数加 H2D 的前 8 步均值为 `2.559 ms`，全程均值/最大值为
`2.351/6.452 ms`，远低于 1 秒停止门；optimizer 均值为 `101.126 ms`。训练总损失从
`6.463618` 降至 `0.705341`。

holdout total loss 从 step 64 的 `2.353767` 降到 step 256 的严格最小值 `1.808747`，之后
step 512 为 `1.856018`，比最优值高 `2.61%`。policy loss 持续下降，但 value loss 在后半程
恶化，因此按冻结合同选用 step 256，而不是最终 step 512。`warm-model.pt` 的 280 个 tensor
已逐项确认等于 checkpoint 中的 `best.*`；warm model state SHA-256 为
`03c97f88b8cd2d21bc9eb4b30a371f0d0ec9c1a0bbb938bcd5695390390a3afe`，artifact 文件 SHA-256
为 `89fbd8f2e8038d21afa514a6fb1c5ea7edca213578c1830b09b6a37506d595a7`。step 512 current
state 独立保存在 generation 7 checkpoint，SHA-256 为
`a8a1835584cde2296d5ff97c22b57e64f25fcbadbf13a7b5eb4907539bd7cbcf`。

官方 `load_completed_pretraining_bundle` 在禁用 CUDA 后完整复核 50,092 个 teacher positions，
耗时 `859.26s`，峰值 RSS `717,604 KiB`；summary、report、completion、run manifest、warm
artifact、current source、memory smoke、native backend、baseline 和最新 checkpoint 的所有
绑定均通过。v2 GPU 计费为 `92.991844s`（`0.025831 GPU-hours`）；Phase 4D 累计使用
`0.991236/10 hours`，剩余 `9.008764 hours`。Phase 4 产物共约 `0.596 GiB`，当前可用磁盘约
`78.68 GiB`。监督损失只能证明 teacher imitation，不能证明实际棋力；下一阶段仍必须通过
同 seed、同算力的 cold/warm 各 5 代训练和 64 局成对换边筛选决定是否采用 warm。按阶段确认
协议，pilot 尚未创建，等待用户明确确认。

### 19.17 Phase 4D warm/cold pilot 输入性能 NO-GO

用户于 2026-07-25 明确确认开始等 workload 的 warm/cold pilot。正式创建 pilot 前执行了纯 CPU
生产路径审计，没有启动 CUDA，也没有产生 GPU 计费。统计合同本身通过复核：cold/warm 各 5
代，每代 64 局自对弈、64 个 optimizer steps，按 `cold gN -> warm gN` 交替；最终使用 32 个
冻结 dev openings 换边两局，共 64 局。采用 warm 必须同时满足 raw win rate `>=55%` 和按
opening pair bootstrap 得到的 one-sided 95% lower bound `>50%`。

资源审计发现现有 Phase 4 训练每个 `batch=512` 都调用 `ReplayStore.sample_plies`，对命中的
每个 replay shard 重新解码并从初始状态重放整局；当前没有 cache、prefetch、worker 或配置
开关。生产形态的 64-game replay 只读基准使用 3 个固定 seed，单批 sample 分别耗时
`69.323/69.490/69.387s`，native encode 分别耗时 `0.870/0.867/0.888s`，总计均值
`70.275s`；每批均命中 64/64 局并重放 7,181 plies。128-game replay 已命中 120/128 局并
重放约 14,000 plies，说明后续代次只会更慢。前三次基准没有记录主机级并发负载；中止本轮
`prepare` 和并行审计后，以同一 replay、生产函数和固定 seed 再次复测，sample 为
`68.515s`、encode 为 `0.900s`，总计 `69.415s`，仍是预算允许上限 `28.34s/step` 的
`2.45x`。旧 teacher 原始路径约 `182.6s/step`，而 v2 的 `2.351ms` 均值来自预编码 cache；
这些结果与重放量比例一致。

现有 full calibration 明确不执行训练，memory smoke 也只重复使用预构造 synthetic batch，
因此此前约 `3.93 GPU-hours` 的 pilot 粗估漏掉了 replay 输入成本。即使把首代
`70.275s/step` 当作全部 640 steps 的乐观下界，仅输入准备也需约 `12.49 GPU-hours`；再加
校准得到的 10 代自对弈约 `3.57h` 和 arena 约 `0.36h`，总下界约 `16.42h`，已经超过
Phase 4D 剩余 `9.008764h`，且没有计入 optimizer 和后期 replay 增长。该 CPU stall 位于 GPU
ledger 内，会让 GPU 空转但仍消耗阶段预算，因此属于硬 NO-GO。

仍在运行的纯 CPU `prepare` 深验已安全中止；没有创建 `training/runs/phase4/phase4d-pilot`
或任何 lane。下一步应只在 `training/quoridor/phase4_training.py` 内加入保持采样 RNG、
镜像 RNG 和标签完全一致的内存 replay 重建 cache，并增加 production-shaped CPU 输入性能
gate。由于该文件属于冻结 source fingerprint，修复后必须按合同依次完成 CPU 回归、新
current-source memory smoke、复用同一 teacher dataset 在新 root 生成 v3 warm bundle、完整
审计，以及创建新的 pilot lineage；不能复用 v2 warm bundle 或旧 lane。该修复等待用户确认，
在确认前不修改冻结训练源码。

### 19.18 Phase 4D replay-cache 修复完成记录

用户于 2026-07-25 确认开始 replay-cache 修复。本阶段没有启动 CUDA，也没有创建 cold/warm
pilot lineage。raw replay 继续作为唯一持久真相；新增 cache 只在内存中保存已逐 shard 校验的
`ReplayGame`、每个 ply 对应的重建 `State` 和最终胜者。首次打开时完整重建，后续只接受
manifest 的 append-only 扩展；FIFO eviction、manifest/shard 篡改、构建中变化、容量估算越线
或部分发布都会 fail closed。对外返回的 game、metadata 和 `position_counts` 均复制，调用方
不能污染 cache。

采样继续调用完全相同的 `rng.sample(range(total), k)`，cache 本身不额外消耗 RNG；同一个 RNG
随后交给既有 `ReplayBatchEncoder` 做水平镜像。因此 legacy/cache 的样本顺序、最终 RNG state
及 state/policy/value tensors 必须 bit-exact。训练每代的前 8 个 `batch=512` 在 optimizer
update 前执行 live gate：均值不超过 `2.0s`、单批不超过 `3.0s`；之后单批停止线为 `5.0s`，
累计均值仍不得超过 `2.0s`。配置少于 8 个训练 step 会直接拒绝，未完成门禁不能提交
checkpoint。

训练入口的预算生命周期也同步收紧。resume 的 replay-cache 深验保持在 GPU ledger 前；
fresh run 在 ledger 前只检查路径，不创建 manifest 或 replay。`ledger.begin` 和立即执行的
budget guard 之后，才允许 `resolve_device`、CUDA runtime 查询、fresh lineage 写入、模型创建
和 checkpoint 恢复。source/native/evidence 在 cache 前后和 driver 启动前重复绑定；源码在
manifest 构建期间变化时，原子 manifest 不会提交。

CPU 回归为 Python `240/240`、native `3/3`、Node `11/11`，另有 shell syntax 验证。覆盖
legacy/cache 多 seed 与镜像概率的样本、RNG、tensor 等价，append-only resume、eviction、
manifest/shard/mid-build 篡改、返回对象隔离、连续两代 checkpoint replay snapshot、不可变
gate 报告 schema/self-hash/source/native/runtime 绑定及 fresh/resume ledger 顺序。

正式 production-shaped CPU gate 使用 full calibration 的固定
`192x16-a32-s256-g64` replay：64 局、7,181 plies，fixture manifest SHA-256 为
`f3e872e9ed058935a44d60e93e49b88e729ac1f8d7eae1f5bf5e116133351762`。cache 构建耗时
`67.749825s`，吞吐 `105.9929 positions/s`，通过 `<=120s` 和 `>=60 positions/s` 门槛；
legacy 对照批耗时 `69.060046s`，cache 对照为 `0.948844s`。随后 8 个 cache 批次均值
`0.866197s`、最大 `0.936629s`，通过 `<=2.0s`/`<=3.0s` 门槛，并完成 bit-exact parity。

正式报告绑定 source
`4e674b8b99b3f1b584bd7bee4753a3c03e9284d5e60db6e93af60eb9d33d8da6`、native
`ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3` 和 runtime
`8e70f78c7d29e21eef999124206bcb8c83850d849b869ff866a7954ece14f6c5`；文件 SHA-256 为
`50613c0df52913ed80ce69dec3aa52e23700c21c589ff9adf42ff0248da7468b`。第二次执行在
`1.47s` 内严格复用同一报告。GPU ledger 的 SHA、大小和 mtime 在 gate 前后完全不变；
当前 global/Phase 4D 使用量仍为 `5.030089/0.991236 GPU-hours`，Phase 4D 剩余
`9.008764 hours`。Phase 4 产物约 `615 MiB`，可用磁盘约 `79 GiB`。

由于 `phase4_training.py` 已改变，旧 memory smoke 和 v2 warm bundle 按 current-source
合同失效。下一顺序冻结为：用户运行新的 current-source memory smoke；通过复核后复用同一
teacher dataset，在新 root `teacher-pretraining-v3` 生成 warm bundle；完整审计 v3；再次确认
后才创建 pilot contract 并运行 cold/warm pilot。pilot 包装脚本会在创建任何 lane 前先复核
本节 replay-input gate，已有合格报告时只做幂等复用。

### 19.19 Phase 4D replay-cache 后 current-source memory smoke

用户于 2026-07-25 完成 run
`phase4d-training-memory-smoke-20260725-195255-79326`。官方
`find_valid_memory_smoke` 重新验证 report 自哈希、run manifest、自身 source binding 和冻结
`192x16/batch512/CUDA AMP` 合同后，选择该 run 为最新合格证据。source SHA-256 仍为
`4e674b8b99b3f1b584bd7bee4753a3c03e9284d5e60db6e93af60eb9d33d8da6`，与正式
replay-input gate 完全一致。report core SHA-256 为
`e145a7d1e395d977f840ccf51bc59c03d41c8765a913e5a22044e57e2a686809`，report 文件
SHA-256 为 `231d0b4b70e32f5f19416ce0f573910a3db702a2ba2182e9b1214c6056ae3a9f`，
run manifest 自哈希为
`ee8146ec5e198cb54855c91493b0e74eee810244037e71a1902fbacfd6867fec`。

4 次 optimizer update（1 次 warmup、3 次 measured）全部完成，11 个 gate 均为 `true`。
measured step 均值为 `0.102077s`，约 `5,015.80 samples/s`；模型参数保持 float32，forward
输出和 GradScaler 均实际使用 CUDA float16 AMP。PyTorch peak allocated/reserved 分别为
`1,610,482,176`/`1,723,858,944` bytes，与前一份显存 smoke 完全一致；`nvidia-smi` 捕获
2 个样本。最终 run 目录只含 `run-manifest.json` 与 `report.json`，没有 self-play、raw
replay、checkpoint 或 model artifact。

共享 GPU ledger 为本次运行记录 `1.998588s`，即 `0.000555163 GPU-hours`。global 使用量更新
为 `5.030644156/72 hours`，Phase 4D 使用量为 `0.991790999/10 hours`，阶段剩余
`9.008209001 hours`。基线报告仍通过官方复核，Phase 4 产物约 `615 MiB`，可用磁盘约
`79 GiB`。`teacher-pretraining-v3` root 尚不存在；在明确确认前不创建其 lineage。

### 19.20 Phase 4D v3 教师预训练完成记录

用户于 2026-07-25 在全新 root `training/runs/phase4/teacher-pretraining-v3` 完成 512 个
optimizer steps。终端末尾的 `zsh: command not found: --new` 来自粘贴时把 `--new` 作为
训练结束后的独立 shell 命令执行；实际 launcher 在 root 尚不存在时以 auto 模式创建全新
lineage，进度明确记录 `resume=false`，没有恢复或修改 v2。

训练前再次深验 1,352/150 局、45,059/5,033 positions 的 train/holdout 数据；teacher dataset
仍为 `9eb2f39029758589d753c49d2c765001b4f7fe42f7468af01cfa7233a634fe3c`。双水平视角
cache 为 `603,308,048` bytes，构建耗时 `517.946628s`。前 8 步输入均值为 `2.338ms`，
全程输入均值/最大值为 `2.333/5.940ms`，远低于 1 秒停止门；optimizer 均值为
`100.392ms`。

8 次固定 holdout 评估与 v2 bit-exact。严格唯一最优仍为 step 256：total/policy/value loss
分别为 `1.808746580430487`、`0.9211527647992003`、`0.8875938156312866`。导出的 280 个
warm tensors 与 v2 逐张量相同，model state SHA-256 为
`03c97f88b8cd2d21bc9eb4b30a371f0d0ec9c1a0bbb938bcd5695390390a3afe`；最终 step 512
current state SHA-256 为
`a8a1835584cde2296d5ff97c22b57e64f25fcbadbf13a7b5eb4907539bd7cbcf`。v3 warm artifact
文件 SHA-256 为
`d8cac4beec3bd5a7d1875ab9b3af8453a0845d8184ba4dee593c8b676fd43d6a`，绑定本次 report
文件 SHA-256
`ca665b50462f2b955a36300f2ee973635f951ceb5ac6eec7feb42f20ad722e14`，没有引用 v2
report。

最终 checkpoint 为 generation 7/step 512，文件 SHA-256 为
`e5d23d0b4f756307ac6652f670220ddbf3fc5b62e307a2fc96c2aee46274c1ae`；只保留 step
448/512 两份 checkpoint，latest cursor 和 fallback 校验通过。预训练 source 为 25 个文件、
`770,654` bytes，聚合 SHA-256 为
`0283c9f8bf7c27592d3afc53be3c9d090eabb630fe8828795e35e30090f9b33c`；native encoder
仍为 `ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3`。run manifest
self hash 为 `18e942947d462db530bf52561f4bdf9b4ce77867034e4ac64b6fbd36aef75729`，
completion self hash 为
`5766b51a71b5de772dae27aba419be04b364c3441b35c6a87ea825e2e40a3562`。

官方 `load_completed_pretraining_bundle` 在禁用 CUDA 后完整复放 50,092 positions 并验证
summary、report、completion、run manifest、warm artifact、最终 checkpoint、source、native、
memory smoke、baseline 和 teacher dataset 的全部交叉绑定；耗时 `14m17.12s`，峰值 RSS
`718,096 KiB`，零 swap、零文件写入，结果为通过。v2/v3 的 holdout、current/best tensors、
optimizer、trainer runtime、RNG 和 replay snapshot 均完全一致；文件 SHA 差异只来自新的
lineage、时间和资源元数据。

v3 GPU 计费为 `93.010155s`，即 `0.025836154 GPU-hours`。最终 global 使用量为
`5.056480310/72 hours`，Phase 4D 使用量为 `1.017627154/10 hours`，阶段剩余
`8.982372846 hours`。v3 目录为 `434,239,165` bytes，整个 Phase 4 为
`1,074,414,575` bytes（8 GiB 上限的 `12.51%`），可用磁盘约 `78.22 GiB`。

manifest 内部 runId 仍使用冻结的 `phase4d-teacher-pretraining-v2-20260714` 训练器版本标签。
v2/v3 由不同 root、source/config 和 manifest SHA 完整区分，checkpoint、warm artifact、GPU
ledger 和 pilot 消费链均不依赖该标签，因此本阶段将其记录为低风险元数据债务，不为此使已验证
bundle 失效。下一步是在明确确认后创建 warm/cold pilot contract。

### 19.21 Phase 4D replay-cache gate-v2 修复完成记录

用户确认首次 warm/cold pilot 后，cold generation 0 已完整提交：64 局、7,713 plies，
replay-cache 重建耗时 `57.446153s`，吞吐 `134.264864 positions/s`，64 个训练 batch 均值/
最大值为 `0.788068/0.878469s`，checkpoint、generation report 和 replay 均已持久化。warm
generation 0 的 64 局、2,725 plies 也已完整写入 raw replay，但在 optimizer update 前被
`replay cache build throughput fell below 60 positions/s` 门禁安全停止，因此没有 warm
generation 0 checkpoint 或 report。失败运行已按 ledger 计费，未发现非法动作、CUDA 错误
或 replay 损坏。

旧门禁对所有 `decodedPlies >= 1024` 的 cache refresh 强制要求 `>=60 positions/s`，隐含
重建成本只随 plies 线性增长。只读复核显示 warm replay 的完整规则重建约需 `99-101s`；
其短局和墙局面结构带来更高的逐局固定成本与合法墙验证成本，因此按 positions/s 与固定
7,181-ply fixture 直接比较会产生误报。raw replay 仍是唯一持久真相，现有 cold/warm 产物
保留为审计证据，不跨新源码谱系恢复。

用户于 2026-07-25 确认开始 gate-v2 修复。对非空 production replay refresh，冻结的
时间上限为：

```text
allowedSeconds = 120 * max(decodedGames / 64, decodedPlies / 7181)
```

refresh 必须满足 `refreshSeconds <= allowedSeconds`。该合同同时按局数覆盖固定解码/逐局
校验成本，并按 plies 覆盖长局增长；当前 warm workload 的上限为 `120s`，cold workload
约为 `128.89s`。固定 64-game/7,181-ply production-shaped fixture 仍独立保留原
`<=120s` 和 `>=60 positions/s` 的正式性能门禁，不能用 live workload 的缩放上限替代。
manifest/shard 哈希、完整规则重建、append-only、容量估算、构建中变化检测和 4 GiB
replay-cache 内存停止线均保持 fail closed。

实现使用本次 `decodedGames/decodedPlies` 而非 replay 总量计算 envelope，因此增量 refresh
不会重复计入已缓存局面，resume 全量重建则自然按全部 decoded workload 缩放。空 refresh
跳过构建性能门；非空 refresh 的两个 decoded 轴必须同时为正；负数、不一致计数及非有限
耗时均 fail closed；等于上限时通过。固定 fixture 通过显式 cap 调用旧合同，production 的
五个 fresh/resume/incremental 调用使用 gate-v2。cache 算法和持久格式未变化，因此
`REPLAY_CACHE_ALGORITHM_VERSION` 保持为 `1`。

新增的 7 个 gate-v2 测试覆盖空/小 workload、warm/cold 双轴选择、增量只统计 decoded
workload、resume 局数轴和 plies 轴、临界值与 epsilon 超限、固定 fixture 双门槛保留，以及
非法计数 fail-closed。完整 Python 测试为 `247/247 PASS`，native CTest 为 `3/3 PASS`，
Node 测试为 `11/11 PASS`。

新的 current-source production replay-input gate 绑定源码
`bb48c84b08cf641f7b2c5cc8384b4380e841322383c0de763004e5be7930a165`、native
`ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3` 和 runtime
`8e70f78c7d29e21eef999124206bcb8c83850d849b869ff866a7954ece14f6c5`。fixture cache
构建为 `67.892506s`、`105.770141 positions/s`；legacy/cached parity 路径为
`68.749682/0.940666s`；8 个 batch 的均值/最大值为 `0.866847/0.933026s`。sample、RNG、
tensor parity 和全部性能门均通过。报告 SHA 为
`8fbd9bbbb641192cfdf916a423f6822200c67bb433ce711d671e99741260dafb`，随后重复调用命中并
验签同一报告。warm 已记录的 `100.970509s` refresh 在 gate-v2 下获得 `120s` 上限并通过。
本阶段没有运行 CUDA。

由于修复会改变冻结训练源码指纹，现有 memory smoke、v3 warm bundle 和首次 pilot lineage
将按 current-source 合同失效。CPU 验收通过后，后续顺序必须为：用户运行新的 memory smoke；
在新 root 生成并完整审计 `teacher-pretraining-v4`；再在全新 pilot root 重启 cold/warm，
不能把首次 pilot 的 cold checkpoint 或 warm replay 接入新谱系。截至修复开始，global GPU
使用量为 `5.601031824/72 hours`，Phase 4D 使用量为 `1.562178667/10 hours`，阶段剩余
`8.437821333 hours`；CPU 验收后账本数值不变，Phase 4 产物约 `1.226 GiB`，可用磁盘约
`77.98 GiB`。下一步必须先获得用户确认，再运行绑定新源码的 Phase 4D memory smoke。

### 19.22 Phase 4D gate-v2 current-source memory smoke 完成记录

用户于 2026-07-25 确认开始 current-source memory smoke，并运行
`./scripts/run-phase4d-training-memory-smoke.sh`。运行
`phase4d-training-memory-smoke-20260725-234218-208557` 状态为 `completed`，冻结的
`192x16`、batch 512、FP32 参数、CUDA AMP/FP16 autocast 合同完成 1 个 warmup step 和
3 个 measured steps，共 4 个 forward/backward/AdamW/scheduler step。policy/value 输出形状、
AMP scaler、optimizer state 和 scheduler epoch 均符合合同。

PyTorch 峰值 allocated/reserved 显存为 `1,610,482,176/1,723,858,944` bytes，与此前稳定
smoke 完全一致；清理后 allocated/reserved 降至 `17,039,360/33,554,432` bytes。两个
`nvidia-smi` 样本成功落盘，设备仍为 Quadro RTX 6000，24 GiB 可见显存。全部 11 个门槛为
true；审计确认 self-play、replay、checkpoint 和 model artifact 写入均为 0，synthetic batch
没有保留。

官方 immutable report/run-manifest 加载器、`find_valid_memory_smoke` 和
`verify_memory_smoke_evidence` 均重新验签通过。证据绑定 gate-v2 源码
`bb48c84b08cf641f7b2c5cc8384b4380e841322383c0de763004e5be7930a165`；report core/file SHA
分别为 `76a4c5a5095e9ffeeb214eb15f4ba4e6372a914736c6a605b5eaab540208651b` 和
`bffe36bc7367bdb07c49307ae80c16b3617265fc9f19601aa190728e65d35d18`；run-manifest core SHA
为 `b80484530c03010df782b3c3f4bb83c4b0a09bb54f422b5593160d8b9eb33208`。

本次 GPU 计费为 `1.962317s`，即 `0.000545088 GPU-hours`。完成后 global 使用量为
`5.601576912/72 hours`，Phase 4D 使用量为 `1.562723755/10 hours`，阶段剩余
`8.437276245 hours`。Phase 4 产物为 `1,316,457,583` bytes，空闲磁盘约 `77.98 GiB`。
下一阶段必须先获得用户确认，再在新 root 执行 `teacher-pretraining-v4`；v3 继续作为旧源码
审计证据，不能直接作为 gate-v2 pilot 的 current-source warm bundle。

### 19.23 Phase 4D teacher-pretraining-v4 完成记录

用户于 2026-07-25 确认开始 `teacher-pretraining-v4`。非 GPU 预检确认目标目录
`training/runs/phase4/teacher-pretraining-v4` 尚不存在，可用 `--new` 创建独立 lineage。
冻结教师数据仍为 1,352 个 train games/45,059 positions 和 150 个 holdout games/5,033
positions，共 50,092 个唯一 raw state 和 augmentation orbit；duplicate trajectory 为 0，
dataset fingerprint 仍为
`9eb2f39029758589d753c49d2c765001b4f7fe42f7468af01cfa7233a634fe3c`。

v4 专用 25-file source fingerprint 为
`0be76672708ef64c9901b19f050ccebad0ad3ce002c7fcc8c0bc2e2a544f2da9`，完整训练源码指纹为
`bb48c84b08cf641f7b2c5cc8384b4380e841322383c0de763004e5be7930a165`。官方 gate 已发现并
验签 current-source memory smoke
`phase4d-training-memory-smoke-20260725-234218-208557`，native encoder SHA 仍为
`ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3`。v3 专用源码指纹
`0283c9f8...9b33c` 与旧 memory smoke 绑定，仅保留作审计证据。

冻结 workload 仍为 `192x16`、seed `20260714`、batch 512、CUDA AMP、512 optimizer steps，
并每 64 steps 执行完整固定 holdout 评估。首次运行必须使用一条完整命令：

```bash
./scripts/pretrain-phase4d-teacher.sh --run-root training/runs/phase4/teacher-pretraining-v4 --new
```

dataset validation 和约 603 MB preencoded cache 构建发生在 GPU ledger 开始前；预期 CPU
阶段约 17 分钟，GPU 训练约 93 秒。若 CPU 阶段中断，可去掉 `--new` 以 auto 模式重跑；
若训练阶段中断，launcher 会提交 completed-step checkpoint，随后用同一 run root 和
`--resume` 严格恢复。启动前 global 使用量为 `5.601576912/72 hours`，Phase 4D 使用量为
`1.562723755/10 hours`、剩余 `8.437276245 hours`；Phase 4 产物约 `1.226 GiB`，空闲磁盘
约 `77.97 GiB`。

用户随后运行上述单行命令。launcher 最后一行仅报告日志路径；日志中其前一条 JSON 已明确
`status=completed`。数据验证覆盖 1,352/150 个 train/holdout games，CPU 双镜像缓存以
`515.517113s` 构建完成，大小为 `603,308,048` bytes。CUDA AMP 训练完成全部 512 个
optimizer steps 和 8 次固定 holdout 评估；最佳结果位于 step 256，total/policy/value loss
分别为 `1.808746580`、`0.921152765` 和 `0.887593816`。step 512 的最终 holdout total loss
为 `1.856017727`。

模型产物
`training/runs/phase4/teacher-pretraining-v4/warm-model.pt` 的文件 SHA 为
`e1e7dbba95e18ea22b0e1c3a16ca22c4eae89c493651f00ca6ef81888f1c6f8d`，选中模型 tensor-state
SHA 为 `03c97f88b8cd2d21bc9eb4b30a371f0d0ec9c1a0bbb938bcd5695390390a3afe`。最终 checkpoint
为 generation 7/step 512，文件 SHA 为
`d0dabaf1408343a7b9bc348c3c70767819cd2558c63d8c9ed622a07cec3efb18`。report core/file SHA
分别为 `ed4e1593a7fc73229c81c0d3e0c451ef21a081e8cf29669561d349811de96cee` 和
`fd40ea591ee9a47d858ba85ca36f09286a1f9c59ce6b1e399249ff5208bf8b8e`；run/completion
manifest SHA 分别为 `8299ffea447c9af99672bf254cfcbd5c89ae3455100f6eda50f6c44268601a5f` 和
`7083d169a5e7dffbcc0d1d203c34540e1fc51c9c69af360dff93e4362d1196c1`。

独立产物审计确认所有 canonical/self hash、账本绑定、保留 checkpoint 和 artifact lineage
一致，且没有临时文件、符号链接或孤立文件。v4 与 v3 的 warm tensor 逐张量完全相等，8 次
评估及所有确定性训练指标也完全一致；这是同一数据、seed 和训练合同下的预期结果，v4 的新
文件 SHA 来自 gate-v2 源码、current-source memory smoke 和路径血缘变化。官方
`load_completed_pretraining_bundle` 又以禁用 CUDA 的方式完整复放全部 50,092 个局面并验签
bundle，耗时 `14:15.24`、峰值 RSS `719,488 KiB`，退出码为 0。

v4 GPU 计费为 `94.837762s`，即 `0.026343823 GPU-hours`。最终账本全局使用量为
`5.627920734/72 hours`，Phase 4D 使用量为 `1.589067578/10 hours`，阶段剩余
`8.410932422 hours`；账本 self SHA 为
`27912d154c9850fddd04b24cfe01b036ac9335af0513ef1dca1dee4fd5191435`。审计时 Phase 4
产物为 `1,750,730,431` bytes（约 `1.630 GiB`），空闲磁盘约 `77.59 GiB`，继续满足
`8 GiB` 产物上限和 `20 GiB` 停止线。旧 `phase4d-pilot` root 已绑定旧源码 partial lineage，
不可复用；下一阶段必须先获得确认，再以 v4 summary 和全新 pilot root 重启等算力
warm/cold pilot。

### 19.24 Phase 4D warm/cold pilot-v2 启动记录

用户于 2026-07-26 确认开始 current-source warm/cold pilot。非 GPU 启动审计确认目标
`training/runs/phase4/phase4d-pilot-v2` 尚不存在；旧默认 `phase4d-pilot` 已包含旧源码
partial contract、cold generation 0 和未完成的 warm generation 0，不能恢复到新谱系。

v4 summary 文件 SHA 为
`9296de67ed5f38a9bf1cb480aff10698933a31381d34e8ee5a00b813f5ab4f53`，canonical bootstrap
config 文件 SHA 为 `08d884616da9b8d6ee19c3d6cf29c15100aad61aa624499e2ec8a955eb38a887`。当前完整训练
源码/native SHA 仍分别为
`bb48c84b08cf641f7b2c5cc8384b4380e841322383c0de763004e5be7930a165` 和
`ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3`。正式 replay-input
gate 已重新验签并复用报告
`8fbd9bbbb641192cfdf916a423f6822200c67bb433ce711d671e99741260dafb`，启动输入一致。

固定 workload 为 cold/warm 各 5 代，严格按 `cold g0 -> warm g0 -> ... -> cold g4 ->
warm g4` 交错；每代 64 局、32 actors、每手 256 simulations、64 optimizer steps。两条
lane 完成并终验后，自动使用 32 个冻结 dev openings 换边两局，运行 64 局 paired screen。
唯一允许的首次启动与恢复命令为：

```bash
./scripts/run-phase4d-warm-cold-pilot.sh --pretraining-summary training/runs/phase4/teacher-pretraining-v4/summary.json --output-root training/runs/phase4/phase4d-pilot-v2
```

wrapper 会先运行 native tests、current-source replay gate 和 v4 bundle 全量深验；bundle
深验约需 14 分钟 CPU 时间，期间没有训练进度属于正常现象。随后每次只把下一条 lane 推进到
绝对累计 generation target，并在返回调度器前深验 checkpoint、replay 和 provenance。
lane 训练中断后使用同一完整命令恢复；已提交局和 checkpoint 保留，checkpoint 已写而 report
未写的窗口会 reconcile。arena 中断则从 64 局头部重跑，partial JSONL 不算结果且已消耗 GPU
时间仍计账。不得省略 `--output-root`，也不得在 contract 创建后修改源码、native、base config
或 v4 bundle。

启动前全局 GPU 使用量为 `5.627920734/72 hours`，Phase 4D 使用
`1.589067578/10 hours`，剩余 `8.410932422 hours`。以旧 cold generation 0 的较慢实测
`0.414307h` 作为全部 10 代代理，再加校准 arena `0.356991h`，保守估算完整 pilot 为
`4.5001 GPU-hours`，预计阶段仍余约 `3.9109 hours`；硬阶段账本继续 fail closed。当前
Phase 4 产物为 `1.6305 GiB`，预计完成后约 `2.3 GiB`；空闲磁盘约 `77.58 GiB`，继续满足
`8 GiB` 产物上限和 `20 GiB` 停止线。CUDA 训练仍由用户运行脚本并回传完整终端结果。

### 19.25 Phase 4D warm/cold pilot-v2 完成记录

pilot-v2 严格按冻结 schedule 完成 cold/warm 各 generation 0-4，共 10 代、640 局自我对弈
和每条 lane 320 个 optimizer steps。10 个训练日志均以 `status=completed` 收尾。cold replay
为 320 局/21,974 plies，warm replay 为 320 局/14,175 plies；两边均通过完整规则重放，
`teacherReferences=0`。初始化证明显示 cold 从冻结随机基线
`51e500bbc612222bdc4902438af9dbfcf367c95cc523d5384309403e8b3ba010` 开始，warm 从 v4
选中模型 `03c97f88b8cd2d21bc9eb4b30a371f0d0ec9c1a0bbb938bcd5695390390a3afe` 开始，二者
optimizer 和 replay 初始均为空。

cold 最终 checkpoint/model-state SHA 分别为
`e1a181b1338da77eb600e5e94ce54d05e398647655e1f32bded74456ce9bf777` 和
`80308641b2917e049a5b80c7902d23826277221545f88a98cadce38c360c912e`；warm 分别为
`032290a1fb77ed129c2d2e654f1d03c8007d6ea01fedabf80dcc5fe04d8147f7` 和
`a69073538072b191845a21e1ce35f583bc954af7b484c64ed83bdc8388e59b0a`。官方 lane validator
在禁用 CUDA 下分别深验两条 lane，均返回 `status=completed`、`nextGeneration=5` 并以
退出码 0 结束。contract core SHA 为
`6b6cb5126a3fd1a719d33a43c48d3243a5e0f12b78591a3642e88761addfaf78`，源码和 native
仍严格绑定 gate-v2 指纹。

最终 paired screen 使用 32 个冻结 dev openings 各换边两局，共 64 局/1,907 plies。独立
复放确认每手均合法，before/after/final position hash、move number、winner 和 termination
reason 全部一致；61 局以 goal 结束，3 局以 repetition 结束。warm 取得 62-2：30 个 opening
pair 为 2-0，另 2 对为 1-1；作为 player 0/player 1、先走/后走四种拆分均为 31-1，没有
可见偏侧。按 opening pair、seed `20260714` 做 10,000 次 cluster bootstrap，独立复算 raw
win rate 为 `0.96875`、单侧 95% 下界为 `0.921875`，正确通过冻结的 `>=0.55` 和 `>0.50`
双门槛，因此 `selectedLane=warm`。

arena games/report SHA 分别为
`469e68a0c3503a76a19d16506d57aea3e1d19afb43d07c48aa58da638fb2af00` 和
`d35d2d57ee233be71569e1386c224b40c4208c6e00873d30c0c8639cf6065648`；result manifest
文件/core SHA 分别为 `12328f88bef07fcf350b7a2722083450a5eb8f0b2c64380564ad29b48422b2e9` 和
`83fcd90c2f86edb565b09aebccf9575ec9705f4d535c5c9c2b75a02dd6c7f67a`。文件大小、self hash、
checkpoint/config/source/native lineage 和 64 个唯一 game id 均验签通过，没有 partial
arena 文件。

pilot-v2 实际计费 `2.831778171 GPU-hours`，其中 cold `1.250019540h`、warm
`0.765041248h`、arena `0.816717383h`。最终全局使用量为 `8.459698905/72 hours`；
Phase 4D 从冻结 baseline 起累计使用 `4.420845748/10 hours`，剩余 `5.579154252 hours`。
GPU 账本 self SHA 为
`8b1e870fc183a4819276e97430323213e800ff322667c3d4046d9663993fd5bd`，11 个 pilot-v2
计费 run 均为 completed，lane、arena、提交快照与最终账本闭合。Phase 4 产物约
`2.175 GiB`，其中 pilot-v2 为 `584,118,474` bytes；空闲磁盘约 `77.01 GiB`，资源门槛
全部通过。

该结果的合同、report 和 result manifest 均明确
`screenOnly=true/formalChampionDeclaration=false`。因此可以确认 warm start 在同算力
bootstrap pilot 中有可信优势，并把 warm g4 作为后续自我对弈起点，但不能据此声明正式
champion 或发布模型。一个非阻断 metadata debt 是新旧 pilot root 内的 lane `runId` 使用
相同固定名称；实际 lineage 已由 contract/config/source/manifest SHA 区分，后续跨目录汇总
不得只把 `runId` 当唯一键。Phase 4D 至此完成；进入 Phase 4E 前必须按阶段协议重新确认。

### 19.26 Phase 4E 前置审计与待确认边界

完成 Phase 4D 后的只读 gap audit 结论是不能直接启动 Standard lane 或正式 champion gate。
warm g4 只通过 64 局 bootstrap screen，不是正式 champion；其纯 self-play replay 为
320 局/14,175 plies，仍低于 `100k` replay 后才从 256 提升到 512 simulations 的扩容条件。
当前 `phase4_training.py` 还绑定 Phase 4D 阶段账本、5 代上限和 `promotion=not-run`，通用
model-only champion 初始化、Phase 4E provenance 和正式晋升编排尚未接通。

现有 replay cache 按约 32 KiB/ply 全量重建；300k Standard replay 约需 `9.16 GiB`，会触发
4 GiB cache 硬停止线，必须先改为可恢复的流式、分块或受控索引采样。正式 arena 当前逐
opening/leg 串行且不能按 pair 恢复；按本次 64 局/256 simulations 实测线性外推，一次
400 局/800 simulations 约需 `15.95 GPU-hours`，不能直接用于频繁 gate 或多历史对手回归。

建议下一阶段限定为 **Phase 4E0 Preparation**，只做 CPU 工程和测试，不运行 CUDA：

1. 冻结 pilot-v2 完成证据，将 warm g4 提取为 model-only `bootstrap-baseline`，明确
   `formalPromotion=false`；
2. 参数化阶段预算、generation ceiling、通用 model-only 初始化和 Phase 4E lineage；
3. 将 warm 的 14,175 条纯 self-play replay 作为可审计迁移候选，同时默认重置 optimizer；
4. 实现确定性、可恢复、受内存上限约束的 replay 流式/分块采样；
5. 新增 Phase 4E driver，编排健康门、64 局 screen、400 局 confirmation、历史
   non-inferiority 和仅在全门通过后的 `ChampionStore` 原子发布；
6. 给正式 arena 增加跨局批处理、actor refill、按 opening pair 持久化恢复、预算 heartbeat
   和顺序 fail-fast；
7. 新增版本化 Bootstrap/Standard 配置、launcher、CPU fixtures 和恢复/篡改测试，不修改
   已被 pilot contract 绑定的 `phase4-bootstrap.json`。

Phase 4E0 完成后再单独确认 CUDA 校准、GPU 子预算和 Bootstrap 续训。建议届时先做
128 games/512 simulations/64 actors 的 Standard-shaped calibration 与 batched formal-arena
throughput smoke，再据实冻结预算；当前不承诺正式 gate 时长。用户已于 2026-07-26 明确
确认开始 Phase 4E0；本次确认不授权 CUDA 校准或训练。

### 19.27 Phase 4E0 启动与运维合同

Phase 4E0 的版本化 execution contract 为
`training/configs/phase4e-execution-prepared-v1.json`。它固定 `stage=phase4e`、
64 代绝对 ceiling 和正式晋升语义，但状态保持 `prepared`，且
`stageBudget.status=pending`；baseline GPU hours、阶段上限和校准报告 provenance 均为空。
因此它是可哈希的准备合同，不是 GPU 授权。

版本化 workload 包括：

- `phase4e-bootstrap-v1.json`：已校准形状 `64 games / 256 simulations / 32 actors`，
  replay 100k、batch 512、每代 64 updates；
- `phase4e-standard-v1.json`：`128 games / 512 simulations / 64 actors`，replay 300k、
  每代 96 updates，并记录 400 局/800 simulations 的正式 confirmation 目标。该文件当前
  只是下一轮 calibration candidate，不是冻结 workload；
- 已被 Phase 4D pilot contract 绑定的 `phase4-bootstrap.json` 保持不变。

CPU-only 准备和深度复核入口为：

```bash
./scripts/prepare-phase4e.sh
./scripts/prepare-phase4e.sh --verify-only
```

它们验证 pilot-v2 完成证据，提取 warm g4 model-only bootstrap baseline，并将 warm 的
320 局/14,175 plies 复制为独立、可审计的 replay migration candidate；不会导入 optimizer，
同时构建/测试 native engine 并提交 current-source CPU replay-input gate；不会声明
champion，也不会写 GPU ledger。用 `Ctrl+C` 停止后可重跑同一命令，完整不可变
产物会被复核复用。Replay 准备需要对 source、target 和最终 lineage 做重复 CPU 深验，可能
持续数分钟；launcher 会先输出阶段提示，该静默窗口不属于 GPU 训练。

统一训练入口为 `./scripts/train-phase4e.sh`。当前默认 prepared contract 会在 native build
和任何 CUDA 访问前 fail closed。未来只有在单独确认 CUDA 校准、阶段 GPU 子预算和
baseline provenance，并提供 `frozen` contract 后才能启动。训练中断时按 `Ctrl+C`；完整
replay games 和最后一个校验通过的 checkpoint 保留，重跑完全相同的命令恢复。自动化应使用
`--target-generations N` 的绝对目标，避免恢复时多推进一代。

CPU 准备和未来训练继续共享 Phase 4 的 `8 GiB` 总产物上限和 `20 GiB` 可用磁盘停止线。
Phase 4E0 不运行 CUDA，不新增 GPU 计费。下一边界仍是单独确认 Standard-shaped calibration、
batched formal-arena throughput smoke 和据此冻结的 GPU 子预算；未确认前不得启动
Bootstrap 或正式 champion gate。

### 19.28 Phase 4E0 完成证据

2026-07-26 的 CPU-only 验收结果如下：

- prepared execution contract canonical SHA-256：
  `636ecdd40dc999614963e8f64eb7e58e533dc5302b7b2adb0e7ebaa46a8f3d87`；
- model-only bootstrap baseline manifest SHA-256：
  `dc1c33619997732e5525840d1c867969ce82ede6d7ecb25092e962f63dd15294`，
  `formalPromotion=false`；
- replay migration manifest SHA-256：
  `372bd024abe700d0284effa2a5ca5dd1de622d4fc4dcf920327dcd4b61ba7174`，
  包含 320 局和 14,175 plies，`lineageVerified=true`；
- current-source replay-input gate source SHA-256 为
  `1a831d7fbb7c074a4f53fd32ce9ab6999d8581ce50b222ff3a87b93f902a2da7`，
  report SHA-256 为
  `09b4e2f5f92c0e133c6a3382d762dc9496b47321c707a7c3cfd91601921f47e0`；
- 新 checkpoint v2 同时提交完整恢复文件和 lineage-bound、weights-only 可加载的
  `candidate-model`；health、arena identity/contract 和 champion 发布统一绑定后者；
- promotion screen 固定为 `dev[:32]`，confirmation/history 固定为完整 200 个 `dev`
  opening，`final` holdout 不参与训练期选择；arena 会重放验证每一手和重新计算成对统计；
- history 从完整 accepted archive 派生最近 5 个不同前代并 fail closed；冠军发布使用跨进程
  CAS/目录锁，同时对整个 Phase 4 root 执行 8 GiB 上限、20 GiB 磁盘 reserve 和路径约束；
- `prepare-phase4e.sh --verify-only` 端到端通过；Python 317 项、原生 C++ 3 项和
  JavaScript 11 项测试全部通过；
- Phase 4 产物共 2,384,410,721 bytes，低于 8 GiB 上限；验收时可用磁盘约
  76.95 GiB，高于 20 GiB 停止线；
- GPU 总账保持 8.459698904923 hours，本阶段没有 CUDA 运行或新增 GPU 计费。

已提交的 execution contract 仍为 `prepared/pending`，因此 `train-phase4e.sh` 会在 native
build 和 CUDA 访问前拒绝运行。Phase 4E1 的 Standard-shaped calibration、batched
formal-arena throughput smoke 和阶段 GPU 子预算必须再次明确确认后才能开始。

### 19.29 Phase 4E1 校准合同与运行边界

用户已于 2026-07-26 明确确认开始 Phase 4E1。开始时 GPU 总账为
`8.459698904922698 / 72 GPU-hours`，ledger SHA-256 为
`8b1e870fc183a4819276e97430323213e800ff322667c3d4046d9663993fd5bd`；
warm generation 4 model-only baseline、prepared execution contract 和两份 Phase 4E
workload 均已通过 CPU preflight。本次确认只授权本机 CUDA 校准与吞吐 smoke，不授权
Bootstrap 训练、训练 checkpoint、正式晋升或 champion 发布。

不可变 E1 合同为 `training/configs/phase4e1-calibration-v1.json`，canonical SHA-256
为 `47fbe72edd9c75c10b97b0b3f1475205bf6d1630234314a5d11d1ceb43899955`。它固定：

- self-play 使用 Standard `192x16` 网络形状、128 局、64 actors、512 simulations，
  `temperatureMoves=24`、seed `20260714`；
- arena smoke 使用 `dev[:40]` 的 40 个 opening pair，共 80 局、64 actors、双方各
  800 simulations；candidate/opponent 都必须绑定同一份 warm Bootstrap model-only
  artifact，必须实际经过 actor refill；完整 arena search payload 的 canonical SHA-256
  为 `89545b401d92a26c2d659ea4736ea6ffc76a56ab7733f8fe1c1f6636c0f6428a`，
  同时绑定 Standard config、PUCT 参数、零温度、关闭 root noise 和 seed `20260714`；
- E1 最多消耗 6 GPU-hours；继续执行全局 72 GPU-hours、Phase 4 产物 8 GiB 和可用
  磁盘 20 GiB 停止线；
- 校准结束后保留 2 GPU-hours，全局余量向下取整到 `1e-6 GPU-hours` 后必须仍能给
  Phase 4E 正式阶段至少 48 GPU-hours；正式阶段 ceiling 固定为 64 generations。

运行入口为：

```bash
./scripts/calibrate-phase4e1.sh
```

脚本先执行原生 C++ 测试和 current-source replay-input gate，再运行两个 CUDA workload。
`Ctrl+C` 会让当前 ledger invocation 以 interrupted 结束；重跑相同命令时，已提交的
self-play report/replay 或已原子落盘的完整 candidate，以及完整 opening pair，会先在 GPU
计费外深验，再只补缺失工作。Arena 的 evaluation batch histogram、每秒 GPU telemetry、
峰值显存和累计时间写入自哈希 progress，因此恢复后的报告不会只采用最后一次 attempt 的
样本。已有 partial pair store 会在新 ledger invocation 和 CUDA 模型创建前逐手深验；
progress 只有在对应 ledger heartbeat 落盘后才可持久化。组件和最终报告均先验证候选内容
与外部证据，再提交不可变文件；最终报告还嵌入自哈希 ledger snapshot，后续只能追加计费
记录，不能回滚报告基线。

最终报告固定为
`training/runs/phase4/phase4e1/calibration/phase4e1-calibration-47fbe72edd9c75c1/report.json`。
完整 verifier 会重新绑定静态 E1 合同、prepared contract、run manifest、当前 source/native
backend、两个精确 effective workload hash、Bootstrap baseline、全部 self-play replay
以及逐手重放的 arena result。训练授权边界会再次调用同一完整 verifier，并重新执行
`2h reserve / >=48h / 64 generations` 策略；仅凭一份可自签 summary JSON 不能解锁训练。

launcher 不会自动写
`training/configs/phase4e-execution-frozen-v1.json`。用户回传最终 summary 后，需先审核
实际 GPU 用量、self-play 吞吐、arena batch/利用率/显存和完整 provenance，再在 Phase 4E1
内单独执行 CPU-only freeze。完成 freeze 后，启动 Bootstrap 训练仍属于新的阶段边界，必须
再次确认。

### 19.30 Phase 4E1 准备验收

Phase 4E1 runner 的 CPU-only 准备验收已完成：

- Python 353 项、原生 C++ 3 项和 JavaScript 11 项测试全部通过；
- current-source replay-input gate 通过，source SHA-256 为
  `1d26d503d9861cb902ac602f46d1db3dde6bec1f93f7b0491efb836947623b34`，
  gate report SHA-256 为
  `e8592ca55a6eb750f97cbf9fc9947f40be96984857b0b5ea70c05b75b65d586c`；
- GPU 总账仍为 `8.459698904922698 / 72 GPU-hours`，ledger self-hash 仍为
  `8b1e870fc183a4819276e97430323213e800ff322667c3d4046d9663993fd5bd`；
- Phase 4 产物为 2,384,412,824 bytes，可用磁盘为 82,548,449,280 bytes，分别满足
  8 GiB 上限和 20 GiB 停止线；
- CUDA 校准尚未运行，固定 E1 report 尚不存在，execution contract 仍是
  `prepared/pending`，训练继续保持锁定。

### 19.31 Phase 4E1 arena telemetry 修复

首次本机 E1 运行完成了 128 局 self-play，随后在 arena 第一个完整 opening pair 的持久化
深验中停止。Native persistent MCTS 每步仍严格执行合同规定的 800 次新模拟，但提升复用
子树后，`visit_counts` 表示包含已有访问的累计 root visits；旧 selector 错误地把该累计值
同时写入 `SelectionResult.simulations`。实际失败 pair 的前两次搜索为 800，从 ply 2 起为
919 至 1362，metadata 中的 `configuredSimulations` 始终为 800。

修复后：

- `simulations` 严格表示本次配置预算，必须等于 800；`nodes/rootVisits` 记录累计访问量，
  `retainedRootVisits` 记录复用部分，arena verifier 校验完整字段和算术关系；
- pair 在任何 durable write 前完成两局逐手 referee replay 和 selection 深验，写后从磁盘
  重新读取 canonical JSON 复验；坏数据不会留下新的 pair 文件；
- arena cleanup 会保留原始异常，reconcile 或资源清理异常只作为 cause，不再覆盖根因；
- 旧 run 绑定 source
  `1d26d503d9861cb902ac602f46d1db3dde6bec1f93f7b0491efb836947623b34`。
  下一次启动将在全局锁内把它整体原子移动到内容寻址的 quarantine 目录，并生成由新
  run manifest 绑定的自哈希 incident；旧 self-play 和 pair 不会跨源码混入新报告。

失败 invocation 已按实际墙钟时间记录 `1793.482689930126s`，即约
`0.498189636 GPU-hours`；全局总账为 `8.9578885410144 / 72 GPU-hours`，状态为
`failed`，没有遗留 `running` 项。该用量会保留并计入 E1 的 6 GPU-hour上限。

修复后的 CPU 验收结果：

- Python 359 项、原生 C++ 3 项和 JavaScript 11 项测试全部通过；
- current-source SHA-256 为
  `7fb9fd3ecd0568e183204f77142189aeb77f175cdd4e4ac603919a18fbdd5acd`；
- replay-input gate report SHA-256 为
  `619dc8e101aad359e7ada2f9c0378ead99d9df72b4c0f731a4cbcdf12dc54bf8`，
  cache throughput 为 `103.948 positions/s`，batch mean/max 为 `0.871s / 0.938s`；
- Phase 4 产物为 2,385,070,092 bytes，可用磁盘为 82,562,084,864 bytes，继续满足
  8 GiB 上限和 20 GiB 停止线；
- 固定 E1 report 仍不存在，execution contract 仍为 `prepared/pending`，训练保持锁定。

### 19.32 Phase 4E1 完整校准与最终审计

修复后的 Phase 4E1 完整运行已于 2026-07-26 完成，最终报告
`reportSha256` 为
`80f3f5807b24af9830a593928caae1a4512fb32c0f92be0ee83fa358eccb21a3`。
完整 verifier 重新深验了 128 局 self-play、6,013 plies、40 个 opening pair 和 80 局
arena，并验证当前 source、native backend、prepared contract、Bootstrap baseline、replay、
逐手选择元数据、账本快照和 quarantine incident。最终状态为 `decisionEligible=true`，
但仍为 `trainingAuthorized=false`、`trainingPerformed=false`、
`formalPromotion=false` 且 `checkpointsWritten=0`。

校准实测如下：

- self-play 完成 128 局，吞吐为 `371.990 games/hour`、`2,485.305 scheduled
  simulations/second`；GPU 利用率 p50 为 `24%`，峰值 NVIDIA 显存为 `1,593 MiB`；
  player 0 以 82-46 占优，后续训练需监控先手偏斜，但不阻断本次吞吐校准；
- arena 完成 40 对/80 局、3,004 plies，吞吐为 `122.558 games/hour`、`1,022.675
  scheduled simulations/second`；64 actors 确实发生 refill，GPU 利用率 p50 为 `23%`，
  峰值 NVIDIA 显存为 `1,646 MiB`；
- arena 每个 ply 的 `simulations` 均严格等于 800；累计 root visits 与
  `retainedRootVisits` 的关系通过逐手深验。双方使用同一 baseline，最终 40-40 且每对
  换边局动作序列相同，只提供 40 条独立轨迹；它证明换边对称性和调度正确，不构成棋力或
  晋升证据，正式异模型 arena 的吞吐也可能略低；
- 旧源码失败 run 已密封至 quarantine，incident SHA-256 为
  `d163edbbfe3457e04927640e608d691f8f35894a74b02c041ccffe4115ac136b`，
  其中 136 个文件和 `607,978` bytes inventory 均已验证。

E1 的失败与成功 invocation 合计使用 `1.579260774677 GPU-hours`，最终报告内的全局账本
快照为 `10.038959679600 / 72 GPU-hours`。按保留 `2 GPU-hours` 并向下取整到
`1e-6 GPU-hours` 的合同，建议冻结 Phase 4E 正式阶段上限为
`59.96104 GPU-hours`，最多 64 generations；64 是 ceiling，不是保证可以全部完成的
运行时预测。按本次吞吐，如果 64 代包含 12 次 400 局 confirmation，仅 self-play、screen
和 confirmation 已约需 `63.33 GPU-hours`，尚未包含训练和 history gate。因此正式执行应
按 5 代分块，并在首个完整 promotion cycle 后重新估算剩余预算。报告生成后账本又追加
4 次 memory smoke，共 `0.002408113539 GPU-hours`；
这些记录保持账本单调，不影响报告有效性，但会按 `currentGlobal - baselineGlobal` 计入
正式阶段用量。

2026-07-28 最终审核时，Phase 4 文件产物为 `2,387,859,790` bytes，可用磁盘超过
`75 GiB`，继续满足 8 GiB 产物上限和 20 GiB 停止线。用户随后明确确认 CPU-only
freeze；完整 verifier 再次通过并原子写入
`training/configs/phase4e-execution-frozen-v1.json`，文件 SHA-256 为
`0ae33064ebd4ded3d9f9f0b96e9aaa89a736dcf3b5b05bcaa44d418afc9743ad`。
合同绑定报告基线 `10.038959679599696 GPU-hours`、正式阶段上限
`59.96104 GPU-hours`、最多 64 generations 和 `formalPromotion=true`。Freeze 未访问
CUDA、未修改 GPU 总账，也未启动训练；Phase 4E stage ledger 将在首次训练入口中初始化。
Bootstrap 训练仍是新的阶段边界，必须再次确认。

### 19.33 Phase 4E Bootstrap 首个五代分块（历史记录，已暂停）

用户已明确确认开始 Phase 4E Bootstrap。CPU-only 启动审计确认：

- frozen execution contract、E1 baseline report、warm generation 4 model-only baseline、
  current source/native backend 和 replay-input gate 全部重新绑定通过；
- `phase4e-bootstrap-v1.json` 的 effective SHA-256 为
  `a79b892a193c1968e58cfbbf0a931eaa063ef9fedade8b108af4f58bd3ff3c24`，
  属于 E1 最终报告的 approved workloads；
- driver 会把 warm baseline 权重严格加载到 generation `-1`，验证 tensor-state SHA，
  并以空 optimizer 开始；配置中的可复现 random base 只用于基础模型合同；
- Bootstrap run directory 尚不存在，没有残留 manifest、replay、checkpoint 或 GPU lock；
  Phase 4E 当前实际剩余 `59.958631886461 GPU-hours`；
- 相关训练 lifecycle、预算 provenance 和运维入口共 34 项 CPU 测试通过。

首个有意义的绝对目标为 5 个已提交 generations，即 generation 0 至 4。每代运行 64 局、
32 actors、每步 256 simulations、前 24 手带温度，然后执行 batch 512 的 64 个 optimizer
steps；generation 4 是首个五代 candidate cadence。当时冻结的命令如下，仅用于历史审计，
不得执行：

```bash
./scripts/train-phase4e.sh \
  --execution-contract training/configs/phase4e-execution-frozen-v1.json \
  --config training/configs/phase4e-bootstrap-v1.json \
  --target-generations 5
```

以上是首次启动时冻结的历史命令，现已暂停且不得重跑；失败后的新源码不再满足旧 E1 报告、
`frozen-v1` 和 Bootstrap run manifest 的 source provenance。

当时的运维合同要求显式 frozen contract，并禁止为恢复添加 `--new`；参考同形 warm pilot，
五代曾预计约使用 `0.77 GPU-hours`。这些历史说明不构成当前运行授权。

本分块只执行 self-play、训练、replay/cache 和 checkpoint/candidate-model 提交。训练
driver 不自动消费 arena 配置，每代报告会明确写入 `promotion.status=not-run`。Generation 4
完成后必须先停下审核；health、64 局 screen、400 局 confirmation、history gate 和 champion
发布属于后续独立阶段，开始前需再次确认。Standard workload 虽已被 E1 报告批准，但当前
没有从 Bootstrap 最新 checkpoint/replay 自动切换的桥接，不得宣称五代后会自动转入
Standard 训练。

### 19.34 Bootstrap 初始化失败与 E1 rebaseline 准备

首次 Bootstrap 运行在创建新 lineage 后、加载 generation `-1` baseline 时失败。
`Phase4TrainingDriver.__init__()` 在方法局部 lazy-import 了 bootstrap verifier，而
`initialize_new_run()` 把该名称当作模块全局使用，因而抛出
`NameError: verify_bootstrap_baseline is not defined`。失败审计确认没有提交 generation、
checkpoint、generation report、replay game、replay ply 或 optimizer step；这不是可恢复的
训练进度，也不能通过重跑旧命令绕过。

失败 invocation 记录了 `1.345055362210s`，即
`0.0003736264895026882 GPU-hours`；当前全局总账为
`10.041741419628574 / 72 GPU-hours`。即使没有产生训练进度，已测得的 GPU 墙钟仍属于真实
资源使用，必须保留在单调 ledger 中，不能回滚、删除或抵扣。72 GPU-hours 全局预算、
8 GiB 产物上限和 20 GiB 磁盘停止线继续有效。

用户确认的 **CPU-only Bootstrap 初始化修复与 E1 rebaseline 准备** 已完成；该确认不授权
CUDA、E1 calibration 或 Bootstrap 训练。本阶段完成了：

1. 修复 generation `-1` model-only baseline 初始化的名称解析，并增加直达 driver 生命周期
   的回归测试；
2. 深验失败 Bootstrap root 确实为零进度，再将整个 lineage 连同 manifest、空 replay/cache
   元数据做保留式 quarantine，并在 incident 中绑定失败日志和 ledger snapshot；
3. 将旧 E1 最终报告和 `phase4e-execution-frozen-v1.json` 标记为历史 provenance，并通过
   可审计 supersede 流程保留，不能原地伪装成新源码证据；
4. 为后续新 E1 报告、`frozen-v2` 和 stage-ledger rebase 准备 fail-closed 工具与 CPU 测试。

最终源码指纹为
`a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381`。
失败 Bootstrap lineage 已移动到内容寻址 quarantine；其 incident 自哈希为
`a3165c13a606a94c42c6f59775e2244f760ebddcb35cede65aabb7a77c69295d`，
并绑定失败日志 SHA-256
`45833c2494d5472c6daeb5ec1dcb618f44f22030ff1d66c66fec890fd421c427`。
旧 E1 完成证据已保留式 supersede；incident 自哈希为
`70089388d5e4c2198de68ceea40868402b8f65316fd28bfe2ea9f3aade176d43`，
supersession attestation 自哈希为
`3b38cc88b7a4a7905bff13a07154293a83856b4bb0a7e5498750e8d251c1e224`。
两项 execute 均通过幂等重验。

current-source replay input gate 已通过 64 局、7,181 plies 的完整校验，CPU 缓存构建吞吐
为 `105.3108 positions/s`；报告 SHA-256 为
`2643d1fc1e67cec4756f66525daab30a3af1e69902f94fa3108e9da7c8701959`。
原生测试 3/3、完整 CPU Python 测试 381/381 均通过。修复改变了 source fingerprint，
因此旧 E1 报告、`frozen-v1` 和旧 Bootstrap manifest 只作为历史证据保留，不能继续授权
新源码。

CPU 修复、preservation quarantine、current-source 验收、新的 E1 calibration 和
`phase4e-execution-frozen-v2.json`/Phase 4E stage-ledger rebase 均已完成。下一步 Bootstrap
训练仍是独立阶段，需要用户另行确认；global GPU ledger 及全部失败计费保持单调。
frozen-v2 和 ledger rebase 已通过。

### 19.35 Phase 4E1 rebaseline calibration 完成

用户已于 2026-07-28 明确确认并完成新的 Phase 4E1 CUDA rebaseline calibration。启动前的
CPU-only 审计结论为 GO，最终报告也已通过官方 verify：

- source fingerprint 为
  `a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381`，
  native backend 为
  `ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3`；
- 新 run manifest 将唯一绑定 supersede incident
  `70089388d5e4c2198de68ceea40868402b8f65316fd28bfe2ea9f3aade176d43`；
- global ledger 为 `10.041741419628574 / 72 GPU-hours`，无 running entry；
- E1 合同的累计 6 GPU-hour cap 已使用 `1.5820425147058756`，剩余
  `4.417957485294124 GPU-hours`；
- Phase 4 产物约 `2.22 GiB`，可用磁盘约 `76.13 GiB`；
- current-source replay gate 已复核为 `reused=true`，相关 CPU 测试 38/38 通过；官方
  report verify 返回 `decisionEligible=true`。

最终报告为
`training/runs/phase4/phase4e1/calibration/phase4e1-calibration-47fbe72edd9c75c1/report.json`，
SHA-256 为
`c162921ba56ed0d4d6b86bc430a74963bddb54fe2029ec02478d34ac4c587039`。本次 self-play 为
128 局、64 actors、512 simulations；arena smoke 为 80 局、64 actors、800 simulations；
两者均完成且 `trainingPerformed=false`、`checkpointsWritten=0`。全局 ledger 为
`11.115946631046116 / 72 GPU-hours`，本次 Phase 4E1 累计 `2.656247726123418 GPU-hours`。

新 E1 报告已审核，frozen-v2 已创建并完成 stage-ledger rebase。frozen-v2 SHA-256 为
`453c60235859f417d5f7b295d0d764ac5778a7b9927bdfcedcb4dade0b304264`，completed receipt
SHA-256 为 `3c7cdd651b107fb4204a020af2813145f1c693e76d164a8a0ac2f045492c6849`，最终
stage ledger SHA-256 为 `a05cf7b338b25c281730f24946cecd7c8f0e8d189d04c2b294c632d395c6ec46`。
该 rebase 没有修改 global GPU ledger，也没有启动训练。

启动和中断恢复由用户运行的命令仍为：

```bash
./scripts/calibrate-phase4e1.sh \
  --contract training/configs/phase4e1-calibration-v1.json \
  --device-index 0 \
  --gpu-hours-used 0
```

`Ctrl+C` 会把当前 ledger invocation 标记为 `interrupted`，已计费用量不回滚。完整 self-play
组件可深验复用；未完整提交的 self-play 会保留 attempt 后从头重跑；arena 按已原子提交的
opening pair 继续。运行期间冻结所有参与 source fingerprint 的源码和脚本。当前 rebase 已
完成，但该确认不自动开始 Bootstrap 训练；训练仍需独立确认。

### 19.36 Phase 4E Bootstrap 首个五代分块完成

用户确认后，Bootstrap generation `0` 至 `4` 已全部完成。每代提交 64 局 self-play、64 个
optimizer steps、深度 replay/cache 校验和 checkpoint/candidate-model；最终 step 为 `320`，
累计 replay 为 `320 局 / 12,813 plies`。最终完整 checkpoint SHA-256 为
`9af626e93f61f704df3df31017c93b8f606c6bb4db977d2dbd16e68df0007cae`，candidate-model SHA-256
为 `5a410ca04ab244863ef7540812ac6eb3635a8b4a51061a7de3f71ce55effc639`。

训练 source fingerprint 仍为
`a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381`，最终全局 GPU ledger 为
`11.996742850779814 / 72 GPU-hours`，Phase 4E 剩余约 `58.003257 GPU-hours`，无 running
entry。每代报告的 promotion 状态均为 `not-run`；health、screen、confirmation、history
和 champion 发布必须作为新的确认阶段执行。

### 19.37 Phase 4E generation 14 正式晋升

Bootstrap 后续分块已训练至 generation 14、step 960。Generation 14 通过模型健康门、
64 局 dev screen（42-22）和 400 局 dev confirmation（254-146，胜率 63.5%，90% 下界
60%），随后原子发布为首个 formal champion
`g00000014-s00000000-67b580c9087f`。Champion 模型 SHA-256 为
`67b580c9087f9f4c73f4a3ca3887b05bfdadd57baa20df2406bea542bed45696`，发布包文件
SHA-256 为 `ff301ffc54523a84bb45efcf78baabdda8b094bbe35b3c493ea2411aad5ce49b`。
发布后 ChampionStore 深验通过；`latest` 指向新 champion，`best`/`product` 仍为空。

截至发布，全局 GPU 总账为 `33.78169428860691 / 72 GPU-hours`，无 running entry。
该晋升证明候选超过 warm bootstrap baseline，但尚未证明超过经典困难 AI，因此不能直接
进入产品发布。

### 19.38 Phase 4F0 Final Evaluation Preparation 完成

用户确认后已完成 CPU-only Phase 4F0。新增的混合 arena 调度保持候选 native MCTS 跨局面
批量推理，同时通过可信经典 bridge 的 stateless session facade 执行经典搜索；每个 opening
pair 仍是唯一原子提交和恢复单位。正式 runner 每 30 秒输出进度并把 GPU 使用写入既有
Phase 4E stage ledger。

准备阶段只在 8 个 `dev` 局面上运行经典 AI。850 ms 校准的节点数范围为实际冻结报告中的
样本分布，最终 fixed 节点预算为 `587`。校准观察到内部 850 ms 截止可有少量超调，因此
wall-clock 合同请求 `800 ms`，但仍以实测每手 `<=850 ms` 作为硬门槛。`final` 开局仅做
结构和哈希验证，未运行任何对局。

冻结合同为 `training/configs/phase4f-final-frozen-v1.json`，文件 SHA-256 为
`bfd4abc327518b6215b27c16bd673ebb5247fe950fcc99d8d0a46fb73c82f6b2`，合同自哈希为
`d56e1fbd1fa83ad200f839fb7f3c7997bc0b5d975e555db6a128ec36cd8e7423`。评测源码指纹为
`e821a422f026210a2813130a510c450f8d6dd2e9d9534f74dd725340fd21a40d`。准备报告明确记录
`cudaAccessed=false`、`lockedFinalGamesRun=0` 和 `productReleaseAuthorized=false`。

完整 Python 测试为 386/386，原生测试为 3/3；`prepare-phase4f.sh --verify-only` 已通过。
下一边界是分别确认并运行 fixed 400 局和 wall-clock 400 局；两个赛道都会额外单独记录
标准初始局面的两盘，不会自动设置 `best` 或 `product` 指针。

### 19.39 Phase 4F1 Fixed 失败分析与恢复准备完成

Phase 4F fixed track 已完成 200 个 final 开局对、400 局正式对局，并额外完成标准初始
局面的两盘。候选取得 71 胜 329 负，胜率为 17.75%，双侧 95% CI 为
`[14.25%, 21.50%]`，未达到胜率 60% 和 CI 下界 55% 的产品门槛。标准初始局面两盘均负。
身份、运行时、800 simulations、经典 AI 587 节点预算以及全部逐步规则重放均通过，因此
结论是棋力失败，不是评测损坏。

CPU-only 诊断确认：66% 的成对开局两盘全负；原始 player 0/1 胜率为 22.5%/13.0%；
开局仅落 0-4 墙时胜率只有 11.67%；败局中 55.09% 的候选决策仍给出 `value >= 0.5`。
详细报告位于 `training/runs/phase4/phase4f/failure-analysis/report.json`，自哈希为
`39676508b615d0777de668a8fedf9dd865f1c26f14122b27b5a8d1134dcab278`，恢复方案见
`docs/PHASE4F_FAILURE_RECOVERY.md`。

本阶段还修复了已完成 gate 重开时 tuple/list JSON 规范化导致的对象比较假失败；改为比较
canonical JSON bytes。该问题不影响已经存储的对局、统计或 gate 内容。由于诊断已经查看
旧 final，禁止将其精确轨迹用于恢复训练；下一次产品评测必须冻结新的未见 final suite。
wall-clock track 不再执行。下一阶段为单独确认的 CPU-only 恢复工程，不授权 GPU。

### 19.40 Phase 4F4 Challenge Freeze 与 Rebaseline Preparation 完成

Phase 4F2 完成 champion model-only 初始化、challenge replay 混合、健康诊断和评测隔离工程；
Phase 4F3 随后生成了 `30,047` 个隔离的 classic-hard challenge 局面。Phase 4F4 已对 train
`27,025` 局面/`538` 局和 holdout `3,022` 局面/`60` 局执行深度重放，并重新核对全部
`21,449` 个禁止评测轨道。数据集指纹为
`2a4864ad464e0448b405e4e8879b40c06249dc2bcea420fcabd204ebbd8b9e95`。

completed challenge contract 自哈希为
`f8b30795269ca9ec17f5e0beec145e5932d4cb92ebe07eb4ca4afe780abac8f7`；recovery workload
文件 SHA-256 为 `74f39388aef5426a8f821c179a5e1da0a45272b48112abcc375ca798913fd7cf`，语义配置
SHA-256 为 `5cba128159e943646259738d1aa6ad707f40ee7ada08cb522ece286c6686eb69`。CPU freeze 报告
自哈希为 `c2a27d4c85df8c9732d5fe9f043c58f3ea155806050a7d59320bc3b28cbd1472`。

本阶段没有访问 CUDA；global GPU ledger 保持 `43.14977626422679 / 72 GPU-hours`，原生
测试 3/3、完整 Python 测试 403/403 通过。recovery workload 仍未获训练授权，因为旧 E1
基线没有批准新的 mixed challenge workload。下一阶段是需单独确认的 Phase 4F5 CUDA
rebaseline；它只做显存、输入吞吐、可恢复一代 pilot 和预算冻结，不自动开始正式恢复训练。

### 19.41 Phase 4F5 CUDA Rebaseline Preparation 完成

用户确认 Phase 4F5 后，CPU preparation 已完成并冻结。`prepared-v1` 因冻结相对 `runDir`
而训练驱动校验绝对 `runDir`，在 self-play 前发生配置身份误判；显存 smoke 已完成并正确计费。
修复后的 `prepared-v2` 同时绑定文件语义和路径解析后的有效语义，保留 v1 为不可变历史证据。
v2 随后在 champion 加载前发现 F2 workload 把旧种子基模 hash 用于 recovery 新种子；该失败
发生在 self-play 前并已正确计费。`prepared-v3` 冻结新的 recovery/pilot v2 配置，绑定按
`20260803` 实际构造的基模 SHA-256
`87e87ec6f4f77b6156d56e095b1c23685d10d33cf14038ca4623b8a89d0b0120`，并改用空的
`pilot-v2` runDir。v3 随后成功加载 champion 并构建完整 challenge cache，但 generation `-1`
checkpoint 合同遗漏 `champion` 模式。`prepared-v4` 新增严格的 published-champion checkpoint
字段、路径和哈希验证，并使用空的 `pilot-v3` runDir；生产 champion provenance 已直接通过
validator。当前源码 SHA-256 为
`6c6da19633ac648c8bc0c7d124ebf2ff8d05a0bce9fd3e6c1f8ef4679bdf5a0e`，v4 preparation 自哈希为
`c1554613bd125705e2879f536f03f11eea477e716246cbcaf779f48d190215b1`。校准执行合同将阶段限制为
一代和 `2 GPU-hours`，SHA-256 为
`d2d939b97d55def02e6bef772a2693f51aba87c754e53d40df997ae1583e8e53`。

隔离 pilot 严格复用正式 recovery workload 的网络、初始化、MCTS 和训练参数，仅修改名称与
runDir。它执行 `128` 局、每手 `512 simulations`、batch `512`、`96` optimizer steps，
challenge replay 比例为 `0.25`。GPU 基线仍为 `43.14977626422679 / 72 GPU-hours`；校准后必须
至少保留 `26.5 GPU-hours`，分别用于正式恢复训练 `14`、development screens `3` 和新 final
`9.5`。全部失败尝试均保留并计费后 global ledger 为 `43.29081197835091`，Phase 4F5
剩余额度为 `1.858964285875878 GPU-hours`；generation-14 champion 已再次 CPU 深验，native
3/3 与完整 Python 410/410 已通过。

CUDA 部分由用户运行以下可恢复脚本：

```bash
./scripts/run-phase4f5-rebaseline.sh --device-index 0 --gpu-hours-used 0
```

本 preparation 不启动正式恢复训练。只有脚本完成 memory smoke、mixed-input gate、单代 pilot
和最终 freeze 后，才会生成 `phase4f-recovery-execution-frozen-v1.json`；正式训练仍需下一次阶段
确认。

### 19.42 Phase 4F5 CUDA Rebaseline 完成

最终 v4 pilot 已完成 `128` 局、`5,968 plies`、每手 `512 simulations` 和 `96` 个 batch-512
optimizer steps；challenge replay 比例为 `0.25`。输入 batch 平均/最大耗时为
`0.051227s / 0.815025s`，显存 allocated/reserved 峰值为
`1,610,482,176 / 1,723,858,944 bytes`。所有校准 gate 通过，幂等 finalize 复核返回
`reused=true`。

Phase 4F5 累计使用 `0.8179142071425716 GPU-hours`，global ledger 为
`43.96769047136936 / 72 GPU-hours`。报告自哈希为
`55f08d39ac7374af67b6e60682868689dc074bbba3adcba8e7532093939be8b2`，receipt 自哈希为
`454b78a317fbc40d53edb3111cd7e4cd487f7a282beba3b228d15691b22344ec`。

正式恢复执行合同 SHA-256 为
`58e3eaff20acd4f6e737dab99bb5b3811b4a2620f94afad62501fac5cb755184`，冻结 `14 GPU-hours`、最多
`16 generations`，并保留 development screens `3` 与新 final `9.5 GPU-hours`。合同已授权但
`trainingStarted=false`。下一阶段为需单独确认的 Phase 4F6 CPU-only launcher/driver preparation；
它必须先增加并测试 `phase4f-recovery` stage 验签，不能直接启动正式训练。

### 19.43 Phase 4F6 Formal Recovery Launcher/Driver Preparation 完成

Phase 4F6 已完成 CPU-only 接入，没有访问 CUDA，也没有创建
`training/runs/phase4/phase4f/recovery-v2`。训练驱动现在支持 `phase4f-recovery` stage，并在
GPU 台账开始计费及 lineage 创建前，严格验签 F5 报告/receipt、正式 execution contract、
`recovery-v2` workload、当前源码、native 后端和 GPU ledger 追加性。

F6 preparation 位于 `training/configs/phase4f6-recovery-prepared-v1.json`，自哈希为
`2699ceb08952d9c1cd7e90afd3ee52ac7a2d3b60136a9528ba6411a9a8c387b7`；绑定源码 SHA-256 为
`ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`。准备时 global GPU
ledger 保持 `43.96769047136936 / 72 GPU-hours`，正式 recovery 上限为 `14 GPU-hours`，后续
仍预留 development screens `3` 和新 final `9.5 GPU-hours`。Native 3/3、完整 Python
415/415 通过。

可恢复启动器为 `scripts/run-phase4f-recovery.sh`。下一阶段确认后先以
`--target-generations 4 --new` 运行首个四代分块；`Ctrl+C` 后用相同绝对 target 自动恢复，
或显式使用 `--resume`。四代完成后先审查训练健康和资源报告，不自动扩展到 16 代，也不自动
开始 development arena 或产品门禁。

### 19.44 Phase 4F Recovery Target 4 完成

正式 recovery-v2 已完成 generation 0-3、step 384，共 `512` 局、`24,866 plies`。四代的
replay/cache 均深验通过，input batch 最大耗时不超过 `0.836s`。Mean loss 从 `2.186555`
降至 `1.923746`；value MSE 从 `0.927367` 降至 `0.581617`，value sign accuracy 从
`67.41%` 升至 `78.84%`，high-confidence wrong rate 从 `19.32%` 降至 `7.07%`。

Generation 3 checkpoint/candidate/model-state SHA-256 分别为
`5b87052725ee354693ee6ad34496b4be8ffec7b81ae5ffaa0de534a75240da62`、
`dadf3628927006f1a5867cc9b23e0b0ec1be9416e297f735bd1b2043a2d1e806`、
`ed6f6247fdf0cf3b259b2fccc4179b54a813f46602c8d03241e7e8a297ba2021`；安全加载和四份 generation
report 自哈希均通过，无 checkpoint fallback。

本分块使用 `2.187431130028692 GPU-hours`，global ledger 为
`46.15512160139805 / 72 GPU-hours`，recovery stage 尚余 `11.812568869971308 GPU-hours`。
下一阶段先单独确认 generation-3 health evidence 与 64 局 `recovery-dev-v1` classic-hard
screen preparation/execution；训练内指标不能替代对局棋力证据。

### 19.45 Phase 4F7 Development Screen 准备完成

Generation 3 的 CPU health evidence 已完成并绑定四代 canonical generation report、空 checkpoint
fallback、deep replay、input gate、checkpoint/candidate/model-state 哈希。Health 自哈希为
`29374e5fde933e07a8d7cc60a0ca8cb1b874a81d339b53618838ec12df552b52`。

64 局开发 screen 合同自哈希为
`13d103144c5703457b2b6c28ba72eacc423ce7d11117be31f7c9115e47b83c8b`，冻结 32 对
`recovery-dev-v1` 开局、candidate 800 simulations、classic-hard 587 nodes、32 actors。评测执行器
放在 source fingerprint 排除的 `training/runs` 下并由合同绑定自身文件哈希，因此 recovery-v2
源码指纹仍为 `ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`，不会因准备 screen
而丢失续训能力。

CPU preparation 未访问 CUDA，global ledger 保持 `46.15512160139805 / 72 GPU-hours`。Native
3/3、完整 Python 测试通过。用户下一步运行冻结执行器；结果只决定恢复不足、继续训练或进入
confirmation 候选区间，不自动开始 400 局或正式发布。

### 19.46 Phase 4F7 Development Screen 完成

Generation 3 对 classic-hard 的冻结 screen 完成 64 局，结果为 25 胜 39 负、无 draw，胜率
`39.0625%`，90% CI `[28.125%, 50.0%]`，Elo `-77.25`。Arena result 和 gate 自哈希分别为
`a98e5dc0d047ebc4932a381f463eff198a4bcc72340b0c8e875e055b82085102` 与
`ca509a14405679f282846672e11b2fe93358f28bbd04697a7cac3f523898cc31`，逐步重放和运行时验签通过。

结果按运行前冻结的决策规则落入 `continue-training`，不满足 50% 的 confirmation 候选边界。
Global GPU ledger 为 `46.76848465306063 / 72 hours`。下一阶段应先 CPU-only 归档 generation 3
证据，再保持源码和训练数据隔离，将 recovery-v2 续到绝对 target 8；不得运行 400 局
confirmation 或使用 development screen 轨迹训练。

### 19.47 Phase 4F8 Target-8 Continuation 准备完成

Generation 3 的 checkpoint、candidate、metadata、run manifest 和 generation report 已在续训裁剪前
完整归档，归档清单自哈希为
`df8806081a71faab7ae52ad27a119eb7703d2577f8d25cfb7fd8eb5b39d7fa87`。Target-8 续训合同自哈希为
`381d1d3004f8f75e27e247eb900ee8f220c6f36667ba5217ef9ced36e8f17fbb`。

续训保持同一 recovery-v2 source/config/run lineage，只新增 generation 4-7。训练配置不包含
development screen 路径，screen 只提供聚合决策信号，逐步轨迹不进入 replay。准备前后 global GPU
ledger 均为 `46.76848465306063 / 72 hours`，源码 SHA-256 保持
`ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa`。Native 3/3、完整 Python
415/415 通过。下一动作由用户运行可恢复 target-8 命令，完成后再单独确认 generation 7 screen。

## 20. 主要风险

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| 多语言规则不一致 | 训练数据失效 | golden fixtures 和逐状态差分测试 |
| 墙动作分支过大 | simulation 利用率低 | 强 policy、批量推理、完整但延迟展开 |
| 棋子循环 | 自我对弈不终止 | 统一重复和最大回合规则 |
| 训练算力不足 | 棋力提升缓慢 | 小模型验证、原生引擎、分阶段扩容 |
| 浏览器推理过慢 | 用户体验差 | 蒸馏、量化、WebGPU/WASM 和 fallback |
| 弱教师偏差 | 策略上限受限 | warm start 后继续完整自我对弈 |
| checkpoint 回退 | 发布棋力下降 | champion arena 和统计发布门槛 |

## 21. 参考资料

- [AlphaGo Zero, Nature 2017](https://www.nature.com/articles/nature24270)
- [AlphaZero, Science 2018](https://doi.org/10.1126/science.aar6404)
- [AlphaZero 官方预印本](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [AlphaZero 官方补充材料](https://science.sciencemag.org/content/sci/suppl/2018/12/05/362.6419.1140.DC1/aar6404-Silver-SM.pdf)
- [当前经典 AI 策略](./AI_STRATEGY.md)
