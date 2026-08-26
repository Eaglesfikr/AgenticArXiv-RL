## 1.

## 一句话定位

把一个原本「命令式按顺序执行工具的 ReAct Agent」改造成「**可训练、可验证、可离线回放的强化学习环境（MDP）**」，训练路径 SFT → DPO → GRPO。

核心改造思路是：**「工具执行」这个原本的动作，被重塑成了 RL 三步的骨架 — State / Action / Transition，而奖励由规则（可验证）给出，不需要人工标注。**

## 三层理解框架（按依赖顺序读）

### 第 1 层：RL 环境的三个模块（改造的核心）

| RL 概念                | 代码位置                                                     | 做了什么改造                                                 |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **State（状态）**      | [rl/env.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/env.py) + [agents/base_agent.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/agents/base_agent.py) | 会话状态从数据库落到**内存 store**（`STORE_BACKEND=memory`），状态 = 任务描述 + 对话历史 + 工具结果 |
| **Action（动作）**     | 4 个工具 + FINISH（[tools/](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/tools/)） | 第 4 个工具 `get_paper_cache_status` 是**纯本地读内存**，无网络 |
| **Transition（转移）** | [env.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/env.py) 的 `execute_tool()` | **这是最关键的一步改造**：`BaseAgent._dispatch_tool()` 优先走 `env.execute_tool()` 而不是直接调全局 `registry.execute_tool()`（[base_agent.py:283-286](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/agents/base_agent.py#L283-L286)） |
| **Reward（奖励）**     | [rl/reward.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/reward.py) 的 `RewardCalculator` | 五分量可验证奖励，下面详述                                   |

关键改造点（[env.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/env.py) 开头 docstring 明确列了原实现的 3 个问题）：**拦不住工具调用**（env 不进链路）、**快照永远为空**（落盘被注释）、**key 含 session_id**（缓存永远 miss）。三条全都修掉了，所以才有真正的环境层。

### 第 2 层：MockArxivEnv — 环境如何做到「离线、确定、可复现」

[env.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/env.py) 的 `MockArxivEnv` 是环境的心跳，解决"训练要反复采样、打真实 arXiv API 不可行"的问题。它有 3 种模式：

- **record**：真实调工具，把 `(tool, key) → result` 落盘成快照
- **replay**：只查快照，miss 就报错（完全离线）
- **auto**：先查快照，miss 才真实调用并顺手记录

4 个工具在离线模式下的处理策略**各不相同**（这是理解环境的核心）：

1. `get_recently_submitted_cs_papers` — **网络请求 → 快照回放**，且按 `aspect` 从已记录论文池**派生**（`_derive_search_result`），允许 `max_results=5/7/10` 参数变化都能命中，避免策略改个参数就 KeyError 污染奖励
2. `download_arxiv_pdf` — **网络请求 → 离线桩**（写占位 PDF 文件 + 更新 store，不发 HTTP）
3. `translate_arxiv_pdf` — **子进程 → 拦截在 side-effect manager**，不进 env
4. `get_paper_cache_status` — **纯本地 → 直接真实执行**（读内存 store）

快照 key 构造时剔除易变字段（`session_id`/`output_path`...），保证跨 session 可命中（[env.py:254-261](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/env.py#L254-L261)）。

### 第 3 层：五分量可验证奖励（RLVR 的"可验证"所在）

[rl/reward.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/reward.py) 借鉴 LLM-TIR 的分层奖励，每分量归一化到 `[-1, 1]`，加权求和：

| 分量       | 权重 | 信号                                                         |
| ---------- | ---- | ------------------------------------------------------------ |
| `format`   | 1    | 每步 action 是否合法 JSON 工具调用或终止符                   |
| `tool`     | 3    | 预测工具序列 vs 期望序列的**顺序感知 LCS-F1**（[benchmark/metrics.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/benchmark/metrics.py)） |
| `argument` | 2    | 参数键召回率 × 精确值准确率；无标准答案时整档踢出分母        |
| `process`  | 1    | 合法步加分 − 解析失败/执行失败/多余调用惩罚                  |
| `outcome`  | 3    | 正确完成 +1、路径错误 +0.25、强制停止 −0.5、错误 −1          |

**课程学习**（`schedule()`）：前 30 步把 `tool`/`argument`/`outcome` 权重 ×1/3，先学 ReAct 结构后学语义 —— 否则只看 total 分不出"策略变强"还是"权重放开"。

**任务声明层**（[benchmark/task_spec.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/benchmark/task_spec.py)）解决了奖励区分度的根源：`expected_tools` 和 `expected_tool_args` 由**同一份 steps 派生**，不再手写两份平行列表漂移。

------

## 两条训练/采样路径（理解它们如何复用同一个环境）

### 路径 A：命令行 Rollout（离线数据收集）

[rl/rollout.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/rollout.py) — 加载任务 → `ReActAgent.run()`（注入 env）→ 算 reward → 存 JSONL 轨迹。用于生成 SFT/DPO 数据。

### 路径 B：GRPO 多轮训练（在线，核心）

[rl/train_grpo.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/train_grpo.py) 组装了完整链路：

1. **数据**：`build_prompt_dataset()` 直接从任务集派生（GRPO 在线，不需要预生成轨迹）
2. **多轮 rollout**：[rl/multiturn_env.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/multiturn_env.py) 的 `AgenticArxivMultiTurnEnv` + [rl/grpo_reward.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/grpo_reward.py) 的 `make_multiturn_rollout_func()` — 每轮由当前 policy 生成动作 → 独立 env 执行 → 把 observation 插回上下文（`env_mask=0`，只作上下文不进 loss）→ 直到 FINISH/解析失败/max_turns
3. **奖励**：完整轨迹交给 `RewardCalculator` 打分，与 rollout/benchmark **共用同一套标准**
4. **训练守卫**：`RewardVarianceGuard`（组内奖励方差为 0 → 中止）、`CanaryCallback`（周期评估防退化）、`StageVerifier`（阶段产出最低阈值）

------

## 建议按这个顺序读代码

1. **benchmark/task_spec.py** + [benchmark/tasks.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/benchmark/tasks.py) — 先看任务是怎么声明的（奖励的"标准答案"来源）
2. **benchmark/metrics.py** — 指标/参数/工具序列的比对逻辑
3. **rl/env.py** — 环境 + MockArxivEnv（改造核心）
4. **rl/reward.py** — 五分量奖励
5. **rl/grpo_reward.py** — GRPO 里如何把 TRL 多轮消息折回统一 history
6. **rl/train_grpo.py** — 组装与守卫
7. **rl/rollout.py** — 离线收集

------

## 一个帮你把握整体的"心智模型"

把这个项目理解成三件套：

- **任务**（`TaskSpec`）= 一张"考卷"，带标准答案（`steps` 派生 `expected_tools` + `expected_tool_args`）
- **环境**（`MockArxivEnv`/`MultiTurnEnv`）= 一个"离线考场"，把工具调用变成可回放、确定性的 transition
- **奖励**（`RewardCalculator`）= 一个"阅卷老师"，用纯规则给你打分（不需要人类）——这就是 **RLVR（Reinforcement Learning with Verifiable Reward）**

三者通过 `base_agent._dispatch_tool()` 被串在一起：模型每生成一个动作，环境执行、返回 observation，老师打分，policy 据此更新。

**不需要理解的部分**（你要忽略的）：`archive/`（原 Web 应用）、`mcp_protocol/`、`skill_cli/`、`AgenticArxivWeb/`、`models/db.py` + `services/`（数据库/SSE，RL 路径已切到 memory 后端）。`agents/` 里只需看 `base_agent.py` 的 `_dispatch_tool` 和 `agent_engine.py` 的 ReAct 解析，其余是原系统的产物。

------

要不要我沿着这个顺序，帮你画一张更细的**数据流图**（模型生成 → 环境执行 → 奖励 → 训练更新的具体字段流转），或者单独深入某一个模块（比如 GRPO 多轮 rollout 的 token 掩码那一段）？





## 2

我还是不理解，我之前搞得时Agent开发，现在转向agrntic rl，可以更我简单讲讲这个项目怎么入门吗

好，你用大白话给你讲。你是搞 Agent 开发的，现在转 agentic RL —— 关键就是"**Agent 开发**和 **Agent RL** 到底哪里不一样"。我先讲这句话，再套到项目上。一句话区别**Agent 开发**：你写代码让 Agent 会干活。工具调用、prompt、循环，目标是"**让它能做对**"。**Agent RL**：你不写"怎么干活"的规则，你造一个**环境**让 Agent 自己练。你只管三件事：**给什么任务、工具怎么返回、怎么打分**。模型靠试错自己学会。说白了：**RL 里你不再"教"它，你只负责"出题 + 当裁判 + 提供考场"**。这个项目就是这三件事的完整实现。先别读代码，先只抓这 4 个词所有 RL 都是这四件套，记住它们，整个项目就对号入座：**State（状态）** —— 现在"对话讲到哪了"+ 有哪些工具结果**Action（动作）** —— Agent 这一步要调哪个工具（或说"搞定/结束"）**Transition（转移）** —— 这一步动作执行完，返回什么结果给模型**Reward（奖励）** —— 打分，教会模型"这样做对/不对"RL 的全部意义：模型在 动作→转移→看奖励→再动作 的循环里反复试错，把"高分动作"练强。**奖励是唯一的学习信号，所以它必须可验证、可算出来** —— 这就是为什么这个项目强调"五分量可验证奖励"，因为 RL 没法靠人一步步标注，只能靠规则打分。套到项目上：三件套对应RL 四件套项目里对应你要看的文件**出题（任务）**`TaskSpec`，一份带标准答案的"考卷"`benchmark/task_spec.py`、`tasks.py`**考场（环境）**`MockArxivEnv`，把工具变成**可回放、确定性**的 transition`rl/env.py`**裁判（奖励）**`RewardCalculator`，纯规则给五分量分`rl/reward.py`**训练**GRPO 训练脚本，把以上串起来`rl/train_grpo.py`你要"入门"就按这个顺序看 3 个文件（10 分钟）**第 1 个：rl/env.py** — 看"考场"长什么样。重点看 3 点：3 种模式 `replay/record/auto`（回放/记录/自动）4 个工具被**分成两类**：搜论文是**网络**（用快照回放），缓查询是**纯本地**（直接真跑）为什么关键 —— 离线、可复现，训练才能反复采样不打真实网络**第 2 个：rl/reward.py** — 看"裁判"怎么打分。只记一句：**分 = format×1 + tool×3 + argument×2 + process×1 + outcome×3 的加权平均**每个分量都是 `[-1, 1]`，"可验证"就是这五个都能用规则算出来，不用人看。**第 3 个：rl/train_grpo.py** — 看"训练"怎么跑。你不用读懂代码，只要知道它的流程：取任务 → 让模型生成动作 → 环境执行→返回观察 → 打到 max_turns 或 FINISH → 用裁判打分 → 更新模型看完这三个，再回头补 `benchmark/metrics.py`（怎么比对工具序列/参数）和 `rl/grpo_reward.py`（GRPO 怎么把 TRL 的多轮消息转回统一格式）。你要忽略的（原来的产物，别浪费时间）`archive/`、`AgenticArxivWeb/`、`mcp_protocol/`、`skill_cli/` —— 原 Web 应用的三套 Agent 架构，RL 只留了 ReAct`models/db.py`、`services/`、`api/` —— 数据库 + SSE，**RL 路径已经切到内存 store**（`STORE_BACKEND=memory`），不看`agents/` 里只需要看两个点：`base_agent.py` 的 `_dispatch_tool()`（工具走环境）和 `agent_engine.py` 的 ReAct 解析，其余是原系统的东西。一句话再给你钉住方向：**你是从"写 Agent 逻辑"转过来，现在要切换到"造考场当裁判"的视角。这个项目就是一套现成的"考场+裁判"，你去读它时带着 RL 四件套（题目/状态/动作/奖励）去对号入座就行。**



## 3

可以带我跑一次，还有这里的rollout是？

先回答 "rollout 是什么"，再看你的环境能不能真跑。**Rollout 是什么**：RL 里每个训练步的开头都要——**把任务扔给当前的模型，让它一步步调工具，把完整过程记录下来**。这个小循环叫 rollout。`发任务 → 模型生成「Thought/Action」 → 环境执行工具 → 返回 observation      → 模型再生成下一步 → … → FINISH 或超轮数 → 记下来 + 打分`相当于"**让现在的学生先做一遍试卷，把答题过程录下来**"。录下来的就是 trajectory（轨迹）。GRPO 训练就是反复 rollout → 打分 → 更新模型 → 再 rollout。

我执行的是（在 WSL 里，**不用训练那些重依赖，直接调 LLM API**）：

```
cd AgenticArxiv && ../.venv/bin/python -m rl.rollout search_01 ../traces_test/
```

拆开看每一步：

| 环节              | 发生了什么                                                   |
| ----------------- | ------------------------------------------------------------ |
| **发任务**        | 取 `search_01`：检索最近7天内 cs.AI 论文，最多5篇            |
| **模型生成**      | DeepSeek-V4-Flash 通过 SiliconFlow API 收到这段 ReAct prompt |
| **工具执行/返回** | Agent 调工具拿 observation                                   |
| **打分**          | `RewardCalculator` 按五分量算                                |
| **落盘**          | 轨迹 JSONL 存到 `traces_test/rollout_20260826_160659.jsonl`  |

## 重点看这个输出 —— 这是 RL 的"考试分数"长啥样

```
任务完成  (指 rollout 过程跑完了，不是任务做对了)
Reward: -0.14                      ← 五个分量的加权和
Metrics: task_completed=False,     ← 没正常 FINISH
         tool_call_accurate=False, ← 工具序列不对
         parse_failures=0,
         tool_exec_failures=0
```

**这是最值得你关注的一次机会**：奖励是 **-0.14（负分）**，任务没完成。为什么？因为**当前这个模型还没学过 ReAct 的格式** —— 这正是 GRPO 训练要解决的问题！模型"乱调工具/不会收尾"，奖励就给负分，训练就靠这个信号把它往正确的工具序列、正确的参数、正确的 FINISH 上推。

换句话说：**你刚亲眼看到了"一个还没学会的 Agent 拿到低分"** —— 这就是 RL 学习的起点。之前我讲"奖励是唯一的学习信号"，现在你看到负分就是"老师打了低分，接下来要罚它往对的方向走"。

## 你注意到的关键点

- **没注入 env=**：这次 rollout 走的是**真实网络**（直接调 registry + LLM API），不是快照回放。这正好是验证 LLM 链路通了 —— 确实通了，模型能返回、工具能执行、奖励能算。
- **RLGRP 训练才会用 MockArxivEnv**：训练时为了可复现、不打爆网络，才切到快照回放（`data/mock_arxiv_snapshot.json`，现在还没有，需要先 `build_snapshot`）。



## 4

不是,我还是没理解这里怎么产生的的SFT数据 @scripts/generate_sft_data.py   @AgenticArxiv/benchmark/tasks.py  @AgenticArxiv/benchmark/task_spec.py

明白，你想理解的是 **SFT 数据到底怎么从 benchmark 任务里来的**。这次不跑命令，直接把这 3 个文件怎么串成一条 SFT 样本讲清楚。先看一眼 `task_spec.py` 和 `tasks.py` 里任务的"标准答案"部分我们已经读过了，再看一下 reward 是怎么"挑成功轨迹"的关联点 —— 但这里有个关键点：**generate_sft_data.py 其实没调用 reward，它是"谁跑对了就收谁"**。先纠正一个直觉上的误区，会让你更清楚：**SFT 阶段根本不写奖励，也根本不管"多轮"，它就是"抄作业"。** 我把完整链路给你拆开：一句话：SFT 数据怎么来的**拿 benchmark 里每个任务的描述当考题 → 让一个已经会做这些题的 Agent 实打实跑一遍 → 把每个成功的工具调用 (thought→action) 单独切成一条样本 → 存成 system/user/assistant 三段的 messages。**所以 SFT 数据里**不包含 reward，也不包含 observation（工具结果）**，只有"问题 → 单个动作"。3 个文件各自干什么文件作用类比[task_spec.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/benchmark/task_spec.py)定义"考试题 + 标准答案"（`TaskSpec`、`Step`）题库里的"题目 + 标准答案"[tasks.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/benchmark/tasks.py)具体 8 道题（`search_01`…）+ 从 spec 派生 `expected_tools`试卷[generate_sft_data.py](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/scripts/generate_sft_data.py)**执行**题目、收集成功轨迹、切分样本阅卷 + 抄作业关键：`steps` 不是用来抄的，是用来**验证**的你可能会以为：既然 `tasks.py` 里每道题都有 `steps=(Step("get_recently_submitted_cs_papers", {...}),)` 这个"标准答案"，那 SFT 数据是不是按 `steps` 直接编出来？**不是。** `generate_sft_data.py` 里 `steps` 根本没被用到。它做的是：`result = agent.run(task_def["task"], ...)   # 真跑一遍，让 Agent 决策for step in result["history"]:               # 把每个"动作"单独切出来    if action 是工具调用（不是 FINISH）:        sft_data.append({"messages": [system, user(任务), assistant(单条action)]})``steps`/`expected_tools` 的用处**不在生成 SFT 数据**，而是在 **benchmark 打分 / RL reward 里验证动作对不对**（`metrics.expected_tools`、`argument_match_score`）。这是两件事：`steps` = **标准答案**（用来判对错）SFT 数据 = **Agent 实际跑出来的动作**（用来当"教学示范"）一条 SFT 样本长这样（注意没有 observation、没有 reward）`system:    "你是一个 arXiv 论文检索 Agent，可以调用工具完成任务。"user:      "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇"assistant: "{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI", "days": 7, "max_results": 5}}"`**每一条**只会教模型"**给定这个任务，你应该调这个工具、带这些参数**"—— 一步一个样本，`history` 里几步就切成几条。这正好对应 RL 里说的：SFT 先教"**会用 ReAct 格式**"，reward 那些五分量是 GRPO 才上场的。补一个你可能踩的坑（注意看 54-57 行）`for step in result["history"]:    action = step.get("action", "")    if not action or action == "FINISH":        continue      # ← FINISH 被跳过    sft_data.append(...)  # 只收工具调用`**FINISH 那一步被丢掉了**。也就是说 SFT 只学"怎么调工具"，**不学"怎么收尾"**。所以：SFT 学成后能调工具、但可能不会正确 FINISH你之前在 rollout 看到的 `task_completed=False`、奖励负分，根子就在这 —— 模型没学过完美收尾这正是 DPO/GRPO 后续要补的："什么时候该停止"这种**序列级**的行为，单步 SFT 教不了**一句话总结你现在该记住的：SFT 数据 = 用会做这些题的 Agent 实跑一遍，把每一步"任务→动作"单独抄成一条样本；steps 是标准答案（判对错用），不在生成这步；FINISH 步被丢掉了，所以 SFT 不教收尾。**



## 5

你看下我这个SFT数据生成结果感觉不太对啊，怎么全是error，而且为什么全是error其还是作为样本数据了，不是说只有成功的才作为数据去训练吗

你的直觉是对的，这点确实有问题。让我亲自读你的 `.jsonl`，确认"全是 error"到底指什么 —— 是 `observation` 里的"工具执行失败"，还是 action 本身解析失败，还是我上一条讲的"FINISH 被丢了"让你误解了。先看实际数据。

Read [d:\workspace\AgenticArxiv-RL\data\sft\sft_train.jsonl](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/index.html?id=f4fc19b3-5c7a-4142-825e-4fadbd7dc2e8&parentId=2&origin=534eb72d-44aa-4ee7-b5a4-d16caf0b3db7&swVersion=6&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&session=9fe3b8f9-75dd-41c1-bbd1-9dc74001a7b7#)

你说得对，**这确实有问题** —— 数据里每条都是 `"ERROR"`，这绝对不是"SFT 该有的样子"。这不是你理解错了，是**这条管线在这个环境里真的坏了**。我来把这个根因拆给你看，你会同时明白两件事:一是代码本该怎么做,二是为什么你这里全变成了 ERROR。先回答你最直接的那个疑问**"为什么 ERROR 还是被当成样本存进去了?"** —— 因为脚本的判断条件是宽松的:`if result.get("iteration_count", 0) > 0 and result.get("history"):    # 只要"有几轮"就和"有历史"就收下来    for step in result["history"]:        action = step.get("action", "")        if not action or action == "FINISH":            continue        sft_data.append({...})   # ← 不检查这步到底成没成功`**它只看"每一步 action 是不是非空、非 FINISH",完全不验证这一步到底有没有成功执行、返回是啥、奖励是不是合格的"expert"。** 只要模型吐了一个工具调用文本,哪怕工具根本执行失败(observation 是 ERROR),它也照存。所以"全是 ERROR"就进来了 —— 不是靠奖励筛选的错,是**根本没做筛选**。那为什么每条都变成 ERROR?因为**这个环境里 LLM 调用根本没通**。看脚本里这段:`llm_client = get_env_llm_client()   # 读 .env 的 LLM_BASE_URL/API_KEY/MODELagent = ReActAgent(llm_client, side_effect_mgr=LocalSideEffectManager())result = agent.run(task_def["task"], session_id=...)``config.py` 顶层会 `load_dotenv()` 加载 `.env`,但**它默认读的是"当前工作目录"的 .env**。而你是从**项目根**跑 `python scripts/generate_sft_data.py` 的,此时:`.env` 在 `AgenticArxiv/.env`(不是根目录),不一定被 `load_dotenv()` 找到结果 `LLM_API_KEY` 没读到 → `get_env_llm_client()` 抛 `RuntimeError: Missing env: LLM_API_KEY``agent.run()` 内部调 LLM 失败 → 每个动作的 observation/输出都变成 `ERROR`所以**每条 "ERROR" 其实是"LLM 调用失败了,整个 agent 没跑起来"**,不是"工具真被调了然后失败"。这跟你刚才在 rollout 里看到 `Reward: -0.14` 那次不一样 —— 那次 LLM 通了、是真跑了,只是模型把动作做错了、奖励是负的;这次是**更彻底:LLM 根本没连上,每个动作直接落成 ERROR**。一句话总结根因`你从根目录跑 → .env 没被 load_dotenv 找到 → LLM_API_KEY 缺失 → get_env_llm_client() 抛错    → ReActAgent.run() 每次 LLM 调用都失败 → 每个 action 变成 "ERROR"    → 生成脚本的筛选太宽松(只看 action 非空,不看成功与否)    → 全 ERROR 也被当成"SFT 学习样本"存了进去`**两层问题:① .env 没从根目录加载(环境问题),② 脚本没校验(数据质量问题)。**你可以怎么确认是我说的这样跑一条最小验证,看 LLM 到底连没连上:`# 在 WSL 里,先进到含 .env 的目录再跑,确认能读到大写的那些 envwsl bash -lc "cd /mnt/d/workspace/AgenticArxiv-RL/AgenticArxiv && ../.venv/bin/python -c 'from config import settings; print(settings.antigravity_base_url); print(\"API_KEY:\", \"sk-\" if settings.antigravity_api_key.startswith(\"sk\") else \"NO-KEY\")'"`如果打印 `NO-KEY`,那就坐实了是 `.env` 没加载。如果打到 `sk-...`,那就说明 LLM 是通的,那 ERROR 就更需要查 `agent.run()` 内部到底为什么抛。



## 6.

好的，现在可以了，但是这样的话微调的训练数据原来是这样的啊，那这样微调的目的是什么呢，或者说我系统的目的是什么

问得好 —— 这正好是"看到 SFT 数据很简陋、不禁怀疑它干嘛用"的典型时刻。我从两个层面回答你：**SFT 的目的**，和**整个系统的目的**。

## 第一个问题：SFT 数据为什么长这样，微调目的是什么

你的 SFT 数据 = `system / user(任务) / assistant(单个 action)`，**没有 observation、没有多轮、只有一段 JSON 动作**。

这不是"偷懒"，是这个阶段**刻意只教一件事**：

> **SFT 的目的 = 教会模型"说人话（ReAct 格式 + 合法 JSON 工具调用）"，不是教会它"怎么应对环境反馈"。**

用你做 Agent 开发的经历类比：

- 你之前写 Agent，是靠 **prompt 提醒**模型："请用 `Thought: ...` `Action: {"name":..., "args":...}` 格式输出"。但基座模型天生不会这个格式——它更习惯写自由文本。prompt 说一百遍不如微调一次来得扎实。
- SFT 就是把"任务 → 正确动作"这对映射**直接教进权重里**，让模型**内化**这个响应格式，而不是临时靠提示词约束。
- 每步拆成一条样本（你看到 `generate_sft_data.py` 里 `for step in history` 那样切），叫**行为克隆**：学的是"输入(任务) → 输出(动作)"的单步决策，**不需要环境反馈**。

**最关键的一点，也是你为什么需要它：** GRPO 的奖励里有一个 `format` 分量，专门检查 action 是不是合法 JSON。如果模型连合法动作都吐不出来，那**奖励恒为下限、组内方差恒为 0、训练梯度恒为 0**——就是 `train_grpo.py` 里 `RewardVarianceGuard` 要拦的坑（[train_grpo.py:89-138](vscode-webview://19omhqiuf9167l00400b20sjdp30mhq08gh40g9aer3rmj8p0sg4/AgenticArxiv/rl/train_grpo.py#L89-L138)）。

所以整个链条是环环相扣的：

```
模型不会格式 → GRPO 打分全是地板分 → 零梯度、学不动
        ↑
    SFT 先救这个：教会格式和单步动作，GRPO 才能有梯度可学
```

## 第二个问题：你的系统到底在干什么（全景）

一句话：

> **你的系统 = 把"arXiv 检索/下载/翻译的 Agent"改造成一套"可训练、可验证、不需要人工打分的 RL 训练场",让模型从"会调格式"进化到"会做对决策"。**



## 7

为什么说可以改变模型内部参数啊，我这里不还是调用模型的API吗，调用API好像不可以修改模型吧？

