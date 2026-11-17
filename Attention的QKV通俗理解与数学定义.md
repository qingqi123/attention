# Attention 里 Q、K、V 通俗理解 + 数学定义

**核心一句话：**

> **Q（Query 查询）：我要找什么；K（Key 键）：我有什么；V（Value 值）：我实际要取出的内容。**
>
> 自注意力（Self-Attention）里，**Q/K/V 都来自同一批输入序列**；交叉注意力（Cross-Attention）Q 来自一个序列，K/V 来自另一个。

---

## 目录

1. [符号与来源](#1-符号与来源)
2. [计算流程（Scaled Dot-Product Attention）](#2-计算流程scaled-dot-product-attentiontransformer-原版)
3. [自注意力 vs 交叉注意力](#3-自注意力-vs-交叉注意力)
4. [多头注意力 Multi-Head Attention](#4-多头注意力-multi-head-attention)
5. [注意力变体：MHA / MQA / GQA](#5-注意力变体mha--mqa--gqa现代-llm-的主流)
6. [推理时的 QKV：Prefill 与 Decode + KV Cache](#6-推理时的-qkvprefill-与-decode--kv-cache)
7. [位置信息：为什么 Q/K 要吃 RoPE](#7-位置信息为什么-qk-要吃-rope)
8. [复杂度与实现](#8-复杂度与实现)
9. [容易踩坑的几个误区](#9-容易踩坑的几个误区)
10. [极简例子（token 粒度）](#10-极简例子token-粒度)
11. [直观总结表](#11-直观总结表)

---

## 1. 符号与来源

输入：

$$
\boldsymbol{X} \in \mathbb{R}^{n \times d_{model}}
$$

其中 $n$ 是 token 数量，$d_{model}$ 是隐藏维度。

三个可学习权重矩阵：

$$
W_Q,\ W_K,\ W_V \in \mathbb{R}^{d_{model} \times d_k}
$$

其中 $d_k$ 是 Q/K/V 的维度；多头里会切分，一般 $d_k = d_{model} / h$（$h$ 为头数）。

$$
\boldsymbol{Q} = X W_Q, \qquad
\boldsymbol{K} = X W_K, \qquad
\boldsymbol{V} = X W_V
$$

> ✅ **重点：$W_Q, W_K, W_V$ 是模型学出来的，不是固定规则**，作用是把原始输入投影到三个不同的"语义空间"。
>
> 注意：Q 和 K 必须同维（要点积），**V 的维度可以不同**（$d_v$）；最后经输出投影 $W_O$ 映射回 $d_{model}$。

### 1.1 带 batch / 多头时的张量形状（速查）

实际实现里几乎都是 4 维张量：

| 阶段 | 形状 | 说明 |
|---|---|---|
| 输入 $X$ | `[B, N, d_model]` | B=batch，N=序列长度 |
| 投影后 $Q,K,V$ | `[B, N, d_model]` | 一次投影出三者，再按头切分 |
| 按头切分后 | `[B, H, N, d_k]` | H=头数，$d_k = d_{model}/H$ |
| 打分 $QK^\top$ | `[B, H, N, N]` | **每个 batch/头一张 N×N 的注意力矩阵** |
| softmax 后（权重） | `[B, H, N, N]` | 每个 query 行和为 1 |
| 乘 $V$ 后 | `[B, H, N, d_v]` | |
| 拼头 + $W_O$ | `[B, N, d_model]` | **输出和输入同形状**，才能接残差 |

> 最后一条很重要：注意力层的输入输出形状完全一致，所以 Transformer 里才能直接做残差相加。

### 1.2 N（序列长度）到底是什么

**N = 这一批输入里 token 的个数**，是张量里"按行数"的那个维度。

- 模型不直接吃文字，先把文本切成 **token**（词/子词/字），每个 token 用一个向量表示。
  N 就是**一条序列有多少个 token**。例：`"今天天气真好"` 切成 6 个 token → N = 6。

**N vs B vs d_model（三个最容易混的维度）：**

| 维度 | 含义 | 在 `[B, N, d_model]` 里的位置 |
|---|---|---|
| **B**（batch） | 一次并行处理几条序列 | 最外层：有几条 |
| **N**（序列长度） | 每条序列有多少个 token | 中间：每条多长 |
| **d_model**（隐藏维度） | 每个 token 用多少维向量表示 | 最内层：每个多宽 |

读作：**B 条序列，每条 N 个 token，每个 token 是 d_model 维向量。**

> ⚠️ 别把"长度"和"维度"搞混：**N 是"有几个"，d_model / d_k 是"每个多宽"**。N 和 d_k 完全是两回事。

**为什么注意力一定出现 N×N：** 注意力是"两两比较"——Q 的每一行（一个 token）去和 K 的每一列（一个 token）打分，
**N 个 query × N 个 key = 一张 N×N 的注意力矩阵**（每个 batch、每个头各一张，即形状 `[B,H,N,N]`）。
序列越长，这张矩阵按**平方**增长（$O(N^2)$），长上下文贵就贵在这里；causal mask（屏蔽右上三角）也是在这张矩阵上做的。

**不同阶段 N 不一样（很关键）：**

| 场景 | Q 侧的 N | K/V 侧的 N |
|---|---|---|
| **Prefill**（处理整段 prompt） | N = prompt 的 token 数（如 2000） | 同左，N = prompt 长度 |
| **Decode**（逐 token 生成，每步 1 个新 token） | **N = 1**（只有新 token 的 Q） | N = 已缓存的历史长度（不断变长，即 KV Cache 长度） |

所以 decode 时是 `1 × N_cache` 的打分：只算"新 token 和所有历史 token"的关系；
历史 token 的 K/V 从缓存里取，不重算（见 §6 KV Cache）。

## 2. 计算流程（Scaled Dot-Product Attention，Transformer 原版）

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

分步拆开：

1. **$QK^\top$：相似度打分**

   Q 的一行 = 一个 token 的查询向量；$K^\top$ 的一列 = 一个 token 的 key 向量。矩阵相乘结果 $\in \mathbb{R}^{n \times n}$，**每个位置 $(i,j)$ 代表：第 $i$ 个 token（Q 侧）和第 $j$ 个 token（K 侧）有多相关**。

   > 比如句子：`猫 抓 老鼠`
   > Q 是【老鼠】，去和所有 K 算相似度，发现和【抓】【猫】分数更高。

2. **除以 $\sqrt{d_k}$ 缩放**

   维度 $d_k$ 越大，点积数值方差会越大（约正比于 $d_k$），softmax 输出会变得极度尖锐（大部分接近 0，一个接近 1），梯度不好训。缩放是为了把方差拉回 1 附近。

3. **softmax：转成权重（注意力权重）**

   每一行求和 = 1，变成概率分布：这个 token 应该把多少注意力分给其他 token。
   **每一行相互独立** —— 这就是为什么可以按行并行算。

4. **乘 V：加权求和拿结果**

   用上面得到的权重，对 Value 向量加权相加，得到这个 token 融合了其他相关 token 信息的输出。

> **类比数据库检索（非常经典的类比）：**
>
> - 数据库里每条记录：`Key(索引键) + Value(真实内容)`
> - Query：你输入的检索词
> - 先算 Query 和各个 Key 的匹配度 → 得到权重 → 按权重把 Value 拼起来返回结果

### 2.1 Mask 怎么加（用法要点）

Mask 不是加在 Q/K/V 上，而是**加在 softmax 之前的那张 $N\times N$ 打分矩阵上**：

$$
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

**被屏蔽的位置在 $M$ 里填一个很大的负数**，softmax 之后权重≈0，相当于"看不见"。

| Mask 类型 | 用在哪 | 做法 |
|---|---|---|
| **Causal（因果/前瞻）** | 解码器自注意力 | 上三角（$j > i$）填负无穷，禁止看未来 token |
| **Padding mask** | 变长 batch | 把 padding 位置的 key 屏蔽掉（按 key 列屏蔽） |
| 无 mask | 编码器自注意力、解码器交叉注意力 | $M=0$；交叉注意力可以看全部编码器位置 |

> 实操坑：**不要真的填 `-inf`**，容易出现 `-inf + inf = NaN`。常见做法填 `-1e9`（fp32）或 `-65504`（fp16 能表示的最小值），或直接用 `torch.finfo(dtype).min`。
>
> 也有实现把 mask 做成"加性 bias 矩阵"（$M$ 直接是打分偏移量），ALiBi 就是这种形式。

## 3. 自注意力 vs 交叉注意力

| 类型 | Q 来自 | K / V 来自 | 是否有 causal mask |
|---|---|---|---|
| **编码器自注意力** | 当前层输入 | 同一序列 | 否（双向可见） |
| **解码器自注意力（masked）** | 当前层输入 | 同一序列 | **是**（不能看未来） |
| **解码器交叉注意力** | 解码器当前层 | **编码器输出** | 否（可看全部源序列） |

- **Self-Attention（自注意力）**

  $Q, K, V$ 都来自同一个序列 $X$。每个 token 在序列内部，和序列里其他 token 做交互。
  例：理解一句话内部词语之间的关系。

- **Cross-Attention（交叉注意力，解码器中间那层）**

  Q 来自解码器当前输出序列，$K, V$ 来自编码器输出。
  意思：**当前正在生成的词（Q），去查询原文编码信息（K/V）**。
  机器翻译：正在生成中文词（Q），去看英文源句（K/V）。

## 4. 多头注意力 Multi-Head Attention

不是只用一组 QKV，而是：把 $Q, K, V$ 在特征维度切分成 $h$ 份，每一份单独做一组 scaled dot-product attention，得到 $h$ 组输出，再拼接起来，最后过一个输出投影矩阵 $W_O$。

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(Q W_{Q_i},\ K W_{K_i},\ V W_{V_i}) \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
\end{aligned}
$$

> 👉 多头的意义：**不同头学习不同类型的关联**——有的头学主谓关系，有的头学指代、长距离依赖。
>
> 实现上通常**不是**做 h 次小矩阵乘，而是一次大投影出 `[B,N,H*d_k]` 再 reshape 成 `[B,H,N,d_k]`，等价但快得多。

## 5. 注意力变体：MHA / MQA / GQA（现代 LLM 的主流）

区别只在 **K/V 头数和 Q 头数的关系**——这直接决定推理时 KV Cache 的大小，是工程上最关心的点。

| 变体 | Q 头数 : K/V 头数 | 举例 | KV Cache 大小 |
|---|---|---|---|
| **MHA**（多头） | $H : H$（一一对应） | 原始 Transformer、GPT-2 | 最大 |
| **MQA**（多查询） | $H : 1$（所有人共享一个 K/V） | PaLM、Falcon | 最小 |
| **GQA**（分组查询） | $H : G$（每 $H/G$ 个 Q 头共享一组 K/V） | Llama-2/3、Qwen、多数新模型 | 居中 |

例：Llama-2-70B 是 64 个 Q 头、8 个 KV 头（8 组）。

> 为什么都转向 GQA：**解码时 K/V 要缓存并反复读**，K/V 头越少，显存带宽压力越小，长上下文/大 batch 的吞吐越高，而精度损失很小。
>
> 👉 GQA 之上的两条路线（**MLA** 低秩压缩、**KDA** 线性注意力）见 `attention_variants.md`（该文件未随本文件提供）。

## 6. 推理时的 QKV：Prefill 与 Decode + KV Cache

这是"用法"里最关键的一节——训练时一次算完整序列，**推理时是分两步的**：

| 阶段 | 输入 | Q/K/V | 特点 |
|---|---|---|---|
| **Prefill（预填充）** | 整段 prompt（N 个 token） | 三者都全量计算 | 大矩阵乘，**算力受限**，可并行 |
| **Decode（逐 token 生成）** | 每次新增 1 个 token | **只算新 token 的 Q**；K/V 追加进缓存 | 单 token，**显存带宽受限** |

**KV Cache 的原理**：已经算过的 token 的 K、V 不会变，所以缓存起来，后面每个新 token 只需：

$$
q_{\text{new}} K_{\text{cache}}^\top \;\rightarrow\; \text{softmax} \;\rightarrow\; \cdot V_{\text{cache}}
$$

而不再重算整个前缀。这是自回归推理能实用的前提。

**缓存大小估算：**

$$
\text{bytes} \approx 2 \times L \times H_{kv} \times d_k \times N \times B \times \text{dtype\_size}
$$

（2 = K 和 V；L = 层数；$H_{kv}$ = KV 头数；N = 上下文长度；B = batch）

> 这也是 GQA/MQA 的价值所在：$H_{kv}$ 从 $H$ 降到 $G$ 或 1，缓存直接按比例缩小。
>
> 工程上还会做 **PagedAttention**（把 KV Cache 按页管理，见 vLLM）来解决碎片化和变长序列共享。

## 7. 位置信息：为什么 Q/K 要吃 RoPE

注意力本身**没有顺序概念**——打乱 token 顺序，$QK^\top$ 的结果集合不变（置换等价）。所以要注入位置信息。

现代 LLM 普遍用 **RoPE（旋转位置编码）**：

- **只作用在 Q 和 K 上**（V 不旋转）；
- 把 Q/K 的维度两两成对做旋转，旋转角与位置成正比；
- 好处：旋转后 $q_i^\top k_j$ 只依赖**相对位置** $i-j$，天然具备相对位置性质，且便于外推。

（另一种是给打分矩阵加位置 bias，如 ALiBi；早期 Transformer 用绝对位置编码加到输入 $X$ 上。）

## 8. 复杂度与实现

- **计算量** $O(N^2 d)$，**中间矩阵** $N\times N$ —— 序列越长，注意力越贵，这是长上下文的根本瓶颈。
- **FlashAttention**：不把 $N\times N$ 矩阵写进显存，而是分块（tiling）计算 + 在线 softmax（online softmax，边算边归一），显存从 $O(N^2)$ 降到 $O(N)$，且访存更省 → 明显更快。数学结果与标准注意力等价。
- **算子融合**：实际推理引擎里，$QK^\top$ / scale / mask / softmax / $\cdot V$ 常被融合成一个 kernel，避免中间张量往返（residual 相加、LayerNorm 也常一起融合）。

## 9. 容易踩坑的几个误区

| # | ❌ 常见误解 | ✅ 正确理解 |
|---|---|---|
| 1 | "Q = 查 K"，以为最终输出来自 K | K 只用来打分，**K 本身不会出现在最终输出里，输出只来自 V** |
| 2 | Q/K/V 向量维度必须全部相等 | **Q 与 K 必须同维**（才能点积）；V 可以不同维。Transformer 默认三者同维 |
| 3 | Mask 是 QKV 的一部分 | Mask 不是 QKV 一部分，是**加在 $QK^\top/\sqrt{d_k}$ 之后、softmax 之前**，屏蔽不许看的位置 |
| 4 | $W_Q / W_K / W_V$ 是共享的 | **不共享，三个独立矩阵**；多头内部每个头的 $W_{Q_i}/W_{K_i}/W_{V_i}$ 也各自独立 |
| 5 | 注意力权重就是"解释" | 权重是 softmax 后的相对分配，**不等于因果重要性**，不能直接当模型解释用 |
| 6 | 推理时每步都要重算整条序列 | 有 **KV Cache**，decode 每步只算新 token 的 Q，K/V 复用缓存 |

**数值实现上的坑：**

- softmax 前要**减行最大值**，否则 fp16 下极易溢出成 `inf/NaN`；
- mask 填充用 `-1e9` / `dtype.min`，别真填 `-inf`；
- bf16/fp16 下 $QK^\top$ 累加易失精度，很多实现用 fp32 累加；
- scale 用 $\sqrt{d_k}$ 是默认，某些模型另加 temperature/logit 上限（如 soft-capping）。

## 10. 极简例子（token 粒度）

假设只有两个 token：A、B

- $Q_A$：A 想找哪些 token 和自己有关
- $Q_B$：B 想找哪些 token 和自己有关
- $K_A, K_B$：A、B 对外展示的"特征标签"，用来匹配查询
- $V_A, V_B$：A、B 携带的真实信息内容

计算：

$$
\text{A 的注意力权重} = \text{softmax}\big([Q_A \cdot K_A,\ Q_A \cdot K_B]\big)
$$

$$
\text{A 输出} = w_1 V_A + w_2 V_B
$$

B 同理。

## 11. 直观总结表

| 符号 | 名字 | 角色 |
| --- | --- | --- |
| Q | Query 查询 | 我现在要关注什么、我的诉求 |
| K | Key 键 | 候选对象用来匹配查询的标识 |
| V | Value 值 | 候选对象承载的信息，最终被加权融合 |

**一句话串起来：** Q 拿去找每个 K 打分 → softmax 归一成权重 → 用权重把 V 加权求和。
K 只负责"匹配"，**信息只从 V 流出**；Q/K 上还要吃位置编码（如 RoPE），
而推理时 K/V 会被缓存复用（KV Cache），头数怎么共享（MHA/MQA/GQA）决定了缓存多大。

> （注：部分内容由豆包 AI 生成）