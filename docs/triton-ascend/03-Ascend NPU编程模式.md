# 03 — Ascend NPU 编程模式

> 目标：从 30 个任务书中提炼 5 种算子类型，理解 Ascend 上的编程套路
> 前置：[02 — Triton 到 Ascend 全链路实战](./02-Triton到Ascend全链路实战.md)
> 预估时间：25 分钟

## 1. 30 个任务书的 5 种模式

回顾 [triton-ascend-kernels 任务书汇总](https://github.com/triton-lang/triton-ascend/issues)，所有任务归为 5 类：

### 模式一：逐元素操作（Element-wise）

```
vector_add, sigmoid, silu, gelu
```

**特征**：每个输出元素只依赖相同位置的输入。纯 Vector Unit。

```python
@triton.jit
def sigmoid_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 1 / (1 + tl.exp(-x))  # sigmoid
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Ascend 要点**：load(gm)→compute(ub)→store(gm)，一次搬运即可。最简模式。

### 模式二：归约操作（Reduction）

```
softmax, log_softmax, rms_norm, layer_norm
```

**特征**：需要跨元素聚合（max, sum, mean）。Vector Unit + 多阶段。

```python
@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    # 阶段 1: 找最大值（数值稳定性）
    x_max = tl.max(x, axis=0)
    # 阶段 2: exp(x - max)
    x_exp = tl.exp(x - x_max)
    # 阶段 3: 求和再归一化
    x_sum = tl.sum(x_exp, axis=0)
    output = x_exp / x_sum
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Ascend 要点**：归约不是 Cube Unit 的特长（Cube 做矩阵乘法而非向量归约）。Vector Unit 用多阶段归约。当 n_cols 很大时需分 tile 做 partial reduction。

### 模式三：矩阵乘法（Matmul）

```
bmm (batch matmul)
```

**特征**：二维分块 + 累加。Cube Unit 主战场。

```python
@triton.jit  
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # tile 在 M 和 N 维度上偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维度滑动
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)                    # ← Cube Unit

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
```

**Ascend 要点**：
- `tl.dot()` → Cube Unit 16×16 矩阵乘法
- tile 大小受 L1 限制：`BLOCK_M × BLOCK_N × BLOCK_K ≤ 1MB / sizeof(float16)`
- 沿 K 维度的循环是 Tiling 的关键

### 模式四：归一化（Normalization）

```
instance_norm, group_norm, vector_norm
```

**特征**：归约 + 广播 + 逐元素。归约部分用 Vector Unit。

```python
# group_norm 的伪结构
# 1. 按 group 维度分 tile
# 2. 每个 group 内：求 mean, var (归约)
# 3. normalize: (x - mean) / sqrt(var + eps) (逐元素)
# 4. scale + shift (逐元素)
```

**Ascend 要点**：norm 类算子的瓶颈在归约的同步开销。Ascend 上通过 `tl.atomic_add` 或分段归约优化。

### 模式五：融合算子（Fusion）

```
fused_add_mul, fused_layernorm + residual
```

**特征**：多个操作合并为一个 kernel，减少 gm↔ub 搬运。

```
# 不融合：3 次 kernel 启动 + 6 次 gm↔ub 搬运
  load(A) → add → store(tmp)
  load(tmp) → mul → store(tmp2)
  load(tmp2) → relu → store(C)

# 融合：1 次 kernel 启动 + 2 次 gm↔ub 搬运
  load(A) → add → mul → relu → store(C)
```

**Ascend 要点**：融合是 Ascend 编译器的核心优化。husion dialect 专门做这件事。你写的 fused kernel 越多，编译器能帮你做的越多。

---

## 2. 5 种模式的对比

| 模式 | 代表算子 | 主要单元 | 瓶颈 | Ascend 优化 |
|------|---------|---------|------|------------|
| 逐元素 | sigmoid, gelu | Vector Unit | 内存带宽 | 增大 BLOCK_SIZE 隐藏延迟 |
| 归约 | softmax, rms_norm | Vector Unit | 同步开销 | 分 tile 减少同步次数 |
| 矩阵乘 | bmm | Cube Unit | tile 大小 | 最大化利用 L1 Buffer |
| 归一化 | group_norm | Vector Unit | 归约+广播 | 融合 scale/shift |
| 融合 | fused_layernorm | Vector/Cube | — | 减少 kernel 启动和搬运 |

## 3. 30 个任务书与模式对照

| 任务编号 | 任务名称 | 模式 | 类型 |
|---------|---------|:--:|------|
| 1-2 | MLIR Dump Log 脚本 | — | 工具开发 |
| 3-4 | bmm | 矩阵乘 | 开发 / 迁移 / 优化 |
| 5 | triton-ascend 安装 | — | 环境搭建 |
| 6-9 | instance_norm / group_norm | 归一化 | 开发 / 迁移 / 优化 |
| 10-12 | vector_norm | 归约 | 开发 / 迁移 / 优化 |
| 13-15 | silu | 逐元素 | 开发 / 迁移 / 优化 |
| 16-18 | sigmoid | 逐元素 | 开发 / 迁移 / 优化 |
| 19-21 | rms_norm | 归约 | 开发 / 迁移 / 优化 |
| 22-24 | log_softmax | 归约 | 开发 / 迁移 / 优化 |
| 25-27 | gelu | 逐元素 | 开发 / 迁移 / 优化 |

**规律**：每个算子都是"开发→迁移→优化"三段式，对应三个难度级别。

## 4. 后续学习路径

写完 Phase 4 后：

```
你的水平：能写 5 种模式的 Triton kernel，理解 Ascend 硬件约束
    ↓
参考 FlagGems：150+ 个真实 Triton 算子实现 (github.com/FlagOpen/FlagGems)
    ↓
Phase 5: 深入理解 Ascend NPU 编译器的内部（hivm, hacc, Pass）
    ↓
极客营任务书：从 30 个任务中挑一个，完成"开发→迁移→优化"全流程
```

---

## 验证

- [ ] 能说出 5 种算子模式的名称和代表算子
- [ ] 知道逐元素和归约的区别
- [ ] 知道为什么融合减少 gm↔ub 搬运
- [ ] 能在 30 个任务书中定位任意任务的模式

> 📖 [术语表](../glossary.md)
> **下一步**：[动手项目 — triton-ascend-lab](../../projects/triton-ascend-lab/README.md)
