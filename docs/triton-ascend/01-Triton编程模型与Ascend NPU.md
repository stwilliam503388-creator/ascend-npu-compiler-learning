# 01 — Triton 编程模型与 Ascend NPU

> 目标：理解 Triton 的三层编程抽象 + Ascend NPU 的硬件关键概念
> 前置：[00 — 为什么学 Triton-Ascend 编程](./00-为什么学Triton-Ascend编程.md)
> 预估时间：35 分钟

## 1. Triton 编程模型：三层抽象

### 1.1 三层关系

```
Grid (启动配置)
  └─ Program (每个 block 一个 program)
       └─ Block (thread 组，tile 级执行单元)
```

| 层 | 含义 | 代码 | 类比 |
|----|------|------|------|
| Grid | 整体任务分割成多少个 program | `grid=(N_BLOCKS,)` | 把 100 页书分给 10 个人 |
| Program | 每个 program 处理自己的数据块 | `pid = tl.program_id(0)` | 每个人拿到 10 页 |
| Block | 一个 program 内的并行粒度 | `BLOCK_SIZE: tl.constexpr` | 每个人一次处理多少行 |

### 1.2 第一条 Triton kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 这个 program 在 grid 中的位置
    pid = tl.program_id(axis=0)
    # 这个 program 负责的数据范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 边界保护
    mask = offsets < n_elements
    # 加载 → 计算 → 存储
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

**关键点**：
- `@triton.jit` — JIT 编译，不是普通的 Python 函数
- `tl.program_id(0)` — program 在 grid 中的索引
- `tl.arange(0, BLOCK_SIZE)` — 生成 [0, 1, 2, ..., BLOCK_SIZE-1] 张量
- `mask` — 边界保护，必须写
- `tl.load` / `tl.store` — 显式内存操作（不是 `+` 那么自动）

### 1.3 为什么叫 "tile-based"？

```python
# Triton 的 tile 思维：
#   "我把数组切成大小为 BLOCK_SIZE 的 tile，每个 program 处理一个 tile"
#
# 数组: [a0, a1, a2, ..., a_N-1]
#         ├── tile 0 ──┤├── tile 1 ──┤...
#         program 0      program 1
```

这和 GPU/NPU 硬件设计直接对应——硬件就是一批并行单元，每批处理一个 tile。

---

## 2. Ascend NPU 硬件关键概念

### 2.1 内存层次

```
  HBM (Global Memory, "gm")    ← 32GB+, 慢
         ↕ 显式搬运 (load/store)
  L1 Buffer (Unified Buffer, "ub")  ← 1MB, 快
         ↕
  Cube Unit / Vector Unit      ← 计算单元
```

| 地址空间 | 全称 | 容量 | 速度 | 厨房类比 |
|---------|------|------|------|---------|
| `gm` | Global Memory (HBM) | 32GB+ | 慢 (~TB/s) | 冰箱 |
| `ub` | Unified Buffer (L1) | 1MB | 快 (~数十 TB/s) | 案板 |

**规则**：Cube/Vector 只能读 `ub` 不能读 `gm`。所以必须先 `load gm→ub`，算完 `store ub→gm`。

这在 IR 中表现为：
```mlir
hivm.hir.load  ins(%gm_buf) outs(%ub_buf)   // gm → ub
hivm.hir.vadd  ins(%ub_a, %ub_b) outs(%ub_c)  // 在 ub 上计算
hivm.hir.store ins(%ub_c) outs(%gm_out)       // ub → gm
```

### 2.2 计算单元分工

| 单元 | 处理 | 对应操作 | Triton 映射 |
|------|------|---------|------------|
| Cube Unit | 矩阵乘法 16×16 | matmul, conv | `tl.dot()` |
| Vector Unit | 向量运算 | add, mul, relu, softmax 的归约 | `+`, `*`, `tl.sum()` |

**一个 kernel 可以同时用两个单元**——例如 matmul_tile kernel：
```python
acc = tl.dot(a, b)        # → Cube Unit (矩阵乘法)
acc = acc + bias          # → Vector Unit (逐元素加法)
```

### 2.3 SIMT/SIMD 双模执行

Ascend V310/V500 支持两种执行模式：

| 模式 | 适用 | 特点 |
|------|------|------|
| SIMD (Vector) | 规整的向量运算 | 一条指令处理 32/64 个元素 |
| SIMT | 分支多的逻辑 | 类似 GPU warp，处理条件分支 |

编译器自动选择。但你写 kernel 时要意识到：`if mask` 分支在 SIMT 下可能有 warp divergence 开销。

### 2.4 为什么 BLOCK_SIZE 不能随便选？

```
L1 Buffer = 1MB
每个 float16 = 2B
每个 BLOCK 至少需要 3 个 buffer（输入 A + 输入 B + 输出 C）

最大 BLOCK_SIZE = 1MB / (3 × 2B) = 174,762 个元素 ≈ 128² 矩阵
```

这就是为什么 Triton-Ascend kernel 的 `BLOCK_SIZE` 通常在 64~256 之间。

---

## 3. GPU Triton vs Ascend Triton 的关键差异

| | GPU (CUDA) | Ascend NPU |
|---|---|---|
| 内存层次 | Global → Shared → Reg | gm → ub (仅两层) |
| 矩阵单元 | Tensor Core | Cube Unit |
| 地址空间 | 自动管理 | 显式 load/store (gm↔ub) |
| IR 方言 | TritonGPU | HIVM |
| 调试工具 | nsight | msprof |

**最大的思维转变**：GPU 上共享内存是"优化手段"，Ascend 上 gm↔ub 搬运是"必须做的"。每个 kernel 必须显式管理数据搬运。

---

## 验证

- [ ] 能画出 Grid → Program → Block 三层关系
- [ ] 知道 `tl.load` / `tl.store` 的边界保护必须写 `mask`
- [ ] 知道 gm 和 ub 分别是什么，数据如何搬运
- [ ] 知道 Cube Unit 处理矩阵乘法，Vector Unit 处理向量运算
- [ ] 知道为什么 BLOCK_SIZE 受 L1 1MB 限制

> 📖 [术语表](../glossary.md)
> **下一步**：[02 — Triton 到 Ascend 全链路实战](./02-Triton到Ascend全链路实战.md)
