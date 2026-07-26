# 00 — 为什么学 Triton-Ascend 编程

> 目标：理解从"看懂了"到"能写"的这一步为什么重要
> 前置：[Phase 3 MLIR](../mlir/README.md)
> 预估时间：10 分钟

## 你现在的位置

```
Phase 3 之后：
  能看懂 linalg.generic → 能读 hivm.hir.vadd → 知道 Triton 编译链路
                          ↓
                      但你能写吗？
```

Phase 3 教会你的是**编译器后端的工作方式**——Pass 怎么写，Dialect 怎么定义，IR 怎么 Lowering。这些都是编译器开发者的视角。

Phase 4 换个角度：**用户视角**。你是一个 AI 工程师，想在 Ascend NPU 上跑你的算子。你不需要写 Pass，你需要写 **Triton kernel**。

## 为什么必须手写一遍？

| | 只读分析文档 | 手写并跑通 |
|---|---|---|
| 理解 block_size | "哦，512 是个数" | "512 是因为 L1 1MB，我的数据类型是 float16，最大 tile = 1MB / 2B = 512K 元素" |
| 理解 gm/ub | "load 和 store 搬运数据" | "我这行 load 如果不 mask 边界，会 segfault" |
| 理解融合 | "两个 op 合并减少搬运" | "fused 版本跑 TTIR，load 次数从 6 次变 3 次" |
| 理解 SIMT/SIMD | "双模执行" | "matmul 走 Cube，bias_add 走 Vector，一个 kernel 里两个模式" |

## 从 30 个任务书看真实需求

你之前整理的 triton-ascend 任务书汇总（位于 `../../../Triton-Ascend-Project/triton-ascend-kernels/triton-ascend-任务书汇总.md`）里，每个任务都是：

```
[极客营]：xxx 算子开发/迁移/优化 任务书
```

三段式结构：
1. **开发**：从零实现算子（写 Triton kernel）
2. **迁移**：GPU Triton → Ascend NPU（理解硬件差异）
3. **优化**：msprof 分析 + autotune（性能调优）

写完 Phase 4，你就能理解这些任务书在要求什么，甚至能自己做一个。

## Phase 4 和后续 Phase 5 的关系

```
Phase 4 (本阶段)              Phase 5 (下一阶段)
  "我写了个 softmax"      →    "softmax 在 Ascend 上怎么 Lowering？"
  "为什么 tile 要 128"    →    "Da Vinci L1 到底多大？怎么查？"
  "融合后 load 少了"      →    "husion 融合 Pass 怎么写？"
```

Phase 4 提出"为什么"，Phase 5 回答"因为硬件/IR 设计如此"。

## 你需要什么环境？

**不需要 Ascend 硬件。** macOS 上安装 `triton` 即可 dump TTIR：

```bash
pip install triton
```

所有 kernel 在 CPU 上执行（或 --device cpu），dump 出 TTIR 验证即可。

## 验证

- [ ] 能说出 Phase 3（看懂）和 Phase 4（能写）的区别
- [ ] 知道 30 个任务书的三段式结构（开发/迁移/优化）
- [ ] 知道学完 Phase 4 能做什么

> 📖 [术语表](../glossary.md)
> **下一步**：[01 — Triton 编程模型与 Ascend NPU](./01-Triton编程模型与Ascend NPU.md)
