# Phase 4 — Triton-Ascend 编程实战

> 衔接 Phase 3（MLIR），学会写 Triton kernel 并理解它在 Ascend NPU 上的执行
> 前置：[Phase 3 MLIR](../mlir/README.md)

## 学习路径

```
① 为什么学编程  ──→ ② Triton + NPU 概念 ──→ ③ 全链路实战 ──→ ④ 编程模式
    (10 min)           (35 min)                 (40 min)          (25 min)
```

## 文档列表

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | [为什么学 Triton-Ascend 编程](./00-为什么学Triton-Ascend编程.md) | 从"看懂 IR"到"写出 kernel" | 10 min |
| 01 | [Triton 编程模型与 Ascend NPU](./01-Triton编程模型与Ascend NPU.md) | tile/program/grid + gm/ub/cube/vector | 35 min |
| 02 | [Triton 到 Ascend 全链路实战](./02-Triton到Ascend全链路实战.md) | vector_add：Python → TTIR → hivm.hir | 40 min |
| 03 | [Ascend NPU 编程模式](./03-Ascend NPU编程模式.md) | 5 种算子类型 + 30 个任务书导读 | 25 min |

## 动手项目

| 项目 | 说明 |
|------|------|
| **[triton-ascend-lab](../../projects/triton-ascend-lab/)** | 4 个递增 kernel：vector_add → softmax → matmul_tile → fused_layernorm |

每个 kernel 含 Python 源码 + TTIR dump + Ascend NPU 对照解析。

## 与 Phase 3 的衔接

| Phase 3 学会的 | Phase 4 怎么用到 |
|---------------|----------------|
| MLIR-L06 TritonMLIR 体系 | 写出来的 kernel 走的就是这套 TTIR→TritonGPU 管线 |
| MLIR-L07 triton-ascend 后端 | ascend_interpreter.py 会处理你的 kernel IR |
| MLIR-L08 ascendnpu-ir-demo | 同样的 .mlir 格式，现在从 Python 侧生成 |

## 学完验证

- [ ] 能写出一个简单的 vector_add Triton kernel
- [ ] 能说出 block/grid/program 三者的关系
- [ ] 知道 gm/ub 地址空间如何影响 kernel 写法
- [ ] 能画出 Triton kernel 到 hivm.hir 的大致路径
- [ ] 知道 Cube Unit 和 Vector Unit 分别处理什么类型的操作

> 📖 [术语表](../glossary.md)
> **下一步**：[Phase 5 — Ascend NPU 后端深入](../ascend/README.md)
