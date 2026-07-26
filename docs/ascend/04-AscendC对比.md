# 04 — Triton vs AscendC：选择指南

> 目标：理解两种 Ascend NPU 编程路径的区别和适用场景
> 前置：[Phase 4 Triton-Ascend](../triton-ascend/README.md)
> 预估时间：10 分钟

## 两条路径

```
PyTorch 模型
    ├──→ Triton kernel (@triton.jit)  ──→ TTIR → ascendnpu-ir → LLVM
    │         Python DSL, 写一次跑多硬件
    │
    └──→ AscendC (.cpp)              ──→ Bisheng Compiler → 机器码
              C/C++ 扩展, 只有 Ascend NPU
```

## 对比

| | Triton | AscendC |
|---|--------|---------|
| 语言 | Python DSL (`@triton.jit`) | C/C++ (`AscendC` 扩展) |
| 学习成本 | 低（Python 用户友好） | 中（C++ + NPU 硬件知识） |
| 性能上限 | 好（自动优化） | 极致（手动控制指令级） |
| 可移植性 | GPU/Ascend 一套代码 | 仅 Ascend NPU |
| 调试工具 | print + TRITON_DUMP_IR | msprof + Ascend Insight |
| 内存管理 | 隐式（编译器处理 gm↔ub） | 显式（手动 `DataCopy`/`SetVector`） |
| 融合能力 | 编译器自动融合 | 手动设计融合算子 |
| 适合场景 | 快速原型、跨平台迁移 | 极致性能、紧密结合 CANN |

## 什么时候用哪个？

```
需求：快速验证一个算子想法
  → Triton：10 行 Python，3 分钟跑通

需求：把 GPU kernel 迁移到 Ascend
  → Triton：改几行（device check + tile size）

需求：在 Ascend 上压榨最后 20% 性能
  → AscendC：手动控制 Cube/Vector 调度、L1 分配

需求：调用 CANN 特定 API（如 ATC 离线编译）
  → AscendC：Triton 不暴露 CANN 底层接口
```

## 本项目覆盖了哪条？

```
Phase 1-3: 编译器基础（两条路径都需要）
Phase 4: Triton-Ascend 编程（Triton 路径）
Phase 5: Ascend NPU 后端（IR 层，两条路径共用）
```

**本教程以 Triton 为主**，因为它的学习曲线更平缓，学会后迁移到 AscendC 只需要补 C++ 和硬件细节。

## 进一步学习 AscendC

- [昇腾社区 AscendC 文档](https://www.hiascend.com/document?tag=ascendc)
- [AscendC 官方样例](https://gitee.com/ascend/samples/tree/master/Operator/Labs/Samples)

> 📖 [术语表](../glossary.md)
> **下一步**：[05 — msprof 性能分析入门](./05-msprof入门.md)
