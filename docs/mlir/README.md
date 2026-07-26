# Phase 3 — MLIR 学习

> 衔接 Phase 2（LLVM），用已学会的 LLVM 概念理解 MLIR
> 前置：[Phase 2 LLVM](../llvm/README.md)

## 学习路径

```
① LLVM→MLIR 概念对照 ──→ ② Toy Tutorial 带读 ──→ ③ ascendnpu-ir 上手
      (15 min)                (60 min)                  (30 min)
                                    ↓
                         ④ MLIR-L 实战文档（9 篇选读）
```

## 文档列表

### 桥接文档（必读）

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | [从 LLVM 到 MLIR](./00-从LLVM到MLIR.md) | 概念对照 + Dialect 解剖 | 15 min |
| 01 | [Toy Tutorial 导读](./01-Toy-Tutorial导读.md) | 跑通官方教程 Ch1-3 | 60 min |
| 02 | [ascendnpu-ir 快速上手](./02-ascendnpu-ir快速上手.md) | 项目结构 + 核心 Dialect | 30 min |

### 实战深入（选读）

| # | 文档 | 内容 | 时间 |
|---|------|------|------|
| - | [MLIR-L00](./MLIR-L00-速通与AscendNPU-IR实战.md) | 5 步 roadmap | 30 min |
| - | [MLIR-L01](./MLIR-L01-ToyTutorial速通-Ch1-Ch2.md) | Toy AST → Dialect | 30 min |
| - | [MLIR-L02](./MLIR-L02-ToyTutorial速通-Ch3-Ch6.md) | Pass 优化 + 代码生成 | 40 min |
| - | [MLIR-L03](./MLIR-L03-自定义AscendNPU-IR-Pass实战.md) | 手写 op-counter Pass | 30 min |
| - | [MLIR-L04](./MLIR-L04-Standalone实战总结.md) | Standalone 项目模板 | 15 min |
| - | [MLIR-L05](./MLIR-L05-ToyMini从零实现.md) | 简化版 Toy 实现 | 15 min |
| - | [MLIR-L06](./MLIR-L06-TritonMLIR体系分析.md) | Triton 编译流程 | 20 min |
| - | [MLIR-L07](./MLIR-L07-triton-ascend后端深度分析.md) | Triton→Ascend | 25 min |
| - | [MLIR-L08](./MLIR-L08-ascendnpu-ir-demo可运行流水线.md) | 一键运行脚本 | 10 min |

## 动手项目

| 项目 | 说明 |
|------|------|
| **[mlir-hello](../../projects/mlir-hello/)** | MLIR 版 HelloPass，45 行，一键运行 |

和 Phase 2 hello-pass 的精确定位对照，见项目 README。

## 与 Phase 2 的衔接

| Phase 2 学会的 | Phase 3 怎么用到 |
|---------------|----------------|
| LLVM Pass（`run(Function &F)`） | MLIR Pass（`runOnOperation()`） |
| LLVM IR（`.ll` 文件） | MLIR（`.mlir`），多了一个 Dialect 前缀 |
| hello-pass 项目 | mlir-hello 项目，结构对应 |

## 学完验证

- [ ] 能用 LLVM 概念解释 MLIR 的对应物
- [ ] 能跑通 Toy Tutorial Ch1-3
- [ ] 能说出 ascendnpu-ir 的核心 Dialect 和 Lowering 流程

> **下一步**：[Phase 4 — Triton-Ascend 编程实战](../triton-ascend/README.md)
