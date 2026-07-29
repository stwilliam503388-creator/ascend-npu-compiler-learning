# 参考资源

按学习阶段分层的 Triton + Ascend NPU 外部项目导航。

## 分层学习路径

```
Tier 1: 入门学习  →  Tier 2: 实战参考  →  Tier 3: 编译器深入  →  Tier 4: 前沿与部署
  (跑通 kernel)        (看源码学写法)         (理解底层)             (生产级应用)
```

---

## Tier 1：入门学习

| 项目 | 链接 | 说明 |
|------|------|------|
| TritonAcademy | https://github.com/MekkCyber/TritonAcademy | 最佳 Triton 入门教程，从 vector_add 到 flash attention |
| triton-resources | https://github.com/rkinas/triton-resources | Triton 生态全景导航（论文/教程/工具索引） |
| Triton 官方 Tutorials | https://triton-lang.org/main/getting-started/tutorials/ | vector_add → matmul → fused attention |
| MLIR Toy Tutorial | https://mlir.llvm.org/docs/Tutorials/Toy/ | 官方 MLIR 入门（Ch1-7） |

**建议顺序：** TritonAcademy 跑通 3 个 kernel → 本项目 Phase 4 lab (8 kernel) → triton-resources 找进阶

---

## Tier 2：实战参考

| 项目 | 平台 | 说明 |
|------|:--:|------|
| [FlagGems](https://github.com/flagos-ai/FlagGems) | GitHub ⭐1.1k | 150+ 个真实 Triton 算子，LLM 全栈覆盖，最佳代码参考 |
| [triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels) | Gitcode | 华为官方 Ascend Triton 算子，生产级 Ascend 适配写法 |
| [FlagAttention](https://github.com/flagos-ai/FlagAttention) | GitHub ⭐301 | 高效注意力算子合集（Flash/MLA/GQA），学内存优化 |
| [Awesome-Triton-Kernels](https://github.com/zinccat/Awesome-Triton-Kernels) | GitHub ⭐199 | 社区 Triton kernel 合集 |

**建议顺序：** FlagGems 搜一个感兴趣算子 → 对照本项目 kernel 看差异 → triton-ascend-kernels 看 Ascend 适配 → FlagAttention 学进阶

---

## Tier 3：编译器深入

| 项目 | 平台 | 说明 |
|------|:--:|------|
| [triton-lang/triton](https://github.com/triton-lang/triton) | GitHub ⭐19.8k | Triton 编译器源码（TTIR→TritonGPU→LLVM） |
| [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) | GitHub ⭐119 | Triton→Ascend 后端（ascend_interpreter.py、CANN 对接） |
| [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) | Gitcode ⭐486 | 昇腾 MLIR 编译器：hivm/hacc dialect、Lowering pass |
| [FlagTree](https://github.com/flagos-ai/FlagTree) | GitHub ⭐300 | Triton fork，支持多芯片后端 |

**建议顺序：** triton-ascend 看 `ascend_interpreter.py` → AscendNPU-IR 看 `ConvertLinalgToHivm` → FlagTree 看多后端架构

---

## Tier 4：前沿与部署

| 项目 | 平台 | 说明 |
|------|:--:|------|
| [SageAttention](https://github.com/thu-ml/SageAttention) | GitHub ⭐3.5k | 顶会量化注意力（ICLR/NeurIPS） |
| [ninetoothed](https://github.com/InfiniTensor/ninetoothed) | GitHub ⭐332 | Triton 上层 DSL |
| [Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend) | Gitcode | Triton 分布式通信（对标 `tl.dist`） |
| [libtriton_jit](https://github.com/flagos-ai/libtriton_jit) | GitHub ⭐37 | C++ Triton JIT，减少 Python 开销 |
| [FlagScale](https://github.com/flagos-ai/FlagScale) | GitHub ⭐526 | 大模型分布式训练框架 |

---

## FlagOS 社区全览

[FlagOS](https://github.com/flagos-ai)（49 个仓库）是目前 Triton 生态最大、最完整的开源社区：

| 组件 | 说明 | Tier |
|------|------|:--:|
| FlagGems | 150+ Triton 算子库 | 2 |
| FlagTree | 统一多芯片编译器 | 3 |
| FlagScale | 分布式训练/推理框架 | 4 |
| FlagAttention | 高效注意力算子 | 2 |
| FlagCX | 统一通信库 | 4 |
| FlagDNN/BLAS/FFT/Sparse | 领域专用计算库 | 2 |
| KernelGen | AI 辅助算子开发 | 4 |

## 官方文档

| 资源 | 链接 |
|------|------|
| LLVM 文档 | https://llvm.org/docs/ |
| MLIR 文档 | https://mlir.llvm.org/docs/ |
| Triton 文档 | https://triton-lang.org/ |
| 昇腾社区 | https://www.hiascend.com/ |
| AscendC 文档 | https://www.hiascend.com/document?tag=ascendc |

## 推荐阅读

| 书籍/课程 | 说明 |
|-----------|------|
| *Engineering a Compiler* (Cooper & Torczon) | 编译器工程经典教材 |
| *Getting Started with LLVM Core Libraries* | LLVM 入门实战 |
| CMU 15-745: Optimizing Compilers | 编译器优化课程（进阶） |

## 本项目内部参考

- [LLVM 环境搭建](../docs/llvm/00-环境搭建.md)
- [术语表](../docs/glossary.md)
- [Phase 4 Triton-Ascend 教程](../docs/triton-ascend/README.md)
