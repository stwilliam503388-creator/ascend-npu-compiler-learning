# Ascend NPU Compiler Learning

# 昇腾 NPU 编译器学习

> 从零基础编译器概念，到 LLVM Pass、MLIR Dialect，再到 AscendNPU-IR / Triton 对接的教程与动手项目合集。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM%2FMLIR-18%2B%20recommended-blue)](https://llvm.org)
[![AscendNPU-IR](https://img.shields.io/badge/AscendNPU--IR-reference-blueviolet)](https://github.com/Ascend/AscendNPU-IR)

---

## 项目定位

本仓库是一个 **AI 编译器学习教程项目**，面向想理解 Triton、MLIR、Ascend NPU 编译栈的学习者。项目用中文文档和小型工程串起以下主线：

```text
为什么学 → Primer 零基础 → LLVM IR / Pass → MLIR Dialect / Lowering
        → Triton-Ascend 编程实战 → Ascend NPU 后端深入 → 可运行 demo 与测试用例
```

> 术语说明：本仓库中出现的 **AscendNPU-IR** 与 **BishengIR** 指向同一类 Ascend NPU MLIR 编译器项目语境。AscendNPU-IR 是官方仓库名称，BishengIR 是相关源码中常见的命名空间和工具名前缀。

## 推荐学习路线

| 阶段 | 目标 | 必读入口 | 动手项目 | 验证方式 |
|---|---|---|---|---|
| 0. 为什么学 | 理解学习价值和使用场景 | [docs/why-ascend.md](docs/why-ascend.md) | — | 能说明 AI 编译器后端能解决什么问题 |
| 1. 快速开始 | 2 小时建立完整路线感 | [docs/quickstart.md](docs/quickstart.md) | [projects/hello-pass](projects/hello-pass/) | `bash setup.sh`、`./run.sh` |
| 2. Primer | 零基础理解 AST / IR / Pass / Lowering | [docs/primer/](docs/primer/) | — | 能看懂一个 `linalg.generic` 用例 |
| 3. LLVM | 能读 LLVM IR，能写简单 Pass | [docs/llvm/](docs/llvm/) | [hello-pass](projects/hello-pass/)、[opt-pass](projects/opt-pass/) | 项目 `run.sh` |
| 4. MLIR | 理解 Dialect、Operation、Pattern、Lowering | [docs/mlir/](docs/mlir/) | [mlir-hello](projects/mlir-hello/)、[standalone-mlir](projects/standalone-mlir/) | `mlir-hello/run.sh`、TableGen/CMake 验证 |
| 5. Triton-Ascend | 写 Triton kernel，理解 Ascend NPU 执行 | [docs/triton-ascend/](docs/triton-ascend/) | [triton-ascend-lab](projects/triton-ascend-lab/) | 4 个 kernel 全部跑通 |
| 6. Ascend NPU 后端 | 深入 Ascend 编译器内部 | [docs/ascend/](docs/ascend/)、[docs/ascendnpu-ir/](docs/ascendnpu-ir/) | [ascend-samples](projects/ascend-samples/)、[ascendnpu-ir-op-counter](projects/ascendnpu-ir-op-counter/) | 阅读 input/expected，或使用自建 `bishengir-opt` |
| 7. 综合 demo | 观察 Linalg 到底层 IR 的降级膨胀与优化 | [projects/ascendnpu-ir-demo](projects/ascendnpu-ir-demo/) | 31 个 MLIR 用例 + 28 个 Triton 对照 | `bash run-tests.sh`、`bash run-demo.sh` |

## 项目结构

```text
ascend-npu-compiler-learning/
├── README.md                    # 项目入口和权威学习路线
├── SUMMARY.md                   # 历史交付和阶段总结
├── setup.sh                     # 环境检查
├── docs/                        # 教程文档
│   ├── quickstart.md            # 2 小时快速入门
│   ├── why-ascend.md            # 学习动机
│   ├── primer/                  # 零基础入门
│   ├── llvm/                    # LLVM IR / Pass
│   ├── mlir/                    # MLIR / Dialect / Lowering
│   ├── triton-ascend/           # Triton-Ascend 编程实战
│   ├── ascend/                  # Ascend 后端概念和构建调试
│   ├── ascendnpu-ir/            # AscendNPU-IR 官方文档翻译/分析
│   └── reference/               # 术语速查手册
├── projects/                    # 动手项目
│   ├── hello-pass/              # 第一个 LLVM Pass
│   ├── opt-pass/                # 修改 IR 的 LLVM Pass
│   ├── mlir-hello/              # 第一个 MLIR Pass
│   ├── triton-ascend-lab/       # Phase 4: 4 个递增 Triton kernel
│   ├── toy-mini/                # 纯 C++17 Toy 前端
│   ├── standalone-mlir/         # 自定义 MLIR Dialect 工程模板
│   ├── ascendnpu-ir-op-counter/ # AscendNPU-IR Pass 参考代码
│   ├── ascend-samples/          # Ascend Lowering 精选用例
│   └── ascendnpu-ir-demo/       # 综合 MLIR 降级 demo
├── references/                  # 外部资料索引
├── plans/                       # 历史计划和后续 backlog
└── scripts/                     # 仓库检查和辅助脚本
```

## 快速运行

```bash
# 1. 环境检查
bash setup.sh

# 2. 第一个 LLVM Pass
cd projects/hello-pass
./run.sh

# 3. 第一个 Triton kernel（无需 Ascend 硬件）
cd ../triton-ascend-lab
bash run-all.sh

# 4. MLIR/AscendNPU-IR 综合 demo（无 mlir-opt 时会自动降级为标注检查）
cd ../ascendnpu-ir-demo
bash run-tests.sh
```

如果本机没有 `mlir-opt`，`projects/ascendnpu-ir-demo/run-tests.sh` 仍会检查所有 `.mlir` 用例是否包含 `// RUN:` 标注，适合在轻量环境或 CI 中做基础验证。

## 质量检查

```bash
# 检查 Markdown 本地链接、关键项目目录、MLIR RUN 标注和用例数量
bash scripts/check-docs.sh
```

该检查不依赖 LLVM/MLIR 工具链；真正执行 lowering 仍需要安装 `mlir-opt`。

## 学完后你应该能做到

- 解释 AST、IR、SSA、Pass、Lowering、Dialect 的区别。
- 读懂一个简单 LLVM IR / MLIR 文件。
- 运行并修改一个 LLVM Pass 或 MLIR Pass。
- 写出一个 Triton kernel 并理解 block/grid/tile 三层编程模型。
- 说明 Triton kernel 如何编译为 TTIR → hivm.hir，最终在 Ascend NPU 上执行。
- 知道 Ascend NPU 的 gm/ub 内存层次和 Cube/Vector 计算单元如何影响 kernel 写法。
- 画出 AscendNPU-IR 中 Linalg → Husion → HIVM → LLVM/CANN 的大致链路。
- 理解为什么 NPU 编译器需要保留 matmul / conv 等高级语义，而不是过早展开为标量循环。

## 常用入口

- 快速入门：[docs/quickstart.md](docs/quickstart.md)
- 完整学习路径：[docs/triton-ascend/README.md](docs/triton-ascend/README.md)（含 7 个 Phase）
- 术语表：[docs/glossary.md](docs/glossary.md)
- 技术术语速查手册：[docs/reference/技术术语速查手册.md](docs/reference/技术术语速查手册.md)
- Triton-Ascend 动手：[projects/triton-ascend-lab/README.md](projects/triton-ascend-lab/README.md)
- 动手项目索引：[projects/README.md](projects/README.md)
- 综合 demo：[projects/ascendnpu-ir-demo/README.md](projects/ascendnpu-ir-demo/README.md)
- 历史总结：[SUMMARY.md](SUMMARY.md)

## 相关资源

### 官方仓库

| 仓库 | 说明 |
|------|------|
| [triton-lang/triton](https://github.com/triton-lang/triton) | Triton 官方仓库（GPU kernel 语言与编译器） |
| [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) | Triton Ascend 后端，对接昇腾 NPU |
| [Ascend/AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) | 华为昇腾 NPU MLIR 编译器项目 |
| [Ascend/triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels) | 极客营 Triton 算子开发仓库（Gitcode） |

### 官方文档

| 文档 | 链接 | 说明 |
|------|------|------|
| Triton 语言文档 | https://triton-lang.org/ | Triton API、编程模型、tutorials |
| MLIR 文档 | https://mlir.llvm.org/docs/ | Dialect、Pass、Pattern 体系 |
| LLVM 文档 | https://llvm.org/docs/ | LLVM IR、Pass 编写指南 |
| 昇腾社区 | https://www.hiascend.com/ | Ascend NPU 硬件文档、CANN 软件栈 |
| AscendC 文档 | https://www.hiascend.com/document?tag=ascendc | AscendC 算子开发语言文档 |

### 官方教程

| 教程 | 链接 | 说明 |
|------|------|------|
| Triton 官方 Tutorials | https://triton-lang.org/main/getting-started/tutorials/ | vector_add → matmul → fused attention |
| MLIR Toy Tutorial | https://mlir.llvm.org/docs/Tutorials/Toy/ | 官方 MLIR 入门教程（Ch1-7） |
| LLVM Kaleidoscope | https://llvm.org/docs/tutorial/ | LLVM 官方语言实现教程 |

## 许可证

本项目使用 [MIT License](LICENSE)。
