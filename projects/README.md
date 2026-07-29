# 动手项目

本目录包含与学习路线各阶段配套的实践项目。建议按“可直接运行 → 需要 LLVM/MLIR → 需要 AscendNPU-IR”的顺序学习。

## 项目总览

| 项目 | 难度 | 阶段 | 环境要求 | 说明 | 验证方式 |
|---|---:|---|---|---|---|
| [hello-pass](./hello-pass/) | ⭐ | LLVM 入门 | LLVM `opt` / CMake | 第一个 LLVM FunctionPass：打印函数信息 | `cd hello-pass && ./run.sh` |
| [opt-pass](./opt-pass/) | ⭐⭐ | LLVM 进阶 | LLVM `opt` / CMake | 死代码消除 Pass，展示如何修改 IR | `cd opt-pass && ./run.sh` |
| [mlir-hello](./mlir-hello/) | ⭐ | MLIR 入门 | MLIR 库 / clang++ | 45 行独立程序，遍历 `func.func` | `cd mlir-hello && ./run.sh` |
| [triton-ascend-lab](./triton-ascend-lab/) | ⭐ | Triton-Ascend | `pip install triton` | 4 个递增 Triton kernel，从 vector_add 到 fused_layernorm | `cd triton-ascend-lab && bash run-all.sh` |
| [toy-mini](./toy-mini/) | ⭐ | 编译器前端 | C++17 编译器 | 纯 C++17 Toy 解析器和 MLIR 风格文本输出 | `g++ -std=c++17 -o toymini toymini.cpp` |
| [standalone-mlir](./standalone-mlir/) | ⭐⭐⭐ | MLIR 工程 | LLVM/MLIR + CMake 或 Make | 自定义 Dialect、TableGen、Pass、`standalone-opt` | `cmake --build build` 或 `make` |
| [ascendnpu-ir-op-counter](./ascendnpu-ir-op-counter/) | ⭐⭐⭐ | Ascend Pass | AscendNPU-IR 源码环境 | 分析 Pass 和转换 Pass 参考代码 | 放入 AscendNPU-IR 工程中构建 |
| [ascend-samples](./ascend-samples/) | ⭐⭐ | Ascend 对照 | 可选：自建 `bishengir-opt` | 5 个 Ascend Lowering 精选用例，含 input/expected 解读 | 阅读对照或运行 `bishengir-opt` |
| [ascendnpu-ir-demo](./ascendnpu-ir-demo/) | ⭐⭐ | 综合 demo | 可选：`mlir-opt` | 31 个 MLIR 用例、28 个 Triton 对照、4 种 matmul 优化方案 | `cd ascendnpu-ir-demo && bash run-tests.sh` |

## 推荐顺序

```text
hello-pass → opt-pass → mlir-hello → toy-mini
          → standalone-mlir → triton-ascend-lab → ascendnpu-ir-demo
          → ascend-samples / ascendnpu-ir-op-counter
```

## 分类说明

### 可直接建立直觉

- `hello-pass`：理解 Pass 是“遍历 IR 并做事”。
- `opt-pass`：理解 Pass 可以修改 IR。
- `toy-mini`：理解源码如何变成 AST，再输出 IR 风格文本。
- `triton-ascend-lab`：理解 Triton kernel 的 tile-based 编程模型，无需 GPU/NPU 硬件。

### 需要 LLVM/MLIR 环境

- `mlir-hello`：从 LLVM Pass 迁移到 MLIR Pass。
- `standalone-mlir`：理解 Dialect / ODS / TableGen / Pass 注册。
- `ascendnpu-ir-demo`：用标准 MLIR pass 模拟 AscendNPU-IR lowering。

### 需要或对照 AscendNPU-IR

- `ascend-samples`：从真实 AscendNPU-IR 语义抽取的阅读用例。
- `ascendnpu-ir-op-counter`：面向 AscendNPU-IR 工程的 Pass 参考实现。

## 新增项目约定

- 提供最小、直接的功能复现。
- 保留 `README.md`，说明目标、环境、运行方式和预期输出。
- 能一键运行的项目优先提供 `run.sh`。
- 复杂环境依赖必须说明“无依赖可阅读什么，有依赖可运行什么”。
