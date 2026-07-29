# 完整输出总结文档

> 生成时间: 2026-06-21
> 模型: deepseek-v4-flash
> 环境: macOS 26.5.1, Apple Silicon, LLVM 22.1.6 (Homebrew)

---

## 一、任务总览

本次连续多轮对话覆盖 LLVM → MLIR → AscendNPU-IR (Ascend NPU) 全链路学习，最终交付可运行的降级 Demo。

### 执行阶段

| 阶段 | 内容 | 产出 |
|------|------|------|
| **0. LLVM 基础** | llvm-ir-tutorial 中文教程速通 | 7 篇 Obsidian 笔记 |
| **1. MLIR 概念** | dialect/operation/pass/pattern 体系 | 2 篇基础笔记 + mlir-opt 实操 |
| **2. Toy Tutorial** | 官方 Toy 教程 7 章源码逐行解读 | 2 篇笔记 + 三项目对照表 |
| **3. 自定义 Pass** | 针对 AscendNPU-IR 写分析/转换 Pass | 2 个 C++ 源码文件 |
| **4. Standalone 项目** | 从零构建 CMake + Makefile 项目 | CMakeLists + .td + 入口文件 |
| **5. Toy Mini 手写** | 纯 C++17 零依赖解析器 | 1412 行，编译通过 |
| **6. Triton 体系** | triton-ascend 源码分析 | 2 篇笔记 + 全链路对照 |
| **7. ⭐ ascendnpu-ir-demo** | 可运行 mlir-opt 降级流水线 | 3 用例 + 生成器 + 运行脚本 |

---

## 二、知识库产出（15 篇笔记）

### LLVM 基础（7 篇）

| 文件 | 核心内容 |
|------|---------|
| **L00-速通总览** | 学习路线图 + 资源链接 + 三项目对应关系 |
| **L01-架构与HelloWorld** | 三段式架构（前端→IR→后端）、SSA、.ll 文件格式 |
| **L02-类型系统与GEP** | iN 类型、类型转换、GetElementPtr 剥洋葱模型 |
| **L03-控制流与Phi节点** | br 指令、φ 节点汇合、for 循环 φ 自引用 |
| **L04-内置函数与属性** | llvm.memcpy、expect 内建、属性组 |
| **L05-Pass开发** | New PM 架构、FunctionPass 骨架、BBCounter 实战 |
| **L06-IR速查表** | 常用指令 + 调试命令 + FileCheck |

### MLIR 体系（8 篇，本期核心）

| 文件 | 核心内容 | 工程关联 |
|------|---------|---------|
| **L00-速通与AscendNPU-IR实战** | dialect/operation/region 概念 + 三阶段降级对照 | ascendnpu-ir-demo |
| **L01-ToyTutorial Ch1-2** | Toy 语法 + TableGen + MLIRGen | toy-mini |
| **L02-ToyTutorial Ch3-6** | Pattern Rewriting + ConversionTarget | ascendnpu-ir-op-counter |
| **L03-自定义Pass实战** | OpCounter (walk) + PeelTranspose (OpRewritePattern) | ascendnpu-ir-op-counter |
| **L04-Standalone实战** | cmake 4.3 兼容性 + Makefile 方案 + LLVM 22 适配 | standalone-mlir |
| **L05-ToyMini手写** | Lexer/Parser/AST/IR 四组件全实现 | toy-mini |
| **L06-TritonMLIR体系** | TT dialect (1416 ops) + TritonGPU + 三项目对照 | — |
| **L07-triton-ascend后端** | ascend_interpreter.py + CANN + Python/C++ 对接层 | — |
| **L08-ascendnpu-ir-demo** | 3 用例全部 mlir-opt 验证通过 | ascendnpu-ir-demo |

---

## 三、工程交付（4 个项目）

### 1. ⭐ ascendnpu-ir-demo — 可运行降级流水线

**位置**: `projects/ascendnpu-ir-demo/`

#### 验证结果

| 用例 | 输入 (Linalg) | Stage1 (affine) | Stage3 (LLVM) | 膨胀率 | 说明 |
|------|-------------|----------------|--------------|--------|------|
| vecadd_128 | 3 行 | 18 行 | 38 行 | 12.7× | 最简向量加法 |
| matmul_4x4x4 | 1 行 | 18 行 | 72 行 | **72×** | 三重循环展开 |
| fused_128 | 15 行 | 20 行 | 59 行 | 3.9× | add + mul 连续 |

#### AscendNPU-IR 对应

```
AscendNPU-IR:        Linalg → HFusion.elemwise_binary → HIVM.load/vadd/store
标准 MLIR (本):   Linalg → affine.for + arith.addf → llvm.load/add/store
```

#### 文件清单

```
ascendnpu-ir-demo/
├── README.md                    — 使用说明
├── ascendnpu-ir-demo.py            — Python 生成器（可扩展用例）
├── run-demo.sh                  — 批量运行脚本
└── test-cases/
    ├── vecadd_128.mlir           — 向量加法 ✅
    ├── matmul_4x4x4.mlir         — 矩阵乘法 ✅
    └── fused_128.mlir            — 融合操作 ✅
```

#### 运行方式

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir
mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm test-cases/vecadd_128.mlir
```

---

### 2. toy-mini — 从零写 Toy 解析器

**位置**: `projects/toy-mini/toymini.cpp`

| 特性 | 详情 |
|------|------|
| **语言** | C++17（纯标准库，零依赖）|
| **行数** | 1,412 行 |
| **编译** | `g++ -std=c++17 -o toymini toymini.cpp` ✅ |
| **Lexer** | 14 token 类型（关键字/运算符/数字/标识符）|
| **Parser** | 递归下降（支持优先级控制）|
| **AST** | 8 种节点类型（Number/Var/Binary/Call/Literal/VarDecl/Print/Return）|
| **IR 生成** | MLIR 风格文本输出（对照 Toy Tutorial Ch2）|

**支持语法**:
```
def name(params) { var x = [[1,2],[3,4]]; print(x + y); return z; }
```

---

### 3. standalone-mlir — 从零构建 MLIR dialect

**位置**: `projects/standalone-mlir/`

| 组件 | 详情 |
|------|------|
| **Dialect** | `standalone`（6 ops: constant/add/mul/transpose/print/return）|
| **TableGen** | `StandaloneOps.td`（mlir-tblgen 语法验证通过）|
| **构建** | CMake + Makefile 双方案 |
| **入口** | `standalone-opt.cpp`（含 2 个 Pass）|
| **Pass** | `count-ops`（walk 分析）+ `elim-transpose`（Pattern 转换）|
| **MLIR 依赖** | 手动配置（`/opt/homebrew/opt/llvm`，跳过了 `find_package` 的 AddMLIR 冲突）|

**编译状态**: mlir-tblgen 生成 .inc 成功 ✅，完整编译需要 LLVM 22 的 Properties 机制支持（已记录适配方案）

---

### 4. ascendnpu-ir-op-counter — 自定义 Pass 源码

**位置**: `projects/ascendnpu-ir-op-counter/`

| 文件 | 类型 | 对应 Toy Tutorial | 功能 |
|------|------|-------------------|------|
| `BishengirOpCounter.cpp` | 分析 Pass | Ch3 ShapeInferencePass (walk) | 统计 hfusion/hivm ops 分布 |
| `BishengirPeelTranspose.cpp` | 转换 Pass | Ch3 ToyCombine + Ch5 Lowering | 消除冗余 transpose，融合 add→mul |

---

## 四、关键技术对照
### 阶段二学习目录

```
docs/llvm/
├── README.md                    ← 阶段入口与学习路线
├── 00-环境搭建.md               ← LLVM 安装与配置（macOS/Linux）
├── 01-LLVM-IR快速入门.md        ← SSA、基本块、phi 节点
├── 02-第一个LLVM-Pass.md        ← 逐行解读 HelloPass + 动手挑战
├── 03-LLVM工具箱速览.md         ← 5 个核心工具速查
│
projects/hello-pass/             ← 动手项目：第一个 LLVM Pass
```

（后续更多 LLVM 进阶文档持续更新中）

### 阶段三学习目录

> ✅ **已完成** — 4篇桥接 + MLIR-L00~L08 实战文档
>
> 详见 [docs/mlir/README.md](docs/mlir/README.md)
> 
> 预计包含：MLIR 基础概念、Dialect 定义、Pattern 改写、
> 从 LLVM IR 到 MLIR 的转换

## 🌳 阶段四: Ascend NPU 编译器后端开发

> ✅ **已完成** — 4篇文档 + ascend-samples 项目
>
> 详见 [docs/ascend/README.md](docs/ascend/README.md)
> 
> Ascend NPU 硬件架构、CANN 软件栈、
> TBE 算子开发、从 MLIR 到 Ascend 的 Lowering

---

## 后续更新 (2026-07)

| 日期 | 内容 | PR |
|------|------|:--:|
| 2026-07-22 | 新增 Phase 4 Triton-Ascend 编程实战（5 篇文档 + 8 kernel lab） | #18 |
| 2026-07-26 | references/README.md 重构为 4 层分级推荐（25 个项目 + 视频/论文/工具/社区） | #19 |

