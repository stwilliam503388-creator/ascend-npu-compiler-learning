# 术语表

编译器领域术语密集。遇到不认识的词，回到这里查。

> 初次出现的术语在中英对照表中列出，核心概念附一句话解释。

---

## 核心概念

| 术语 | 英文 | 一句话解释 | 在哪学 |
|------|------|-----------|--------|
| 编译器 | Compiler | 把源代码翻译成机器码的程序 | [primer/00](./primer/00-编译器是什么.md) |
| 解释器 | Interpreter | 逐行执行源代码，不生成机器码 | [primer/00](./primer/00-编译器是什么.md) |
| 前端 | Frontend | 编译器的"入口"，解析源码生成 AST | [primer/01](./primer/01-AST与IR.md) |
| 后端 | Backend | 编译器的"出口"，把 IR 变成机器码 | [primer/00](./primer/00-编译器是什么.md) |
| AST | Abstract Syntax Tree | 抽象语法树，源码的树形表示 | [primer/01](./primer/01-AST与IR.md) |
| IR | Intermediate Representation | 中间表示，前端和后端之间的"通用语言" | [primer/01](./primer/01-AST与IR.md) |
| SSA | Static Single Assignment | 静态单赋值：每个变量只赋值一次 | [llvm/01](./llvm/01-LLVM-IR快速入门.md) |
| 基本块 | Basic Block | 连续执行的直线代码，一个入口一个出口 | [llvm/01](./llvm/01-LLVM-IR快速入门.md) |
| phi 节点 | Phi Node | 多路汇合处的选择器，根据来路选值 | [llvm/01](./llvm/01-LLVM-IR快速入门.md) |
| Pass | Pass | 对 IR 做一次检查或变换的"质检员" | [primer/02](./primer/02-Pass与Lowering.md) |
| Lowering | Lowering | 把高级 IR 逐步"降级"到底层 IR 的过程 | [primer/02](./primer/02-Pass与Lowering.md) |

## LLVM 工具

| 工具 | 全称 | 作用 | 在哪学 |
|------|------|------|--------|
| clang | C language | C/C++ 编译器前端 | [llvm/03](./llvm/03-LLVM工具箱速览.md) |
| opt | Optimizer | IR 优化器，加载 Pass 运行 | [llvm/02](./llvm/02-第一个LLVM-Pass.md) |
| llc | LLVM Compiler | IR → 汇编代码 | [llvm/03](./llvm/03-LLVM工具箱速览.md) |
| lli | LLVM Interpreter | 直接运行 IR（不编译） | [llvm/03](./llvm/03-LLVM工具箱速览.md) |
| llvm-dis | LLVM Disassembler | 二进制 IR → 可读文本 | [llvm/03](./llvm/03-LLVM工具箱速览.md) |

## 文件格式

| 后缀 | 含义 | 能直接读吗 |
|------|------|-----------|
| .c / .cpp | C/C++ 源码 | ✅ |
| .ll | LLVM IR 文本格式 | ✅ |
| .bc | LLVM IR 二进制格式 | ❌ 需用 llvm-dis 转 |
| .s | 汇编代码 | 勉强 |
| .o | 目标文件 | ❌ |

## Ascend 相关

| 术语 | 全称 | 一句话解释 |
|------|------|-----------|
| NPU | Neural Processing Unit | 神经网络处理器，华为 Ascend 的芯片类型 |
| CANN | Compute Architecture for Neural Networks | 华为 Ascend 的软件栈 |
| TBE | Tensor Boost Engine | CANN 中的算子开发框架 |
| Ascend C | — | 华为 NPU 的 C++ 扩展编程语言 |
| gm | Global Memory | HBM 全局显存，容量大（32GB+）但访问慢 |
| ub | Unified Buffer | L1 统一缓存，容量小（1MB）但访问快 |
| Cube Unit | — | Da Vinci 架构的矩阵乘法单元（16×16） |
| Vector Unit | — | Da Vinci 架构的向量运算单元 |
| SIMT | Single Instruction Multiple Threads | 单指令多线程模式，适合分支多的逻辑 |
| SIMD | Single Instruction Multiple Data | 单指令多数据模式，适合规整的向量运算 |

## Triton-Ascend 相关

| 术语 | 全称 | 一句话解释 | 在哪学 |
|------|------|-----------|--------|
| Triton | — | OpenAI 开源 GPU/NPU kernel 编程语言（Python DSL） | [triton-ascend/01](./triton-ascend/01-Triton编程模型与Ascend NPU.md) |
| kernel | — | 在 GPU/NPU 上运行的核函数（`@triton.jit` 修饰） | [triton-ascend/01](./triton-ascend/01-Triton编程模型与Ascend NPU.md) |
| tile | — | 数据切分块，每个 program 处理一个 tile | [triton-ascend/01](./triton-ascend/01-Triton编程模型与Ascend NPU.md) |
| grid | — | 启动配置，决定将任务分割成多少个 program | [triton-ascend/01](./triton-ascend/01-Triton编程模型与Ascend NPU.md) |
| program | — | grid 中的一个并行任务单元 | [triton-ascend/01](./triton-ascend/01-Triton编程模型与Ascend NPU.md) |
| TTIR | Triton IR | Triton 编译产出的 MLIR 方言 | [triton-ascend/02](./triton-ascend/02-Triton到Ascend全链路实战.md) |
| hivm.hir | Huawei Instruction Virtual Machine | Ascend 华为虚拟指令集方言 | [triton-ascend/02](./triton-ascend/02-Triton到Ascend全链路实战.md) |
| 算子融合 | Operator Fusion | 多个操作合并为一个 kernel，减少 gm↔ub 数据搬运 | [triton-ascend/03](./triton-ascend/03-Ascend NPU编程模式.md) |
| 归约 | Reduction | 跨元素聚合操作（如 sum、max） | [triton-ascend/03](./triton-ascend/03-Ascend NPU编程模式.md) |
| 逐元素 | Element-wise | 每个输出元素只依赖同位置输入的运算 | [triton-ascend/03](./triton-ascend/03-Ascend NPU编程模式.md) |
